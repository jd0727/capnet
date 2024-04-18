from data.processing import img2piece_persize, ExtractorIndex, piece_merge_img, decode_meta_xyxy, ExtractorClip, \
    img2piece_pyramid
from models.modules import C
from tools import TIMENODE, PERIOD
from utils import *


# <editor-fold desc='可推理模型'>

class IndependentInferableModel(InferableModel):
    PERIOD_PAIRS = (
        ('Time', PERIOD.ITER),
        ('data', PERIOD.LOAD),
        ('infr', PERIOD.INFER),
    )

    def act_init_infer(self, container, **kwargs):
        container.update_periods(IndependentInferableModel.PERIOD_PAIRS)
        return self

    def act_iter_infer(self, container, imgs, labels, **kwargs):
        container.update_time(TIMENODE.BEFORE_INFER)
        if isinstance(self, nn.Module):
            self.eval()
        labels_md = self.imgs2labels(imgs, cind2name=container.cind2name, **kwargs)
        container.update_time(TIMENODE.AFTER_INFER)
        return labels_md

    @abstractmethod
    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        pass


class SurpervisedInferable(InferableModel):
    PERIOD_PAIRS = IndependentInferableModel.PERIOD_PAIRS

    def act_init_infer(self, container, **kwargs):
        container.update_periods(SurpervisedInferable.PERIOD_PAIRS)
        return self

    @abstractmethod
    def imgs_labels2labels(self, imgs, labels, cind2name=None, **kwargs):
        pass

    def act_iter_infer(self, container, imgs, labels, **kwargs):
        container.update_time(TIMENODE.BEFORE_INFER)
        if isinstance(self, nn.Module):
            self.eval()
        labels_md = self.imgs_labels2labels(imgs, labels, cind2name=container.cind2name, **kwargs)
        container.update_time(TIMENODE.AFTER_INFER)
        container.update_periods(IndependentInferableModel.PERIOD_PAIRS)
        return labels_md


# </editor-fold>

# <editor-fold desc='单阶段模型'>

class OneBackwardTrainableModel(TrainableModel):
    PERIOD_PAIRS = (
        ('Time', PERIOD.ITER),
        ('data', PERIOD.LOAD),
        ('tar', PERIOD.TARGET),
        ('frwd', PERIOD.FORWARD),
        ('bkwd', PERIOD.BACKWARD),
        ('optm', PERIOD.OPTIMIZE),
    )

    def act_init_train(self, trainer, **kwargs):
        trainer.update_periods(OneBackwardTrainableModel.PERIOD_PAIRS)
        return self

    @abstractmethod
    def imgs_tars2loss(self, imgs, targets, **kwargs):
        pass

    def act_iter_train(self, trainer, imgs, targets, **kwargs):
        trainer.update_time(TIMENODE.BEFORE_FORWARD)
        self.train()
        with torch.cuda.amp.autocast(enabled=trainer.scaler.is_enabled()):
            loss = self.imgs_tars2loss(imgs, targets)
        trainer.update_time(TIMENODE.AFTER_FORWARD, TIMENODE.BEFORE_BACKWARD)
        trainer.loss_backward(loss, name='Loss')
        trainer.update_time(TIMENODE.AFTER_BACKWARD, TIMENODE.BEFORE_OPTIMIZE)
        trainer.optimizer_step()
        trainer.optimizer_zero_grad()
        trainer.update_time(TIMENODE.AFTER_OPTIMIZE)


class OneStageTorchModel(OneBackwardTrainableModel):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(OneStageTorchModel, self).__init__(backbone=backbone, device=device, pack=pack)

    def export_onnx(self, onnx_pth, batch_size=1):
        W, H = self.img_size
        model2onnx(self.backbone, onnx_pth, input_size=(batch_size, 3, H, W))
        return True

    def export_onnx_trt(self, onnx_pth, trt_pth, batch_size=1):
        # W, homography = self.norm
        # from deploy.onnx import model2onnx
        # from deploy.trt import onnx2trt
        # model2onnx(self.backbone, onnx_pth, input_size=(batch_size, 3, homography, W))
        # onnx2trt(onnx_pth=onnx_pth, trt_pth=trt_pth, max_batch=4, min_batch=1, std_batch=2)
        return True


# </editor-fold>


# <editor-fold desc='VAE'>

def _norm_kl(aver, lg_std):
    kl = torch.mean(aver ** 2 + torch.exp(lg_std) ** 2 - 1 - 2 * lg_std, dim=-1)
    kl = torch.mean(kl)
    return kl


def _norm_reparameterize(aver, lg_std):
    std = torch.exp(lg_std)
    noise = torch.randn_like(lg_std, device=lg_std.device)
    return aver + noise * std


class AE(OneBackwardTrainableModel):

    def __init__(self, encoder, decoder, device=None, pack=None, img_size=(224, 224)):
        super(AE, self).__init__(encoder=encoder, decoder=decoder, device=device, pack=pack)
        self._img_size = img_size

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        self.train()
        latvec = self.pkd_modules['encoder'](imgs)
        fimgs = torch.sigmoid(self.pkd_modules['decoder'](latvec))
        rec_loss = F.mse_loss(imgs, fimgs, reduction='mean')
        return rec_loss

    @property
    def img_size(self):
        return self._img_size

    def labels2tars(self, labels, **kwargs):
        targets = []
        for label in labels:
            assert isinstance(label, CategoryLabel), 'class err ' + label.__class__.__name__
            targets.append(OneHotCategory.convert(label.category).chotN)
        targets = np.array(targets)
        return targets

    def export_onnx_trt(self, **kwargs):
        pass

    @torch.no_grad()
    def reconst_fimgs(self, imgs, **kwargs):
        self.eval()
        imgsT, _ = imgs2imgsT(imgs, device=self.device, img_size=self.img_size)
        latvec = self.pkd_modules['encoder'](imgsT)
        fimgs = torch.sigmoid(self.pkd_modules['decoder'](latvec))
        return fimgs


class VAE(OneBackwardTrainableModel):

    def __init__(self, encoder, decoder, device=None, pack=None, features=50,
                 img_size=(224, 224)):
        super(VAE, self).__init__(encoder=encoder, decoder=decoder, device=device, pack=pack)
        self._img_size = img_size
        self.features = features

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        self.train()
        latvec = self.pkd_modules['encoder'](imgs)
        aver, lg_std = latvec.split((self.features, self.features), dim=1, )
        noise = _norm_reparameterize(aver, lg_std)
        fimgs = torch.sigmoid(self.pkd_modules['decoder'](noise))

        # kl_loss = _norm_kl(aver, lg_std)
        rec_loss = F.smooth_l1_loss(imgs, fimgs, beta=0.2, reduction='mean')
        return OrderedDict(rec=rec_loss)

    @property
    def img_size(self):
        return self._img_size

    def labels2tars(self, labels, **kwargs):
        targets = []
        for label in labels:
            assert isinstance(label, CategoryLabel), 'class err ' + label.__class__.__name__
            targets.append(OneHotCategory.convert(label.category).chotN)
        targets = np.array(targets)
        return targets

    def export_onnx_trt(self, **kwargs):
        pass

    @torch.no_grad()
    def reconst_fimgs(self, imgs, **kwargs):
        self.eval()
        imgsT, _ = imgs2imgsT(imgs, device=self.device, img_size=self.img_size)
        latvec = self.pkd_modules['encoder'](imgsT)
        aver, lg_std = latvec.split((self.features, self.features), dim=1, )
        noise = _norm_reparameterize(aver, lg_std)
        print(aver, lg_std)
        fimgs = torch.sigmoid(self.pkd_modules['decoder'](noise))
        return fimgs


# </editor-fold>


# <editor-fold desc='GAN'>

class GANTrainable(TrainableModel):
    PERIOD_PAIRS = (
        ('Time', PERIOD.ITER),
        ('data', PERIOD.LOAD),
        ('Gfwd', PERIOD.FORWARD_GEN),
        ('Dfwd', PERIOD.FORWARD_DIS),
        ('Gbwd', PERIOD.BACKWARD_GEN),
        ('DBwd', PERIOD.BACKWARD_DIS),
        ('Gopt', PERIOD.OPTIMIZE_GEN),
        ('Dopt', PERIOD.OPTIMIZE_DIS),
    )

    def act_init_train(self, trainer, **kwargs):
        trainer.update_periods(GANTrainable.PERIOD_PAIRS)
        return self

    def __init__(self, generator, discriminator, device=None, pack=None, generate_repeat=5):
        MTorchModel.__init__(self, generator=generator, discriminator=discriminator, device=device, pack=pack)
        self.generate_repeat = generate_repeat

    @abstractmethod
    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        pass

    @abstractmethod
    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        pass

    def act_iter_train(self, trainer, imgs, targets, **kwargs):
        self.train()
        while True:
            trainer.update_time(TIMENODE.BEFORE_FORWARD_GEN)
            gloss = self.imgs_tars2gen_loss(imgs, targets)
            trainer.update_time(TIMENODE.AFTER_FORWARD_GEN, TIMENODE.BEFORE_BACKWARD_GEN)
            trainer.loss_backward(gloss, name='GLoss')
            trainer.update_time(TIMENODE.AFTER_BACKWARD_GEN, TIMENODE.BEFORE_OPTIMIZE_GEN)
            trainer.optimizer_step(name='generator')
            trainer.optimizer_zero_grad()
            trainer.update_time(TIMENODE.AFTER_OPTIMIZE_GEN)
            # print(gloss.item())
            if gloss.item() < 0:
                break

        while True:
            trainer.update_time(TIMENODE.BEFORE_FORWARD_DIS)
            dloss = self.imgs_tars2dis_loss(imgs, targets)
            trainer.update_time(TIMENODE.AFTER_FORWARD_DIS, TIMENODE.BEFORE_BACKWARD_DIS)
            trainer.loss_backward(dloss, name='DLoss')
            trainer.update_time(TIMENODE.AFTER_BACKWARD_DIS, TIMENODE.BEFORE_OPTIMIZE_DIS)
            trainer.optimizer_step(name='discriminator')
            trainer.optimizer_zero_grad()
            trainer.update_time(TIMENODE.AFTER_OPTIMIZE_DIS)
            # print(dloss.item())
            if dloss.item() < 0:
                break

    def export_onnx_trt(self, **kwargs):
        pass


class FakeImageBuffer():
    def __init__(self, buffer_size=256):
        self.buffer_size = buffer_size
        self._fimgs = None

    def _update_fimgs(self, fimgs):
        fimgs = fimgs.detach()
        if self.buffer_size is None:
            self._fimgs = fimgs
        else:
            if self._fimgs is None:
                self._fimgs = fimgs
            else:
                inds = torch.randperm(self._fimgs.size(0), device=self._fimgs.device)
                self._fimgs = torch.cat([fimgs, self._fimgs[inds]], dim=0)
                self._fimgs = self._fimgs[:self.buffer_size]

    def _fetch_fimgs(self, num_fetch):
        if self.buffer_size is None:
            return self._fimgs
        else:
            if self._fimgs is None:
                return None
            else:
                inds = torch.randint(low=0, high=self._fimgs.size(0), size=(num_fetch,))
                return self._fimgs[inds]


class GAN(GANTrainable, FakeImageBuffer):
    def __init__(self, generator, discriminator, device=None, pack=None, features=50,
                 img_size=(224, 224), generate_repeat=5, buffer_size=512):
        FakeImageBuffer.__init__(self, buffer_size=buffer_size)
        GANTrainable.__init__(self, generator=generator, discriminator=discriminator, device=device, pack=pack,
                              generate_repeat=generate_repeat)
        self._img_size = img_size
        self.features = features

    @property
    def img_size(self):
        return self._img_size

    def get_noise(self, batch_size):
        noise = torch.randn(size=(batch_size, self.features), device=self.device)
        return noise

    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        self.train()
        noise = self.get_noise(imgs.size(0))
        fimgs = torch.sigmoid(self.pkd_modules['generator'](noise))
        self._update_fimgs(fimgs)
        preds = self.pkd_modules['discriminator'](fimgs)
        targets = torch.full_like(preds, fill_value=0.9, device=self.device)
        loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')
        return loss

    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        self.train()
        fimgs = self._fetch_fimgs(imgs.size(0))
        preds_real = self.pkd_modules['discriminator'](imgs)
        preds_fake = self.pkd_modules['discriminator'](fimgs)
        targets_real = torch.full_like(preds_real, fill_value=0.9, device=self.device)
        targets_fake = torch.full_like(preds_fake, fill_value=0.1, device=self.device)
        loss_real = F.binary_cross_entropy_with_logits(preds_real, targets_real, reduction='mean')
        loss_fake = F.binary_cross_entropy_with_logits(preds_fake, targets_fake, reduction='mean')
        return loss_real + loss_fake

    def labels2tars(self, labels, **kwargs):
        targets = []
        for label in labels:
            assert isinstance(label, CategoryLabel), 'class err ' + label.__class__.__name__
            targets.append(OneHotCategory.convert(label.category).chotN)
        targets = np.array(targets)
        return targets

    def export_onnx_trt(self, **kwargs):
        pass

    @torch.no_grad()
    def gen_fimgs(self, batch_size, with_discriminator=False, **kwargs):
        self.eval()
        noise = self.get_noise(batch_size)
        generator = self.pkd_modules['generator'].eval()
        fimgs = torch.sigmoid(generator(noise))
        if with_discriminator:
            preds = self.pkd_modules['discriminator'](fimgs)
            print(preds)
        return fimgs


class WeightClamper(object):
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def __call__(self, module):
        for name in ['weight', 'bias']:
            if hasattr(module, name):
                w = module.weight.data
                w = w.clamp(min=self.min, max=self.max)
                module.weight.data = w


class WGAN(GAN):
    def __init__(self, generator, discriminator, device=None, pack=None, features=50,
                 img_size=(224, 224), generate_repeat=5, buffer_size=512, discriminator_clip=0.1):
        super(WGAN, self).__init__(generator=generator, discriminator=discriminator, device=device, pack=pack,
                                   features=features, img_size=img_size, generate_repeat=generate_repeat,
                                   buffer_size=buffer_size)
        self.discriminator_clip = discriminator_clip

    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        self.train()
        noise = self.get_noise(imgs.size(0))
        fimgs = torch.sigmoid(self.pkd_modules['generator'](noise))
        self._update_fimgs(fimgs)
        preds = self.pkd_modules['discriminator'](fimgs)
        return - torch.mean(preds)

    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        self.train()
        fimgs = self._fetch_fimgs(imgs.size(0))
        discriminator = self.pkd_modules['discriminator']
        if self.discriminator_clip is not None and self.discriminator_clip > 0:
            discriminator.apply(WeightClamper(min=-self.discriminator_clip, max=self.discriminator_clip))
        preds_real = discriminator(imgs)
        preds_fake = discriminator(fimgs)
        return torch.mean(preds_fake) - torch.mean(preds_real)


class SpectralNormAdder(object):
    def __init__(self, ):
        pass

    def __call__(self, module):
        if isinstance(module, C):
            module.conv = nn.utils.spectral_norm(module.conv)


class SNGAN(GAN):
    def __init__(self, generator, discriminator, device=None, pack=None, features=50,
                 img_size=(224, 224), generate_repeat=5, buffer_size=512):
        discriminator.apply(SpectralNormAdder())
        super(SNGAN, self).__init__(generator=generator, discriminator=discriminator, device=device, pack=pack,
                                    features=features, img_size=img_size, generate_repeat=generate_repeat,
                                    buffer_size=buffer_size)

    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        self.train()
        noise = self.get_noise(imgs.size(0))
        fimgs = torch.sigmoid(self.pkd_modules['generator'](noise))
        self._update_fimgs(fimgs)
        preds = self.pkd_modules['discriminator'](fimgs)
        return - torch.mean(preds)

    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        self.train()
        fimgs = self._fetch_fimgs(imgs.size(0))
        discriminator = self.pkd_modules['discriminator']
        preds_real = discriminator(imgs)
        preds_fake = discriminator(fimgs)
        return torch.mean(preds_fake) - torch.mean(preds_real)


# </editor-fold>


# <editor-fold desc='GAN+VAE'>
class GANAETrainable(TrainableModel):

    def act_init_train(self, trainer, **kwargs):
        trainer.update_periods(GANTrainable.PERIOD_PAIRS)
        return self

    def __init__(self, encoder, decoder, discriminator, device=None, pack=None, generate_repeat=5):
        MTorchModel.__init__(self, encoder=encoder, decoder=decoder, discriminator=discriminator,
                             device=device, pack=pack)
        self.generate_repeat = generate_repeat

    @abstractmethod
    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        pass

    @abstractmethod
    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        pass

    def act_iter_train(self, trainer, imgs, targets, **kwargs):
        self.train()
        for i in range(self.generate_repeat):
            trainer.update_time(TIMENODE.BEFORE_FORWARD_GEN)
            gloss = self.imgs_tars2gen_loss(imgs, targets)
            trainer.update_time(TIMENODE.AFTER_FORWARD_GEN, TIMENODE.BEFORE_BACKWARD_GEN)
            trainer.loss_backward(gloss, name='GLoss')
            trainer.update_time(TIMENODE.AFTER_BACKWARD_GEN, TIMENODE.BEFORE_OPTIMIZE_GEN)
            trainer.optimizer_step(name='encoder')
            trainer.optimizer_step(name='decoder')
            trainer.optimizer_zero_grad()
            trainer.update_time(TIMENODE.AFTER_OPTIMIZE_GEN)

        trainer.update_time(TIMENODE.BEFORE_FORWARD_DIS)
        dloss = self.imgs_tars2dis_loss(imgs, targets)
        trainer.update_time(TIMENODE.AFTER_FORWARD_DIS, TIMENODE.BEFORE_BACKWARD_DIS)
        trainer.loss_backward(dloss, name='DLoss')
        trainer.update_time(TIMENODE.AFTER_BACKWARD_DIS, TIMENODE.BEFORE_OPTIMIZE_DIS)
        trainer.optimizer_step(name='discriminator')
        trainer.optimizer_zero_grad()
        trainer.update_time(TIMENODE.AFTER_OPTIMIZE_DIS)

    def export_onnx_trt(self, **kwargs):
        pass


class GANVAE(GANAETrainable):

    def __init__(self, encoder, decoder, discriminator, device=None, pack=None, features=50,
                 img_size=(224, 224), generate_repeat=1):
        super(GANVAE, self).__init__(encoder=encoder, decoder=decoder, discriminator=discriminator, device=device,
                                     pack=pack, generate_repeat=generate_repeat)
        self._img_size = img_size
        self.features = features
        self.fimgs = None

    @property
    def img_size(self):
        return self._img_size

    def get_noise(self, batch_size):
        noise = torch.randn(size=(batch_size, self.features), device=self.device)
        return noise

    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        self.train()
        latvec = self.pkd_modules['encoder'](imgs)
        aver, lg_std = latvec.split((self.features, self.features), dim=1, )
        noise = _norm_reparameterize(aver, lg_std)
        fimgs = torch.sigmoid(self.pkd_modules['decoder'](noise))

        self.fimgs = fimgs.detach()
        preds = self.pkd_modules['discriminator'](fimgs)
        targets = torch.full_like(preds, fill_value=0.9, device=self.device)
        gan_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')

        # kl_loss = norm_std_kl(aver, lg_std)
        rec_loss = F.l1_loss(imgs, fimgs, reduction='mean')
        return OrderedDict(gan=gan_loss, rec=rec_loss * 5)

    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        self.train()
        preds_real = self.pkd_modules['discriminator'](imgs)
        preds_fake = self.pkd_modules['discriminator'](self.fimgs)
        targets_real = torch.full_like(preds_real, fill_value=0.9, device=self.device)
        targets_fake = torch.full_like(preds_fake, fill_value=0.1, device=self.device)
        loss_real = F.binary_cross_entropy_with_logits(preds_real, targets_real, reduction='mean')
        loss_fake = F.binary_cross_entropy_with_logits(preds_fake, targets_fake, reduction='mean')
        return loss_real + loss_fake

    def labels2tars(self, labels, **kwargs):
        targets = []
        for label in labels:
            assert isinstance(label, CategoryLabel), 'class err ' + label.__class__.__name__
            targets.append(OneHotCategory.convert(label.category).chotN)
        targets = np.array(targets)
        return targets

    def export_onnx_trt(self, **kwargs):
        pass

    @torch.no_grad()
    def gen_fimgs(self, batch_size, **kwargs):
        self.eval()
        noise = self.get_noise(batch_size)
        generator = self.pkd_modules['decoder'].eval()
        fimgs = torch.sigmoid(generator(noise))
        return fimgs

    @torch.no_grad()
    def reconst_fimgs(self, imgs, **kwargs):
        self.eval()
        imgsT, _ = imgs2imgsT(imgs, device=self.device, img_size=self.img_size)
        latvec = self.pkd_modules['encoder'](imgsT)
        aver, lg_std = latvec.split((self.features, self.features), dim=1, )
        noise = _norm_reparameterize(aver, lg_std)
        print(aver, lg_std)
        fimgs = torch.sigmoid(self.pkd_modules['decoder'](noise))
        return fimgs


# </editor-fold>


# <editor-fold desc='分类'>
class OneStageClassifier(OneStageTorchModel, IndependentInferableModel):
    def __init__(self, backbone, device=None, pack=None, img_size=(224, 224), num_cls=10):
        super(OneStageClassifier, self).__init__(backbone=backbone, device=device, pack=pack)
        self.img_size = img_size
        self._num_cls = num_cls
        self.forward = self.imgs2labels

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        self.eval()
        imgs, _ = imgs2imgsT(imgs, img_size=self.img_size)
        _, _, H, W = imgs.size()
        chotsT = self.pkd_modules['backbone'](imgs.to(self.device))
        chotsT = torch.softmax(chotsT, dim=-1)
        cates = chotsT2cates(chotsT, img_size=(W, H), cind2name=cind2name)
        return cates

    def labels2tars(self, labels, **kwargs):
        targets = []
        for label in labels:
            assert isinstance(label, CategoryLabel), 'class err ' + label.__class__.__name__
            targets.append(OneHotCategory.convert(label.category).chotN)
        targets = np.array(targets)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        self.train()
        imgs = imgs.to(self.device)
        pred = self.pkd_modules['backbone'](imgs)
        target = torch.as_tensor(targets).to(pred.device, non_blocking=True)
        loss = F.cross_entropy(pred, target, reduction='mean')
        return loss

    def grad_cam(self, imgs, cindsN=None):
        imgs, _ = imgs2imgsT(imgs, img_size=self.img_size)
        hmaps = grad_cam(self.backbone, modules=(self.backbone.pool,), imgs=imgs.to(self.device), cindsN=cindsN)
        return hmaps

    def grad_cam_visual(self, imgs, cts, cls2name=None):
        clses = self.imgs2clses(imgs, cls2name=cls2name)
        hmap = self.grad_cam(imgs, clses)
        imgs = torch.cat([imgs, hmap], dim=1)
        mclses = []
        for cls, ct in zip(clses, cts):
            mcls = copy.deepcopy(cls)
            mcls_2 = copy.deepcopy(ct)
            mcls_2['tcls'] = mcls_2['chots_tg']
            del mcls_2['chots_tg']
            del mcls_2['name']
            mcls.update(mcls_2)
            mclses.append(mcls)
        return imgs, mclses


def fwd_hook(module, data_input, data_output, dist, ind):
    fwd_buffer = data_input[0].detach()
    dist[ind] = fwd_buffer
    return None


def bkwd_hook(module, grad_input, grad_output, dist, ind):
    bkwd_buffer = grad_input[0]
    dist[ind] = bkwd_buffer
    return None


def grad_cam(model, modules, imgs, cindsN=None):
    model.eval()
    model.zero_grad()
    # 确定层
    fwd_buffer = {}
    bkwd_buffer = {}

    fwd_handlers = []
    bkwd_handlers = []
    for ind, module in enumerate(modules):
        fwd_handler = module.register_forward_hook(partial(fwd_hook, dist=fwd_buffer, ind=ind))
        bkwd_handler = module.register_backward_hook(partial(bkwd_hook, dist=bkwd_buffer, ind=ind))
        fwd_handlers.append(fwd_handler)
        bkwd_handlers.append(bkwd_handler)

    # 传播
    imgs = imgs.to(next(iter(model.parameters())).device)
    chotsT = model(imgs)
    cindsT = torch.softmax(chotsT, dim=-1)

    mask = torch.zeros_like(cindsT).to(cindsT.device)
    if cindsN is None:
        ids = torch.argmax(cindsT, dim=-1)
        mask[torch.arange(cindsT.size(0)), ids] = 1
    else:
        mask[torch.arange(cindsT.size(0)), cindsN] = 1
    torch.sum(cindsT * mask).backward()

    # 画图
    hmaps = torch.zeros(imgs.size(0), 1, imgs.size(2), imgs.size(3))
    for ind in range(len(modules)):
        fwd_data = fwd_buffer[ind]
        bkwd_data = bkwd_buffer[ind]
        pows = bkwd_data.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        hmap_ind = torch.clamp(torch.sum(pows * fwd_data, dim=1, keepdim=True), min=0)
        hmap_ind = F.interpolate(input=hmap_ind, size=(imgs.size(2), imgs.size(3)), mode='bicubic',
                                 align_corners=True).detach().cpu()
        hmaps += hmap_ind

    # 移除hook
    for fwd_handler in fwd_handlers:
        fwd_handler.remove()
    for bkwd_handler in bkwd_handlers:
        bkwd_handler.remove()
    return hmaps


# </editor-fold>

# <editor-fold desc='语义分割'>
class OneStageSegmentor(OneStageTorchModel, IndependentInferableModel):
    def __init__(self, backbone, device=None, pack=None, num_cls=10, img_size=(128, 128)):
        super(OneStageSegmentor, self).__init__(backbone=backbone, device=device, pack=pack)
        self.forward = self.imgs2labels
        self._num_cls = num_cls
        self._img_size = img_size

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    @torch.no_grad()
    def imgs2labels(self, imgs, cind2name=None, conf_thres=0.4, sxy=40, srgb=10, num_infer=0, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        Nb, _, H, W = imgsT.size()
        maskssT = self.pkd_modules['backbone'](imgsT.to(self.device))
        maskssT = torch.softmax(maskssT, dim=1)
        labels = []
        for i, (masksT, imgT) in enumerate(zip(maskssT, imgsT)):
            if num_infer > 0:
                masksT = masksT_crf(imgT=imgT, masksT=masksT, sxy=sxy, srgb=srgb, num_infer=num_infer)
            segs = SegsLabel.from_masksT(masksT, num_cls=self.num_cls, conf_thres=conf_thres, cind2name=cind2name)
            labels.append(segs)
        return labels_rescale(labels, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        masks_tg = [np.zeros(shape=(0, self._img_size[1], self._img_size[0]), dtype=np.int32)]
        # time1=time.time()
        for label in labels:
            assert isinstance(label, RegionExportable), 'class err ' + label.__class__.__name__
            masksN = label.export_masksN_enc(img_size=self.img_size, num_cls=self.num_cls)
            masks_tg.append(masksN[None, ...])
        masks_tg = np.concatenate(masks_tg, axis=0, dtype=np.int32)
        # time2 = time.time()
        # print(time2-time1)
        return masks_tg

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        preds = self.pkd_modules['backbone'](imgs)

        masks_tg = torch.as_tensor(targets).to(preds.device, non_blocking=True).long()

        masks_tg_chot = torch.zeros(
            size=(masks_tg.size(0), self.num_cls + 1, masks_tg.size(1), masks_tg.size(2)),
            device=masks_tg.device)
        filler = torch.ones_like(masks_tg, device=masks_tg.device, dtype=torch.float32)
        masks_tg_chot.scatter_(dim=1, index=masks_tg[:, None, :, :], src=filler[:, None, :, :])
        loss = F.cross_entropy(preds, masks_tg_chot, reduction='mean')
        return loss


class OneStageBoxSupervisedSegmentor(OneStageTorchModel, SurpervisedInferable):
    def __init__(self, backbone, device=None, pack=None, num_cls=10, img_size=(128, 128)):
        self._num_cls = num_cls
        self._img_size = img_size
        super(OneStageBoxSupervisedSegmentor, self).__init__(backbone=backbone, device=device, pack=pack)
        self.forward = self.imgs_labels2labels

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    def labels2masks_attn(self, labels):
        masks_attn = [np.zeros(shape=(0, self.img_size[1], self.img_size[0]))]
        for label in labels:
            msk = label.export_border_masksN_enc(img_size=self.img_size, num_cls=self.num_cls)
            masks_attn.append(msk[None])
        masks_attn = np.concatenate(masks_attn, axis=0)
        masks_attn = (masks_attn < self.num_cls).astype(np.float32)[:, None, :, :]
        return masks_attn

    @torch.no_grad()
    def imgs_labels2labels(self, imgs, labels, cind2name=None, conf_thres=0.4, sxy=80, srgb=10, num_infer=0,
                           only_inner=True, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        labels = labels_rescale(labels, imgsT, ratios)
        imgsT = imgsT.to(self.device)
        masks_attn = self.labels2masks_attn(labels)
        masks_attnT = torch.from_numpy(masks_attn).to(self.device).float()
        Nb, _, H, W = imgsT.size()
        maskssT = self.pkd_modules['backbone'](imgsT, masks_attnT)
        maskssT = torch.softmax(maskssT, dim=1)
        labels_pd = []
        for i, (masksT, label, imgT) in enumerate(zip(maskssT, labels, imgsT)):
            if num_infer > 0:
                masksT = masksT_crf(imgT=imgT, masksT=masksT, sxy=sxy, srgb=srgb, num_infer=num_infer)
            insts = InstsLabel.from_boxes_masksT_abs(
                boxes=label, masksT=masksT, conf_thres=conf_thres, cind=0, only_inner=only_inner)
            labels_pd.append(insts)
        return labels_rescale(labels_pd, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        masks_attn = self.labels2masks_attn(labels)
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        xyxys_tg = [np.zeros(shape=(0, 4))]
        cinds_tg = [np.zeros(shape=(0,), dtype=np.int32)]
        for i, label in enumerate(labels):
            assert isinstance(label, BorderExportable) and isinstance(label, CategoryExportable)
            cinds = label.export_cindsN()
            xyxys = label.export_xyxysN()
            inds_b_pos.append(np.full(fill_value=i, shape=len(cinds)))
            xyxys_tg.append(xyxys)
            cinds_tg.append(cinds)
        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        xyxys_tg = np.concatenate(xyxys_tg, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        targets = (inds_b_pos, xyxys_tg, cinds_tg, masks_attn)
        return targets


# </editor-fold>

# <editor-fold desc='级联模型'>

def _label_merge(label, by_cls=True, urate_thres=0.8):
    if urate_thres <= 0:
        return label
    xyxys = label.export_xyxysN()
    cinds = label.export_cindsN() if by_cls else None
    pairs = mrg_xyxysN(xyxys, cindsN=cinds, opr_thres=urate_thres, opr_type=OPR_TYPE.URATE)
    label_mrgd = label.empty()
    for idx1, idx2 in pairs:
        xyxy_mrg = np.concatenate(
            [np.minimum(xyxys[idx1, :2], xyxys[idx2, :2]),
             np.maximum(xyxys[idx1, 2:4], xyxys[idx2, 2:4])], axis=0)
        item = copy.deepcopy(label[idx1])
        item.border = XYXYBorder(xyxy_mrg, item.border.size)
        label_mrgd.append(item)
    for i, item in enumerate(label):
        if i not in pairs:
            label_mrgd.append(item)
    return label_mrgd


def _feed_with_batch(imgs2labels, imgs, batch_size=32, **kwargs):
    labels_all = []
    while len(imgs) > batch_size:
        imgs_feed, imgs = imgs[:batch_size], imgs[batch_size:]
        labels_feed = imgs2labels(imgs_feed, **kwargs)
        labels_all.extend(labels_feed)
    return labels_all


class PatchDetector(IndependentInferableModel):

    def __init__(self, detector_base, img_size=(5000, 3000), piece_size=(2048, 2048), over_lap=(512, 512),
                 merge_iou_thres=0.5, merge_by_cls=True, merge_iou_type=IOU_TYPE.IRATE2, merge_urate_thres=0.9,
                 batch_size=32):
        self.detector_base = detector_base
        self._img_size = img_size
        self._piece_size = piece_size
        self._over_lap = over_lap
        self.merge_iou_thres = merge_iou_thres
        self.merge_by_cls = merge_by_cls
        self.merge_iou_type = merge_iou_type
        self.merge_urate_thres = merge_urate_thres
        self.batch_size = batch_size

    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        labels = []
        for img in imgs:
            if self.img_size is not None:
                img_scld, ratio = imgP_lmtsize_pad(img2imgP(img), max_size=self.img_size)
            else:
                img_scld, ratio = img, 1.0
            pieces, plabels_epty = img2piece_persize(
                img_scld, items=BoxesLabel([], img_size=self.img_size, meta=''), piece_size=self._piece_size,
                over_lap=self._over_lap,
                ignore_empty=False,
                with_clip=True, fltr=None, meta_encoder=None, extractor=ExtractorClip())
            xyxy_rgns = [decode_meta_xyxy(plabel_epty.meta)[1] for plabel_epty in plabels_epty]

            plabels = _feed_with_batch(self.detector_base.imgs2labels, pieces, batch_size=self.batch_size,
                                       cind2name=cind2name, **kwargs)
            label = piece_merge_img(plabels, xyxy_rgns=xyxy_rgns, iou_thres=self.merge_iou_thres,
                                    by_cls=self.merge_by_cls, meta=None, iou_type=self.merge_iou_type)
            label = BoxesLabel(label, img_size=img2size(img_scld), meta=None)
            label = _label_merge(label, by_cls=True, urate_thres=self.merge_urate_thres)
            label.linear_(scale=1 / np.array([ratio, ratio]), size=img2size(img))
            labels.append(label)
        return labels

    @property
    def piece_size(self):
        return self._piece_size

    @property
    def img_size(self):
        return self._img_size

    @property
    def img_size_base(self):
        return self.detector_base.img_size

    @property
    def num_cls(self):
        return self.detector_base.num_cls


class PyramidDetector(IndependentInferableModel):
    PIECE_SIZES_1024 = ((1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192))
    OVER_LAPS_1024 = ((256, 256), (512, 512), (1024, 1024), (0, 0),)

    PIECE_SIZES_640 = ((640, 640), (1280, 1280), (2560, 2560), (5120, 5120))
    OVER_LAPS_640 = ((160, 160), (320, 320), (640, 640), (0, 0),)

    def __init__(self, detector_base, img_size=(5000, 3000),
                 piece_sizes=PIECE_SIZES_1024,
                 over_laps=OVER_LAPS_1024,
                 merge_iou_thres=0.5, merge_by_cls=True, merge_iou_type=IOU_TYPE.IRATE2,
                 merge_urate_thres=0.9,
                 batch_size=32):
        self.detector_base = detector_base
        self._img_size = img_size
        self._piece_sizes = piece_sizes
        self._over_laps = over_laps
        self.merge_iou_thres = merge_iou_thres
        self.merge_by_cls = merge_by_cls
        self.merge_iou_type = merge_iou_type
        self.merge_urate_thres = merge_urate_thres
        self.batch_size = batch_size

    @torch.no_grad()
    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        labels = []
        for img in imgs:
            if self.img_size is not None:
                img_scld, ratio = imgP_lmtsize_pad(img2imgP(img), max_size=self.img_size)
                img_size = self.img_size
            else:
                img_scld, ratio = img, 1.0
                img_size = img2size(img)
            pieces, plabels_epty = img2piece_pyramid(
                img_scld, items=BoxesLabel([], img_size=img_size, meta=''), piece_sizes=self._piece_sizes,
                over_laps=self._over_laps,
                ignore_empty=False,
                with_clip=True, fltr=None, meta_encoder=None, extractor=ExtractorClip())
            xyxy_rgns = [decode_meta_xyxy(plabel_epty.meta)[1] for plabel_epty in plabels_epty]

            plabels = _feed_with_batch(self.detector_base.imgs2labels, pieces, batch_size=self.batch_size,
                                       cind2name=cind2name, **kwargs)
            label = piece_merge_img(plabels, xyxy_rgns=xyxy_rgns, iou_thres=self.merge_iou_thres,
                                    by_cls=self.merge_by_cls, meta=None, iou_type=self.merge_iou_type)
            label = BoxesLabel(label, img_size=img_size, meta=None)
            # 二次融合
            label = _label_merge(label, by_cls=True, urate_thres=self.merge_urate_thres)
            label.linear_(scale=1 / np.array([ratio, ratio]), size=img2size(img))
            labels.append(label)
        return labels

    @property
    def piece_sizes(self):
        return self._piece_sizes

    @property
    def img_size(self):
        return self._img_size

    @property
    def img_size_base(self):
        return self.detector_base.img_size

    @property
    def num_cls(self):
        return self.detector_base.num_cls


class CascadeDetector(IndependentInferableModel):

    def __init__(self, regionprop, classifier, expend_ratio=1.2, batch_size=32):
        self.regionprop = regionprop
        self.classifier = classifier
        self.expend_ratio = expend_ratio
        self.batch_size = batch_size

    @property
    def img_size(self):
        return self.regionprop.img_size

    @property
    def num_cls(self):
        return self.classifier.num_cls

    def imgs2labels(self, imgs, conf_thres=0.3, iou_thres=0.45, with_classifier=True, cind2name=None,
                    **kwargs):
        labels_rpn = self.regionprop.imgs2labels(imgs=imgs, conf_thres=conf_thres, iou_thres=iou_thres)
        if not with_classifier:
            return labels_rpn
        for img, label_rpn in zip(imgs, labels_rpn):
            if len(label_rpn) == 0:
                continue
            imgP = img2imgP(img)
            patchs = []
            for box in label_rpn:
                border_ext = copy.deepcopy(XYXYBorder.convert(box.border))
                border_ext.expend(self.expend_ratio).clip(xyxyN_rgn=np.array([0, 0, imgP.size[0], imgP.size[1]]))
                patchs.append(imgP.crop(border_ext.xyxyN.astype(np.int32)))
            cates = _feed_with_batch(self.classifier.imgs2labels, patchs, batch_size=self.batch_size,
                                     cind2name=cind2name, **kwargs)
            for cate, box in zip(cates, label_rpn):
                cate.category.conf_scale(box.category.conf)
                box.category = cate.category
                box.update(cate)
        return labels_rpn


class CascadeSegmentor(IndependentInferableModel):

    @property
    def img_size(self):
        return self.regionprop.img_size

    @property
    def num_cls(self):
        return self.segmentor.num_cls

    def __init__(self, regionprop, segmentor, expend_ratio=1.2, batch_size=32):
        self.regionprop = regionprop
        self.segmentor = segmentor
        self.expend_ratio = expend_ratio
        self.batch_size = batch_size

    def imgs2labels(self, imgs, conf_thres=0.3, iou_thres=0.45, with_segmentor=False, cind2name=None,
                    **kwargs):
        labels_rpn = self.regionprop.imgs2labels(imgs=imgs, conf_thres=conf_thres, iou_thres=iou_thres)
        if not with_segmentor:
            return labels_rpn
        labels = []
        for img, label_rpn in zip(imgs, labels_rpn):
            imgP = img2imgP(img)
            xyxys = []
            patchs = []
            for box in label_rpn:
                border_ext = copy.deepcopy(XYXYBorder.convert(box.border))
                border_ext.expend(self.expend_ratio).clip(xyxyN_rgn=np.array([0, 0, imgP.size[0], imgP.size[1]]))
                xyxy = border_ext.xyxyN.astype(np.int32)
                xyxys.append(xyxy)
                patchs.append(imgP.crop(xyxy))

            labels_seg = _feed_with_batch(self.segmentor.imgs2labels, patchs, batch_size=self.batch_size,
                                          cind2name=cind2name, **kwargs)
            label = SegsLabel(img_size=self.img_size)
            for label_seg, xyxy in zip(labels_seg, xyxys):
                for seg in label_seg:
                    seg.linear(bias=xyxy[:2], img_size=img2size(img))
                    label.append(seg)
                    labels.append(label)
        return labels


class LabelPostProcessor(IndependentInferableModel):
    def __init__(self, model, func=None):
        self.model = model
        self.func = func

    def imgs2labels(self, imgs, cind2name=None, **kwargs):
        labels = self.model.imgs2labels(imgs, cind2name=cind2name, **kwargs)
        if self.func is not None:
            for i in range(len(labels)):
                labels[i] = self.func(labels[i])
        return labels

    @property
    def img_size(self):
        return self.model.img_size

    @property
    def num_cls(self):
        return self.model.num_cls


# </editor-fold>

# <editor-fold desc='目标检测中心先验'>
class RadiusBasedCenterPrior():

    @property
    def num_oflb(self):
        return self.offset_lb.shape[0]

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius
        if self.radius < 1:
            self.offset_lb = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]) * radius
        else:
            offset_lb = np.stack(np.meshgrid(
                np.arange(np.floor(-radius), np.ceil(radius) + 1),
                np.arange(np.floor(-radius), np.ceil(radius) + 1)), axis=2).reshape(-1, 2).astype(np.int32)
            self.offset_lb = offset_lb[np.linalg.norm(offset_lb, axis=1) <= radius]


def matcher_cenpiror(xys_lb, layers, offset_lb):
    fltr_full = []
    ids_ancr = []
    offset_layer = 0
    for j, layer in enumerate(layers):
        stride, Wf, Hf = layer.stride, layer.Wf, layer.Hf
        ixys = (xys_lb[:, None, :] / stride + offset_lb).astype(np.int32)
        fltr_valid = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
        ids_ancr_ly = ixys[..., 1] * Wf + ixys[..., 0]
        fltr_full.append(fltr_valid)
        ids_ancr.append(offset_layer + ids_ancr_ly)
        offset_layer = offset_layer + Wf * Hf

    fltr_full = np.concatenate(fltr_full, axis=1)
    ids_ancr = np.concatenate(ids_ancr, axis=1)

    ids_lb, ids_mtch = np.nonzero(fltr_full)
    ids_ancr = ids_ancr[ids_lb, ids_mtch]
    return ids_lb, ids_ancr, ids_mtch


class CategoryWeightAdapter():
    def __init__(self, adpat_power=-0.5):
        self._weight_cls = None
        self._adpat_power = adpat_power

    def fit_weight_cls(self, objnums: np.ndarray):
        objnums = objnums / np.sum(objnums)
        weight = np.where(objnums > 0, np.power(objnums, self._adpat_power), 0)
        weight = weight / np.sum(weight * objnums)
        self._weight_cls = arrsN2arrsT(weight)

    # def get_weight_cls(self, chots_pd: torch.Tensor, expand: bool = True, pos_only: bool = True):
    #     if self._weight_cls is None:
    #         return None
    #     weight_cls = self._weight_cls.to(chots_pd.device)
    #     if expand:
    #         weight_cls = weight_cls.expand_as(chots_pd)
    #         if pos_only:
    #             weight_cls = torch.where(chots_pd > 0, weight_cls, torch.as_tensor(1.0).to(chots_pd.device))
    #     return weight_cls

    def get_weight_cls(self, chots_pd: torch.Tensor):
        if self._weight_cls is None:
            return None
        weight_cls = self._weight_cls.to(chots_pd.device)
        weight_cls = torch.sum(weight_cls * chots_pd, dim=-1, keepdim=True)
        weight_cls = torch.where(weight_cls > 0, weight_cls, torch.as_tensor(1.0).to(chots_pd.device))
        return weight_cls

# </editor-fold>
