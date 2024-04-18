from models.gan.pennet import ResUNetInPMain, PatchDiscriminator, GatedGenerator
from models.modules import *
from models.template import *


def masked_l1_loss(feat_pd, feat_tg, mask, wfg=100.0, wbg=10.0):
    weight = torch.where(mask > 0, torch.as_tensor(wfg).to(feat_pd.device),
                         torch.as_tensor(wbg).to(feat_pd.device))
    l1_loss = torch.mean(torch.abs(feat_pd - feat_tg) * weight)
    return l1_loss


def mutilev_l1_loss(feats_pd, feat_tg, mask, wfg=100.0, wbg=10.0):
    l1_loss = 0.0
    for feat_pd in feats_pd:
        feat_tg_i = F.interpolate(feat_tg, size=(feat_pd.size(2), feat_pd.size(3)))
        mask_i = F.interpolate(mask, size=(feat_pd.size(2), feat_pd.size(3)))
        l1_loss += masked_l1_loss(feat_pd, feat_tg_i, mask_i, wfg=wfg, wbg=wbg)
    return l1_loss


class Filler(GANTrainable):

    def __init__(self, generator, discriminator, device=None, pack=PACK.AUTO, num_cls=20):
        super(Filler, self).__init__(generator=generator, discriminator=discriminator, device=device, pack=pack)
        self._num_cls = num_cls

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self.generator.img_size

    def labels2tars(self, labels, expand_ratio=2.0, **kwargs):
        masks_tg = [np.zeros(shape=(0, 1, self.img_size[1], self.img_size[0]))]
        for label in labels:
            mask_i = np.zeros(shape=(self.img_size[1], self.img_size[0]))
            cap_ms = label.filt(lambda item: item.category.cindN == 4)
            cap_rs = label.filt(lambda item: item.category.cindN == 2)

            for item in cap_ms:
                xlyl = XLYLBorder.convert(item.border).xlylN
                xlyl = xlylN_expand(xlyl, ratio=expand_ratio)
                pow = xlylN2maskN_dpow(xlyl, size=self.img_size, normlize=True)
                mask_i = mask_i + pow

            if len(cap_ms) == 0 and len(cap_rs) > 0:
                item = cap_rs[np.random.choice(len(cap_rs))]
                xlyl = XLYLBorder.convert(item.border).xlylN
                xlyl = xlylN_expand(xlyl, ratio=expand_ratio)
                xyxy = xlylN2xyxyN(xlyl).astype(np.int32)
                xyxy_patch = xyxyN_samp_size(xyxyN=np.array([0, 0, self.img_size[0], self.img_size[1]]),
                                             patch_size=xyxy[2:4] - xyxy[:2])
                xlyl = xlyl + xyxy_patch[:2] - xyxy[:2]
                pow = xlylN2maskN_dpow(xlyl, size=self.img_size, normlize=True)
                mask_i = mask_i + pow

            masks_tg.append(mask_i[None, None])
        masks_tg = np.concatenate(masks_tg, axis=0)

        # show_arrs(masks_tg[:, 0])
        # plt.pause(1e5)
        return masks_tg

    def imgs_tars2gen_loss(self, imgs, targets, **kwargs):
        self.train()
        masks_tg = arrsN2arrsT(targets, device=self.device)
        masks_shp = (masks_tg > 0).float()
        imgs_shp = imgs * (1 - masks_shp)
        imgs_shp_masks = torch.cat([imgs_shp, masks_shp], dim=1)
        fimgs_shp, fimgss_proj = self.pkd_modules['generator'](imgs_shp_masks)
        fimgs = fimgs_shp * masks_shp + imgs_shp
        fimgs_masks = torch.cat([fimgs, masks_shp], dim=1)
        self.fimgs_masks = fimgs_masks.detach()
        self.imgs_masks = torch.cat([imgs, masks_shp], dim=1)
        preds = self.pkd_modules['discriminator'](fimgs_masks)
        gan_loss = - torch.mean(preds)
        rec_loss0 = masked_l1_loss(fimgs, imgs, masks_tg, wfg=100, wbg=10)

        rec_loss_lv = mutilev_l1_loss(fimgss_proj, imgs, masks_tg, wfg=100, wbg=10)
        return OrderedDict(gan=gan_loss, rec=rec_loss0 + rec_loss_lv)

    def imgs_tars2dis_loss(self, imgs, targets, **kwargs):
        self.train()
        preds_real = self.pkd_modules['discriminator'](self.imgs_masks)
        preds_fake = self.pkd_modules['discriminator'](self.fimgs_masks)
        loss = - torch.mean(preds_real) + torch.mean(preds_fake)
        return loss

    def imgs_labels2fimgs(self, imgs, labels, **kwargs):
        self.eval()
        imgsT, _ = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        masks_tg = self.labels2tars(labels, expand_ratio=2.0, )
        masks_tg = arrsN2arrsT(masks_tg, device=self.device)
        masks_shp = (masks_tg > 0).float()

        # show_arrs(masks_shp[:,0])

        imgs_masks_input = torch.cat([imgsT * (1 - masks_shp), masks_shp], dim=1)
        imgs_rec, _ = self.pkd_modules['generator'](imgs_masks_input)

        show_arrs(imgs_rec.permute(0, 2, 3, 1))
        return imgs_rec * masks_shp + imgsT * (1 - masks_shp)

    # @staticmethod
    # def ResUNetR34(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
    #     backbone = ResUNetMain.R34(act=ACT.RELU, in_channels=in_channels + 1, img_size=img_size, out_channels=3)
    #     return Filler(backbone=backbone, device=device, pack=pack, num_cls=num_cls)

    @staticmethod
    def InPLV5(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        generator = ResUNetInPMain.LV5(act=ACT.LK, in_channels=in_channels + 1, img_size=img_size, out_channels=3)
        discriminator = PatchDiscriminator(channels=64, in_channels=4, out_channels=1, act=ACT.LK, )
        return Filler(generator=generator, discriminator=discriminator, device=device, pack=pack, num_cls=num_cls)

    @staticmethod
    def GatedG(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        generator = GatedGenerator(channels=64, act=ACT.LK, in_channels=in_channels + 1, img_size=img_size,
                                   out_channels=3)
        discriminator = PatchDiscriminator(channels=64, in_channels=4, out_channels=1, act=ACT.LK, )
        return Filler(generator=generator, discriminator=discriminator, device=device, pack=pack, num_cls=num_cls)
