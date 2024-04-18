from torch.cuda import amp

from tools.interface import *
from utils import *


# <editor-fold desc='优化器构建'>
class OptimizerBuilder(DCTExtractable):

    @abstractmethod
    def build_optimizer(self, name, module):
        pass

    @property
    def name(self):
        return self.__class__.__name__


class GroupSGDBuilder(OptimizerBuilder, InitDCTExtractable):
    def __init__(self, lr=0.001, momentum=0.937, dampening=0, weight_decay=5e-4, ):
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.lr = lr

    def build_optimizer(self, name, module):
        group_bias, group_weight, group_weight_ndcy = [], [], []
        for name, para in module.named_parameters():
            # print(name)
            if 'bias' in name:  # bias (no decay)
                group_bias.append(para)
            elif 'weight' in name and 'bn' in name:  # weight (no decay)
                group_weight_ndcy.append(para)
            else:
                group_weight.append(para)

        optimizer = torch.optim.SGD(group_bias, lr=self.lr, momentum=self.momentum, nesterov=True)
        optimizer.add_param_group({'params': group_weight, 'weight_decay': self.weight_decay})
        optimizer.add_param_group({'params': group_weight_ndcy, 'weight_decay': 0.0})

        return optimizer


class SGDBuilder(OptimizerBuilder, InitDCTExtractable):

    def __init__(self, lr=0.001, momentum=0.8, dampening=0, weight_decay=1e-5, nesterov=True):
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.lr = lr
        self.nesterov = nesterov

    def build_optimizer(self, name, module):
        return torch.optim.SGD(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            momentum=self.momentum, weight_decay=self.weight_decay, lr=self.lr, dampening=self.dampening,
            nesterov=self.nesterov)


class RMSpropBuilder(OptimizerBuilder, InitDCTExtractable):

    def __init__(self, lr=0.01, alpha=0.99, weight_decay=0, momentum=0, ):
        self.momentum = momentum
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.lr = lr

    def build_optimizer(self, name, module):
        return torch.optim.RMSprop(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            momentum=self.momentum, weight_decay=self.weight_decay, lr=self.lr, alpha=self.alpha)


class AdamBuilder(OptimizerBuilder, InitDCTExtractable):

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, ):
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr = lr

    def build_optimizer(self, name, module):
        return torch.optim.Adam(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            eps=self.eps, weight_decay=self.weight_decay, lr=self.lr, betas=self.betas)


# </editor-fold>

# <editor-fold desc='训练原型'>


class Trainer(LossCollector, IterBasedTemplate):
    OPT_BUILDER = SGDBuilder()
    SCOPE_TRAIN = 'train'
    SCOPE_PROCESS = 'process'
    SCOPE_AUGMENTATION = 'augmentation'
    SCOPE_ACTOR = 'actor'

    LEARNNING_RATE = 'learning_rate'
    ACCU_STEP = 'accu_step'
    IMAGE_SIZE = 'img_size'
    ETA = 'eta'
    GRAD_NORM = 'grad_norm'
    LOSS_GAIN = 'loss_gain'
    ENABLE_AMP = 'enable_amp'

    TOTAL_EPOCH = IterBasedActorContainer.TOTAL_EPOCH
    TOTAL_ITER = IterBasedActorContainer.TOTAL_ITER
    IND_EPOCH = IterBasedActorContainer.IND_EPOCH
    IND_ITER = IterBasedActorContainer.IND_ITER
    IND_ITER_INEP = IterBasedActorContainer.IND_ITER_INEP

    def __init__(self, loader, opt_builder=OPT_BUILDER, total_epoch=None, total_iter=None, accu_step=1,
                 collector=None, main_proc=True, grad_norm=None, enable_amp=False, loss_gain=1.0):
        IterBasedTemplate.__init__(
            self, loader=loader, device=None, processor=None,
            total_epoch=total_epoch, total_iter=total_iter, collector=collector, main_proc=main_proc)
        LossCollector.__init__(self)

        self.model = None
        self.kwargs_train = {}
        self.optimizers = {}
        self.img_size = loader.img_size if hasattr(loader, 'img_size') else None
        self.accu_step = accu_step

        self.grad_norm = grad_norm
        self.scaler = amp.GradScaler(enabled=enable_amp)
        self.loss_gain = loss_gain
        self.opt_builder = opt_builder

    # <editor-fold desc='optimizer处理'>

    def optimizer_step(self, name=None, with_accu=True):
        if with_accu and self.ind_iter % self.accu_step != 0:
            return self
        for name_opt, optimizer in self.optimizers.items():
            if name is not None and not name == name_opt:
                continue
            self.scaler.unscale_(optimizer)
            if self.grad_norm is not None and self.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.scaler.step(optimizer)  # optimizer.step
            self.scaler.update()
        return self

    def optimizer_zero_grad(self, name=None, with_accu=True):
        if with_accu and self.ind_iter % self.accu_step != 0:
            return self
        for name_opt, optimizer in self.optimizers.items():
            if name is not None and not name == name_opt:
                continue
            optimizer.zero_grad()
        return self

    def loss_backward(self, loss, name='Loss'):
        loss, names, losses = LossCollector.process_loss(
            loss, loss_gain=1.0 if self.loss_gain is None else self.loss_gain)
        self.update_loss(name, loss)
        self.update_losses(names, losses)
        self.scaler.scale(loss).backward()
        return self

    def optimizer_attr_set(self, value, name=None, group_index=None, attr='lr'):
        for name_opt, optimizer in self.optimizers.items():
            if name is not None and not name == name_opt:
                continue
            for k, param_group in enumerate(optimizer.param_groups):
                if group_index is not None and not group_index == k:
                    continue
                param_group[attr] = value
        return self

    def optimizer_lr_set(self, learning_rate, name=None, group_index=None):
        self.kwargs_train[Trainer.LEARNNING_RATE] = learning_rate
        return self.optimizer_attr_set(learning_rate, name, group_index)


    # </editor-fold>

    # <editor-fold desc='参数管理'>
    @property
    def img_size(self):
        return self.kwargs_train.get(Trainer.IMAGE_SIZE, None)

    @img_size.setter
    def img_size(self, img_size):
        if self.model is not None:
            self.model.img_size = img_size
        self.loader.img_size = img_size
        self.kwargs_train[Trainer.IMAGE_SIZE] = img_size

    @property
    def accu_step(self):
        return self.kwargs_train.get(Trainer.ACCU_STEP, 1)

    @accu_step.setter
    def accu_step(self, accu_step):
        self.kwargs_train[Trainer.ACCU_STEP] = accu_step

    @property
    def grad_norm(self):
        return self.kwargs_train.get(Trainer.GRAD_NORM, 1)

    @grad_norm.setter
    def grad_norm(self, grad_norm):
        self.kwargs_train[Trainer.GRAD_NORM] = grad_norm

    @property
    def loss_gain(self):
        return self.kwargs_train.get(Trainer.LOSS_GAIN, 1.0)

    @loss_gain.setter
    def loss_gain(self, loss_gain):
        self.kwargs_train[Trainer.LOSS_GAIN] = loss_gain

    @property
    def learning_rate(self):
        return self.kwargs_train.get(Trainer.LEARNNING_RATE, 0)

    @property
    def learning_rates(self):
        lr_dct = {}
        for name_opt, optimizer in self.optimizers.items():
            lr_dct[name_opt] = [pg['lr'] for pg in optimizer.param_groups]
        return lr_dct

    def collect_kwargs(self, names):
        kwargs_all = {}
        kwargs_all.update(self.kwargs_train)
        kwargs_all.update(self.kwargs_proc)
        vals = [kwargs_all.get(name, None) for name in names]
        return vals

    # </editor-fold>

    # <editor-fold desc='权重保存管理'>
    def save(self, save_pth):
        self.save_model(save_pth)
        self.save_optimizer(save_pth)
        # self.save_json(save_pth)
        for actor in self.actors_dct[SaveActor]:
            actor.act_save(self, save_pth)
        return self

    def load(self, save_pth):
        self.load_model(save_pth)
        self.load_optimizer(save_pth)
        # self.load_json(save_pth)
        for actor in self.actors_dct[LoadActor]:
            actor.act_load(self, save_pth)
        return self

    def save_model(self, save_pth):
        model_pth = os.path.abspath(ensure_extend(save_pth, 'pth'))
        self.broadcast('Save model to ' + model_pth)
        self.model.save(model_pth)
        return self

    def load_model(self, save_pth):
        model_pth = os.path.abspath(ensure_extend(save_pth, 'pth'))
        self.broadcast('Load model from ' + model_pth)
        self.model.load(model_pth)
        return self

    def save_optimizer(self, save_pth):
        opt_pth = os.path.abspath(ensure_extend(save_pth, 'opt'))
        opt_dct = {}
        for name, optimizer in self.optimizers.items():
            opt_dct[name] = optimizer.state_dict()
        self.broadcast('Save optimizer to ' + opt_pth)
        torch.save(opt_dct, opt_pth)
        return self

    def load_optimizer(self, save_pth):
        opt_pth = os.path.abspath(ensure_extend(save_pth, 'opt'))
        self.broadcast('Load optimizer from ' + opt_pth)
        opt_dct = torch.load(opt_pth)
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(opt_dct[name])
        return self

    def extract_dct(self):
        kwargs_actor = {}
        for actor in self.actors:
            kwargs_actor[actor.name] = actor.extract_dct()
        aug_seq = self.loader.aug_seq
        dct = {
            Trainer.SCOPE_TRAIN: self.kwargs_train,
            Trainer.SCOPE_PROCESS: self.kwargs_proc,
            Trainer.SCOPE_ACTOR: kwargs_actor,
            Trainer.SCOPE_AUGMENTATION: aug_seq.extract_dct() if aug_seq is not None else None,
        }
        return dct

    def refrom_dct(self, dct):
        self.kwargs_train.update(dct)
        for actor in self.actors:
            actor.refrom_dct(dct[actor.name])
        return self

    # </editor-fold>

    # <editor-fold desc='训练执行'>

    def act_init(self, model, *args, **kwargs):
        self.model = model
        self.device = model.device
        self.running = True
        for name, pkd_module in model.pkd_modules.items():
            self.optimizers[name] = self.opt_builder.build_optimizer(name, pkd_module)
        self.processor = model.labels2tars
        # self.add_actor(model)
        self.model.act_init_train(self)
        return None

    def act_return(self):
        return None

    def act_ending(self):
        return None

    def act_iter(self):
        imgs, targets = self.batch_data
        self.model.act_iter_train(self, imgs, targets)
    # </editor-fold>


# </editor-fold>


# <editor-fold desc='显示管理'>


class TrainerBroadcaster(IntervalTrigger, Broadcaster, BeforeEpochActor, AfterIterActor, InitDCTExtractable):

    def __init__(self, step=50, offset=0, first=False, last=False):
        super(TrainerBroadcaster, self).__init__(step=step, offset=offset, first=first, last=last)

    def get_loss_msg(self, trainer, **kwargs):
        loss_dct = trainer.loss_dct
        return ''.join([name + ' %-5.4f ' % val for name, val in loss_dct.items()])

    def get_time_msg(self, trainer, **kwargs):
        return ''.join([name + ' %-5.4f ' % val for name, val in trainer.periods.items()])

    def get_eta_msg(self, trainer, **kwargs):
        sec = trainer.eta
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)

    def act_after_iter(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_iter_inep, total=len(trainer)):
            msg = 'Iter %06d ' % (trainer.ind_iter + 1) + '[ %04d ]' % (trainer.ind_iter_inep + 1) + ' | '
            msg += 'Lr %-7.6f ' % trainer.learning_rate + '| '
            msg += self.get_loss_msg(trainer, **kwargs) + '| '
            msg += self.get_time_msg(trainer, **kwargs) + '| '
            trainer.broadcast(msg)

    def act_before_epoch(self, trainer, **kwargs):
        msg = '< Train > Epoch %d' % (trainer.ind_epoch + 1) + '  Data %d' % trainer.num_data + \
              '  Batch %d' % trainer.num_batch + '  BatchSize %d' % trainer.batch_size + '[x%d]' % trainer.accu_step + \
              '  ImgSize ' + str(trainer.img_size) + '  ETA ' + self.get_eta_msg(trainer, **kwargs)
        trainer.broadcast(msg)


class PrintBasedTrainerBroadcaster(PrintBasedBroadcaster, TrainerBroadcaster):
    def __init__(self, step=50, offset=0, first=True, last=True, formatter=FORMATTER.BROADCAST):
        super(PrintBasedTrainerBroadcaster, self).__init__(formatter=formatter)
        super(PrintBasedBroadcaster, self).__init__(step=step, offset=offset, first=first, last=last, )


class LogBasedTrainerBroadcaster(LogBasedBroadcaster, TrainerBroadcaster):
    def __init__(self, log_pth, step=50, offset=0, first=True, last=True, formatter=FORMATTER.BROADCAST,
                 with_print=True, log_name='trainer'):
        super(LogBasedTrainerBroadcaster, self).__init__(
            log_pth, with_print=with_print, log_name=log_name, formatter=formatter)
        super(LogBasedBroadcaster, self).__init__(step=step, offset=offset, first=first, last=last)


# </editor-fold>

# <editor-fold desc='保存管理'>
class Saver(DCTExtractable):
    @property
    def name(self):
        return self.__class__.__name__


class EpochBasedSaver(IntervalTrigger, Saver, AfterEpochActor, InitDCTExtractable):

    def __init__(self, save_pth, step, offset=0, first=False, last=False, cover=True):
        super(EpochBasedSaver, self).__init__(step=step, offset=offset, first=first, last=last)
        self.save_pth = save_pth
        self.cover = cover

    def act_after_epoch(self, trainer, **kwargs):
        if not self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            return None
        if not self.cover:
            save_pth = os.path.splitext(self.save_pth)[0] + '_' + str(trainer.ind_epoch + 1)
        else:
            save_pth = self.save_pth
        trainer.save(save_pth)


class IterBasedSaver(IntervalTrigger, Saver, AfterIterActor, InitDCTExtractable):

    def __init__(self, save_pth, step, offset=0, first=False, last=False, cover=True):
        super(IterBasedSaver, self).__init__(step=step, offset=offset, first=first, last=last)
        self.save_pth = save_pth
        self.cover = cover

    def act_after_iter(self, trainer, **kwargs):
        if not self.trigger(ind=trainer.ind_iter, total=trainer.total_iter):
            return None
        if not self.cover:
            save_pth = os.path.splitext(self.save_pth)[0] + '_' + str(trainer.ind_iter + 1)
        else:
            save_pth = self.save_pth
        trainer.save(save_pth)


# </editor-fold>

# <editor-fold desc='运行数据记录'>
class IterBasedRecorder(Recorder, AfterIterActor, SaveActor, LoadActor, InitDCTExtractable):
    def act_save(self, trainer, save_pth, appendix='', **kwargs):
        xlsx_pth = os.path.abspath(ensure_extend(save_pth, 'xlsx'))
        trainer.broadcast('Save record to ' + xlsx_pth)
        self.record.to_excel(xlsx_pth, index=False)

    def act_load(self, trainer, save_pth, appendix='', **kwargs):
        xlsx_pth = os.path.abspath(ensure_extend(save_pth, 'xlsx'))
        trainer.broadcast('Load record from ' + xlsx_pth)
        self.record = pd.read_excel(xlsx_pth)

    COMMON_QUERYS = (
        Trainer.IND_ITER,
        Trainer.IND_EPOCH,
        Trainer.IND_ITER_INEP,
        Trainer.LEARNNING_RATE
    )

    def __init__(self, com_query=COMMON_QUERYS):
        self.record = pd.DataFrame()
        self.com_query = com_query

    def get_loss_cols(self, trainer, **kwargs):
        loss_dct = trainer.loss_dct
        names, vals = zip(*loss_dct.items())
        return names, vals

    def get_time_cols(self, trainer, **kwargs):
        periods = trainer.periods
        return list(periods.keys()), list(periods.values())

    def get_com_cols(self, trainer, **kwargs):
        if self.com_query is None:
            return [], []
        vals = trainer.collect_kwargs(self.com_query)
        return self.com_query, vals

    def act_after_iter(self, trainer, **kwargs):
        names_loss, vals_loss = self.get_loss_cols(trainer, **kwargs)
        names_time, vals_time = self.get_time_cols(trainer, **kwargs)
        names_com, vals_com = self.get_com_cols(trainer, **kwargs)
        names = list(names_com) + list(names_time) + list(names_loss)
        vals = list(vals_com) + list(vals_time) + list(vals_loss)
        if len(names) > 0:
            row = pd.DataFrame([dict(zip(names, vals))], columns=names)
            self.record = pd.concat([self.record, row])

# </editor-fold>


# if __name__ == '__main__':

#     loader=
