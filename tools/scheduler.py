from tools.interface import *


# <editor-fold desc='矢量曲线'>

class ScalableFunc(DCTExtractable):
    def __init__(self, num_biter=10, scale=1):
        self.num_biter = num_biter
        self.scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @property
    def num_biter(self):
        return self._num_biter

    @num_biter.setter
    def num_biter(self, num_biter):
        self._num_biter = num_biter

    @property
    def num_siter(self):
        return self.num_biter * self.scale

    @abstractmethod
    def __getitem__(self, ind_siter):
        pass


class ScalableSize(ScalableFunc):
    @staticmethod
    def convert(scsize):
        if isinstance(scsize, ScalableSize):
            return scsize
        elif isinstance(scsize, tuple) or isinstance(scsize, list):
            return ConstSize(size=scsize, num_biter=1)
        else:
            raise Exception('fmt err' + scsize.__class__.__name__)


class ConstSize(ScalableSize, InitDCTExtractable):
    def __init__(self, size, num_biter=10, scale=1):
        ScalableSize.__init__(self, num_biter=num_biter, scale=scale)
        self.size = size

    def __getitem__(self, ind_siter):
        return self.size


class RandSize(ScalableSize, InitDCTExtractable):

    def __init__(self, min_size, max_size, devisor=32, keep_ratio=True, num_biter_keep=1, max_first=True, max_last=True,
                 num_biter=10, scale=1):
        super().__init__(num_biter=num_biter, scale=scale)
        self.min_size = min_size
        self.max_size = max_size
        self.devisor = devisor
        self.keep_ratio = keep_ratio
        self.num_biter_keep = num_biter_keep
        self.max_first = max_first
        self.max_last = max_last

        self.max_w, self.max_h = int(math.floor(max_size[0] / devisor)), int(math.floor(max_size[1] / devisor))
        self.min_w, self.min_h = int(math.ceil(min_size[0] / devisor)), int(math.ceil(min_size[1] / devisor))
        self.last_size = self._rand_size()
        self.kpd = 0

    @property
    def num_siter_keep(self):
        return self.num_biter_keep * self.scale

    def _rand_size(self):
        w = random.randint(self.min_w, self.max_w)
        if self.keep_ratio:
            h = int(1.0 * (w - self.min_w) / (self.max_w - self.min_w) * (self.max_h - self.min_h) + self.min_h)
        else:
            h = random.randint(self.min_h, self.max_h)
        return (w * self.devisor, h * self.devisor)

    def __getitem__(self, ind_siter):
        if (self.max_first and ind_siter <= 0) \
                or (self.max_last and ind_siter >= self.num_siter - self.num_siter_keep):
            size = self.max_size
        elif self.kpd < self.num_siter_keep:
            size = self.last_size
        else:
            size = self._rand_size()
            self.kpd = 0
        self.kpd = self.kpd + 1
        self.last_size = size
        return size


class ScalableCurve(ScalableFunc):
    @property
    def vals(self):
        vals = []
        for i in range(self.num_siter):
            vals.append(self.__getitem__(i))
        return vals

    def __imul__(self, other):
        return self

    @staticmethod
    def convert(curve):
        if isinstance(curve, ScalableCurve):
            return curve
        elif isinstance(curve, float):
            return ConstCurve(val=curve, num_biter=1, scale=1)
        else:
            raise Exception('fmt err' + curve.__class__.__name__)


class ComposedCurve(ScalableCurve):
    def extract_dct(self):
        dct = {}
        for i, curve in enumerate(self.curves):
            dct[i] = curve.extract_dct()
        return dct

    def refrom_dct(self, dct):
        for i, curve in enumerate(self.curves):
            curve.reform_dct(dct[i])
        return dct

    @property
    def num_biter(self):
        return sum([curve.num_biter for curve in self.curves])

    @num_biter.setter
    def num_biter(self, num_biter):
        pass

    @property
    def scale(self):
        return self.num_siter / max(self.num_biter, 1)

    @scale.setter
    def scale(self, scale):
        num_siter = 0
        milestones = []
        for curve in self.curves:
            curve.scale = scale
            milestones.append(num_siter)
            num_siter += curve.num_siter
        self.milestones = milestones

    @property
    def num_siter(self):
        return sum([curve.num_siter for curve in self.curves])

    def __init__(self, *curves, scale=1):
        self.curves = curves
        ScalableCurve.__init__(self, num_biter=0, scale=scale)

    def __imul__(self, other):
        for curve in self.curves:
            curve.__imul__(other)
        return self

    def __getitem__(self, ind_siter):
        for i in range(len(self.milestones) - 1, -1, -1):
            if ind_siter >= self.milestones[i]:
                return self.curves[i].__getitem__(ind_siter - self.milestones[i])
        raise Exception('milestones err')


class MultiStepCurve(ScalableCurve, InitDCTExtractable):

    def __init__(self, val_init=0.1, bmilestones=(0, 1), gamma=0.1, num_biter=10, scale=1):
        if isinstance(bmilestones, int):
            bmilestones = [bmilestones]
        self.bmilestones = list(bmilestones)
        self.gamma = gamma
        self.val_init = val_init
        ScalableCurve.__init__(self, num_biter=num_biter, scale=scale)

    def __imul__(self, other):
        self.val_init *= other
        return self

    def __getitem__(self, ind_siter):
        lr = self.val_init
        for milestone in self.smilestones:
            lr = lr * self.gamma if ind_siter >= milestone else lr
        return lr

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale
        smilestones = []
        for i in range(len(self.bmilestones)):
            smilestones.append(self.bmilestones[i] * scale)
        self.smilestones = smilestones


class PowerCurve(ScalableCurve, InitDCTExtractable):
    def __init__(self, val_init=0.1, val_end=1e-8, num_biter=10, scale=1, pow=2):
        super(PowerCurve, self).__init__(num_biter=num_biter, scale=scale)
        self.val_init = val_init
        self.val_end = val_end
        self.pow = pow

    def __getitem__(self, ind_siter):
        alpha = (ind_siter / self.num_siter) ** self.pow
        val = (1 - alpha) * self.val_init + alpha * self.val_end
        return val

    def __imul__(self, other):
        self.val_init *= other
        self.val_end *= other
        return self


class ExponentialCurve(ScalableCurve, InitDCTExtractable):
    def __init__(self, val_init=0.1, val_end=1e-8, num_biter=10, scale=1):
        super(ExponentialCurve, self).__init__(num_biter=num_biter, scale=scale)
        self.val_init_log = math.log(val_init)
        self.val_end_log = math.log(val_end)

    def __getitem__(self, ind_siter):
        alpha = ind_siter / self.num_siter
        val_log = (1 - alpha) * self.val_init_log + alpha * self.val_end_log
        return math.exp(val_log)

    def __imul__(self, other):
        self.val_init_log += math.log(other)
        self.val_end_log += math.log(other)
        return self


class CosCurve(ScalableCurve, InitDCTExtractable):

    def __init__(self, val_init=0.1, val_end=1e-8, num_biter=10, scale=1):
        super(CosCurve, self).__init__(num_biter=num_biter, scale=scale)
        self.val_init = val_init
        self.val_end = val_end

    def __getitem__(self, ind_siter):
        alpha = ind_siter / self.num_siter
        lr = self.val_end + (self.val_init - self.val_end) * 0.5 * (1.0 + math.cos(math.pi * alpha))
        return lr

    def __imul__(self, other):
        self.val_init *= other
        self.val_end *= other
        return self


class ConstCurve(ScalableCurve, InitDCTExtractable):
    def __init__(self, val=0.1, num_biter=10, scale=1):
        super(ConstCurve, self).__init__(num_biter=num_biter, scale=scale)
        self.val = val

    def __getitem__(self, ind_siter):
        return self.val

    def __imul__(self, other):
        self.val *= other
        return self


# </editor-fold>

# <editor-fold desc='学习率'>
class CurveBasedLRScheduler(LRScheduler, InitDCTExtractable):

    def __init__(self, curve: ScalableCurve, name=None, group_index=None):
        self.curve = curve
        self.name = name
        self.group_index = group_index


class EpochBasedLRScheduler(CurveBasedLRScheduler, BeforeEpochActor):

    def act_add(self, trainer, **kwargs):
        self.curve.scale = 1
        trainer.total_epoch = self.curve.num_biter

    def act_before_epoch(self, trainer, **kwargs):
        learning_rate = self.curve[trainer.ind_epoch]
        trainer.optimizer_lr_set(learning_rate, name=self.name, group_index=self.group_index)

    @staticmethod
    def Const(lr=0.1, num_epoch=10,name=None, group_index=None):
        return EpochBasedLRScheduler(ConstCurve(val=lr, num_biter=num_epoch),
                                     name=name, group_index=group_index)

    @staticmethod
    def Cos(lr_init=0.1, lr_end=1e-8, num_epoch=10,name=None, group_index=None):
        return EpochBasedLRScheduler(CosCurve(val_init=lr_init, val_end=lr_end, num_biter=num_epoch),
                                     name=name, group_index=group_index)

    @staticmethod
    def WarmCos(lr_init=0.1, lr_end=1e-8, num_epoch=10, num_warm=1,name=None, group_index=None):
        curve = ComposedCurve(
            PowerCurve(val_init=0, val_end=lr_init, num_biter=num_warm),
            CosCurve(val_init=lr_init, val_end=lr_end, num_biter=max(0, num_epoch - num_warm))
        )
        return EpochBasedLRScheduler(curve,name=name, group_index=group_index)

    @staticmethod
    def MultiStep(lr_init=0.1, milestones=(0, 1), gamma=0.1, num_epoch=10, name=None, group_index=None):
        return EpochBasedLRScheduler(MultiStepCurve(
            val_init=lr_init, bmilestones=milestones, gamma=gamma, num_biter=num_epoch), name, group_index)


class EpochBasedConsecutiveLRScheduler(CurveBasedLRScheduler, BeforeIterActor):

    def act_add(self, trainer, **kwargs):
        self.curve.scale = len(trainer.loader)
        trainer.total_epoch = self.curve.num_biter

    def act_before_iter(self, trainer, **kwargs):
        learning_rate = self.curve[trainer.ind_iter]
        trainer.optimizer_lr_set(learning_rate, name=self.name, group_index=self.group_index)

    @staticmethod
    def Const(lr=0.1, num_epoch=10, name=None, group_index=None):
        return EpochBasedLRScheduler(ConstCurve(val=lr, num_biter=num_epoch), name, group_index)

    @staticmethod
    def Cos(lr_init=0.1, lr_end=1e-8, num_epoch=10, name=None, group_index=None):
        return EpochBasedLRScheduler(CosCurve(val_init=lr_init, val_end=lr_end, num_biter=num_epoch), name, group_index)

    @staticmethod
    def WarmCos(lr_init=0.1, lr_end=1e-8, num_epoch=10, num_warm=1, name=None, group_index=None):
        curve = ComposedCurve(
            PowerCurve(val_init=0, val_end=lr_init, num_biter=num_warm),
            CosCurve(val_init=lr_init, val_end=lr_end, num_biter=max(0, num_epoch - num_warm))
        )
        return EpochBasedConsecutiveLRScheduler(curve, name, group_index)

    @staticmethod
    def MultiStep(lr_init=0.1, milestones=(0, 1), gamma=0.1, num_epoch=10, name=None, group_index=None):
        return EpochBasedLRScheduler(MultiStepCurve(
            val_init=lr_init, bmilestones=milestones, gamma=gamma, num_biter=num_epoch), name, group_index)


class IterBasedLRScheduler(CurveBasedLRScheduler, BeforeIterActor):

    def act_add(self, trainer, **kwargs):
        self.curve.scale = 1
        trainer.total_iter = self.curve.num_biter

    def act_before_iter(self, trainer, **kwargs):
        learning_rate = self.curve[trainer.ind_iter]
        trainer.optimizer_lr_set(learning_rate, name=self.name, group_index=self.group_index)

    @staticmethod
    def Const(lr=0.1, num_iter=10, name=None, group_index=None):
        return IterBasedLRScheduler(ConstCurve(val=lr, num_biter=num_iter), name, group_index)

    @staticmethod
    def Cos(lr_init=0.1, lr_end=1e-8, num_iter=10, name=None, group_index=None):
        return IterBasedLRScheduler(CosCurve(val_init=lr_init, val_end=lr_end, num_biter=num_iter), name, group_index)

    @staticmethod
    def MultiStep(lr_init=0.1, milestones=(0, 1), gamma=0.1, num_iter=10, name=None, group_index=None):
        return IterBasedLRScheduler(MultiStepCurve(
            val_init=lr_init, bmilestones=milestones, gamma=gamma, num_biter=num_iter), name, group_index)


class FuncBasedMScheduler(IMScheduler, InitDCTExtractable):

    def __init__(self, scsize: ScalableSize):
        self.scsize = scsize


class EpochBasedIMScheduler(FuncBasedMScheduler, BeforeEpochActor):

    def act_add(self, trainer, **kwargs):
        self.scsize.scale = 1
        trainer.total_epoch = self.scsize.num_biter

    def act_before_epoch(self, trainer, **kwargs):
        trainer.img_size = self.scsize[trainer.ind_epoch]

    @staticmethod
    def Const(img_size=(32, 32), num_epoch=10):
        return EpochBasedIMScheduler(ConstSize(size=img_size, num_biter=num_epoch, scale=1))

    @staticmethod
    def Rand(min_size, max_size, devisor=32, keep_ratio=True, num_keep=1, max_first=True, max_last=True,
             num_epoch=10):
        return EpochBasedIMScheduler(RandSize(
            min_size=min_size, max_size=max_size, devisor=devisor, keep_ratio=keep_ratio,
            num_biter_keep=num_keep, max_first=max_first, max_last=max_last,
            num_biter=num_epoch, scale=1))


class IterBasedIMScheduler(FuncBasedMScheduler, BeforeIterActor):

    def act_add(self, trainer, **kwargs):
        self.scsize.scale = 1
        trainer.total_iter = self.scsize.num_biter

    def act_before_iter(self, trainer, **kwargs):
        trainer.img_size = self.scsize[trainer.ind_iter]

    @staticmethod
    def Const(img_size=(32, 32), num_iter=10):
        return IterBasedIMScheduler(ConstSize(size=img_size, num_biter=num_iter, scale=1))

    @staticmethod
    def Rand(min_size, max_size, devisor=32, keep_ratio=True, num_keep=1, max_first=True, max_last=True,
             num_iter=10):
        return IterBasedIMScheduler(RandSize(
            min_size=min_size, max_size=max_size, devisor=devisor, keep_ratio=keep_ratio,
            num_biter_keep=num_keep, max_first=max_first, max_last=max_last,
            num_biter=num_iter, scale=1))


# </editor-fold>


# if __name__ == '__main__':
#     from visual import *
#
#     curve = ScalableCurve.WARM_COS(val_init=0.0025, warm_epoch=1, num_biter=300)
#     # curve = ScalableCurve.WARM_STEP(val_init=0.1, warm_epoch=50, milestones=(20, 30), gamma=0.1, num_biter=100)
#     curve.scale = 100
#     plt.plot(curve.lr_list)

if __name__ == '__main__':
    curve = ConstCurve(val=0.1, num_biter=10, scale=1)
    dct = curve.extract_dct()

    a = eval('ConstCurve')(**dct)
    # curve.load_json('./test')
