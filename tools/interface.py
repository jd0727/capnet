import logging
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Manager

from typing import Dict

from utils import *


# <editor-fold desc='事件动作'>


class Actor(metaclass=ABCMeta):

    def act_add(self, container, **kwargs):
        pass


class InitialActor(Actor):

    @abstractmethod
    def act_init(self, container, **kwargs):
        pass


class EndingActor(Actor):

    @abstractmethod
    def act_end(self, container, **kwargs):
        pass


class BeforeIterActor(Actor):

    @abstractmethod
    def act_before_iter(self, container, **kwargs):
        pass


class AfterIterActor(Actor):
    @abstractmethod
    def act_after_iter(self, container, **kwargs):
        pass


class BeforeEpochActor(Actor):

    @abstractmethod
    def act_before_epoch(self, container, **kwargs):
        pass


class AfterCycleActor(Actor):
    @abstractmethod
    def act_after_cycle(self, container, **kwargs):
        pass


class BeforeCycleActor(Actor):

    @abstractmethod
    def act_before_cycle(self, container, **kwargs):
        pass


class AfterEpochActor(Actor):
    @abstractmethod
    def act_after_epoch(self, container, **kwargs):
        pass


class SaveActor(Actor):

    @abstractmethod
    def act_save(self, container, save_pth, appendix='', **kwargs):
        pass


class LoadActor(Actor):
    @abstractmethod
    def act_load(self, container, save_pth, appendix='', **kwargs):
        pass


class BroadCastActor(Actor):
    @abstractmethod
    def act_broadcast(self, container, msg, **kwargs):
        pass


# </editor-fold>

# <editor-fold desc='时间管理'>
class TIMENODE:
    BEFORE_TRAIN = 'before_train'
    AFTER_TRAIN = 'after_train'
    BEFORE_PROCESS = 'before_process'
    AFTER_PROCESS = 'after_process'
    BEFORE_EVAL = 'before_eval'
    AFTER_EVAL = 'after_eval'
    BEFORE_CALC = 'before_calc'
    AFTER_CALC = 'after_calc'
    BEFORE_INIT = 'before_init'
    AFTER_INIT = 'after_init'
    BEFORE_CYCLE = 'before_cycle'
    AFTER_CYCLE = 'after_cycle'
    BEFORE_EPOCH = 'before_epoch'
    AFTER_EPOCH = 'after_epoch'
    BEFORE_ITER = 'before_iter'
    AFTER_ITER = 'after_iter'
    BEFORE_INFER = 'before_infer'
    AFTER_INFER = 'after_infer'
    BEFORE_LOAD = 'before_load'
    AFTER_LOAD = 'after_load'
    BEFORE_TARGET = 'before_target'
    AFTER_TARGET = 'after_target'
    BEFORE_CORE = 'before_core'
    AFTER_CORE = 'after_core'
    BEFORE_FORWARD = 'before_foward'
    AFTER_FORWARD = 'after_foward'

    BEFORE_IMG_SAVE = 'before_img_save'
    AFTER_IMG_SAVE = 'after_img_save'

    BEFORE_FORWARD_GEN = 'before_foward_gen'
    AFTER_FORWARD_GEN = 'after_foward_gen'

    BEFORE_FORWARD_DIS = 'before_foward_dis'
    AFTER_FORWARD_DIS = 'after_foward_dis'

    BEFORE_BACKWARD = 'before_backward'
    AFTER_BACKWARD = 'after_backward'

    BEFORE_BACKWARD_GEN = 'before_backward_gen'
    AFTER_BACKWARD_GEN = 'after_backward_gen'

    BEFORE_FORWARD_ENC = 'before_foward_enc'
    AFTER_FORWARD_ENC = 'after_foward_enc'

    BEFORE_FORWARD_DEC = 'before_foward_dec'
    AFTER_FORWARD_DEC = 'after_foward_dec'

    BEFORE_BACKWARD_DIS = 'before_backward_dis'
    AFTER_BACKWARD_DIS = 'after_backward_dis'

    BEFORE_OPTIMIZE = 'before_optimize'
    AFTER_OPTIMIZE = 'after_optimize'

    BEFORE_OPTIMIZE_DIS = 'before_optimize_dis'
    AFTER_OPTIMIZE_DIS = 'after_optimize_dis'

    BEFORE_OPTIMIZE_GEN = 'before_optimize_gen'
    AFTER_OPTIMIZE_GEN = 'after_optimize_gen'


class PERIOD:
    TRAIN = (TIMENODE.BEFORE_TRAIN, TIMENODE.AFTER_TRAIN)
    PROCESS = (TIMENODE.BEFORE_PROCESS, TIMENODE.AFTER_PROCESS)
    EVAL = (TIMENODE.BEFORE_EVAL, TIMENODE.AFTER_EVAL)
    CALC = (TIMENODE.BEFORE_CALC, TIMENODE.AFTER_CALC)
    INIT = (TIMENODE.BEFORE_INIT, TIMENODE.AFTER_INIT)
    CYCLE = (TIMENODE.BEFORE_CYCLE, TIMENODE.AFTER_CYCLE)
    EPOCH = (TIMENODE.BEFORE_EPOCH, TIMENODE.AFTER_EPOCH)
    ITER = (TIMENODE.BEFORE_ITER, TIMENODE.AFTER_ITER)
    INFER = (TIMENODE.BEFORE_INFER, TIMENODE.AFTER_INFER)
    LOAD = (TIMENODE.BEFORE_LOAD, TIMENODE.AFTER_LOAD)
    TARGET = (TIMENODE.BEFORE_TARGET, TIMENODE.AFTER_TARGET)
    CORE = (TIMENODE.BEFORE_CORE, TIMENODE.AFTER_CORE)
    FORWARD = (TIMENODE.BEFORE_FORWARD, TIMENODE.AFTER_FORWARD)
    FORWARD_GEN = (TIMENODE.BEFORE_FORWARD_GEN, TIMENODE.AFTER_FORWARD_GEN)
    FORWARD_ENC = (TIMENODE.BEFORE_FORWARD_ENC, TIMENODE.AFTER_FORWARD_ENC)
    FORWARD_DEC = (TIMENODE.BEFORE_FORWARD_DEC, TIMENODE.AFTER_FORWARD_DEC)
    FORWARD_DIS = (TIMENODE.BEFORE_FORWARD_DIS, TIMENODE.AFTER_FORWARD_DIS)
    BACKWARD = (TIMENODE.BEFORE_BACKWARD, TIMENODE.AFTER_BACKWARD)
    BACKWARD_GEN = (TIMENODE.BEFORE_BACKWARD_GEN, TIMENODE.AFTER_BACKWARD_GEN)
    BACKWARD_DIS = (TIMENODE.BEFORE_BACKWARD_DIS, TIMENODE.AFTER_BACKWARD_DIS)
    OPTIMIZE = (TIMENODE.BEFORE_OPTIMIZE, TIMENODE.AFTER_OPTIMIZE)
    OPTIMIZE_GEN = (TIMENODE.BEFORE_OPTIMIZE_GEN, TIMENODE.AFTER_OPTIMIZE_GEN)
    OPTIMIZE_DIS = (TIMENODE.BEFORE_OPTIMIZE_DIS, TIMENODE.AFTER_OPTIMIZE_DIS)
    IMG_SAVE = (TIMENODE.BEFORE_IMG_SAVE, TIMENODE.AFTER_IMG_SAVE)


class TimeCollector():

    def __init__(self):
        self.time_dct = {}
        self.period_pairs = OrderedDict()

    def update_time(self, *name: str, time_cur: float = None):
        time_cur = time.time() if time_cur is None else time_cur
        for n in name:
            self.time_dct[n] = time_cur
        return self

    def update_period(self, name: str, period_pair: Union[float, tuple]):
        self.period_pairs[name] = period_pair
        return self

    def update_periods(self, period_pairs: Union[Iterable[tuple], Dict[str, tuple]]):
        if isinstance(period_pairs, dict):
            self.period_pairs.update(period_pairs)
        elif isinstance(period_pairs, Iterable):
            for name, period_pair in period_pairs:
                self.period_pairs[name] = period_pair
        else:
            raise Exception('err query ' + period_pairs.__class__.__name__)
        return self

    @property
    def periods(self):
        return self._collect_periods(self.period_pairs)

    def _collect_period(self, period_pair: Union[float, tuple]):
        if isinstance(period_pair, tuple):
            return self.time_dct.get(period_pair[1], 0) - self.time_dct.get(period_pair[0], 0)
        else:
            return period_pair

    def _collect_periods(self, period_pairs: Union[Iterable[tuple], Dict[str, tuple]]):
        if isinstance(period_pairs, dict):
            result = period_pairs.__class__.__new__(period_pairs.__class__)
            for name, period_pair in period_pairs.items():
                result[name] = self._collect_period(period_pair)
            return result
        elif isinstance(period_pairs, Iterable):
            result = []
            for period_pair in period_pairs:
                result.append(self._collect_period(period_pair))
            return result
        else:
            raise Exception('err query ' + period_pairs.__class__.__name__)


# </editor-fold>

# <editor-fold desc='数据预读取'>
class Prefetcher(TimeCollector):

    def __init__(self, loader, device=None, processor=None, cycle=False):
        TimeCollector.__init__(self)
        self.loader = loader
        self.cuda_stream = None
        self.device = device
        self.loader_iter = None
        self.cycle = cycle
        self.processor = processor

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = torch.device(device) if device is not None else DEVICE
        self.cuda_stream = torch.cuda.Stream(self.device) if self.device.index is not None else None

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    @property
    def batch_size(self):
        return self.loader.batch_size

    @property
    def num_batch(self):
        return len(self.loader)

    @property
    def num_data(self):
        return self.loader.num_data

    def set_epoch(self, ind_epoch):
        if isinstance(self.loader.sampler, torch.utils.data.distributed.DistributedSampler):
            self.loader.sampler.set_epoch(ind_epoch)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.thread = threading.Thread(target=self._prefetch_data, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        self.thread.join()
        if self.imgs is None:
            if self.cycle:
                self.loader_iter = iter(self.loader)
                self.thread = threading.Thread(target=self._prefetch_data, daemon=True)
                self.thread.start()
            else:
                self.loader_iter = None
            raise StopIteration
        else:
            imgs, labels = self.imgs, self.labels
            self.thread = threading.Thread(target=self._prefetch_data, daemon=True)
            self.thread.start()
            return imgs, labels

    def _prefetch_data(self):
        try:
            time_before = time.time()
            self.imgs, self.labels = next(self.loader_iter)
            self.update_time(TIMENODE.AFTER_LOAD)
            self.update_time(TIMENODE.BEFORE_LOAD, time_cur=time_before)
            if self.cuda_stream is not None and isinstance(self.imgs, torch.Tensor):
                with torch.cuda.stream(self.cuda_stream):
                    self.imgs = self.imgs.to(device=self.device, non_blocking=True)
            if self.processor is not None:
                time_before = time.time()
                self.labels = self.processor(self.labels)
                self.update_time(TIMENODE.AFTER_TARGET)
                self.update_time(TIMENODE.BEFORE_TARGET, time_cur=time_before)
        except StopIteration:
            self.imgs, self.labels = None, None
        return None


class VirtualLoader(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class SimuLoader(VirtualLoader):
    def __init__(self, size=10, delay_next=0.2, delay_iter=1.0, ptr=0):
        self.size = size
        self.delay_next = delay_next
        self.delay_iter = delay_iter
        self.ptr = ptr

    def __len__(self):
        return self.size

    def __iter__(self):
        print('Build iter')
        time.sleep(self.delay_iter)
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr == self.size:
            raise StopIteration
        else:
            print('Fetching ', self.ptr)
            time.sleep(self.delay_next)
            self.ptr = self.ptr + 1
            imgs = torch.zeros(size=(1,))
            labels = []
            return imgs, labels


class SingleSampleLoader(VirtualLoader):
    def __init__(self, imgs, labels, total_iter=10, ptr=0):
        self.imgs = imgs
        self.labels = labels
        self.total_iter = total_iter
        self.ptr = ptr
        self.batch_size = len(imgs)

    @property
    def num_data(self):
        return self.total_iter * self.batch_size

    def __len__(self):
        return self.total_iter

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.total_iter:
            raise StopIteration
        else:
            self.ptr = self.ptr + 1
            return self.imgs, self.labels


# </editor-fold>

# <editor-fold desc='事件动作'>
class ActorContainer(DCTExtractable):
    SUPPORT_ACTORS = [
        InitialActor,
        EndingActor,
        BeforeIterActor,
        AfterIterActor,
        BeforeEpochActor,
        AfterEpochActor,
        SaveActor,
        LoadActor,
        BroadCastActor,
        BeforeCycleActor,
        AfterCycleActor
    ]

    def __init__(self):
        self.actors_dct = OrderedDict([(at, []) for at in ActorContainer.SUPPORT_ACTORS])

    def add_actor(self, actor):
        for actor_type, actors in self.actors_dct.items():
            if isinstance(actor, actor_type):
                actors.append(actor)
        actor.act_add(self)
        return self

    @property
    def actors(self):
        unquie_actors = set()
        for actor_type, actors in self.actors_dct.items():
            for actor in actors:
                unquie_actors.add(actor)
        return unquie_actors

    def extract_dct(self):
        dct = {}
        for actor in self.actors:
            dct[actor.name] = actor.extract_dct()
        return dct

    def refrom_dct(self, dct):
        for actor in self.actors:
            actor.refrom_dct(dct[actor.name])
        return self


class BasicActorContainer(TimeCollector, ActorContainer):

    def __init__(self, main_proc=True):
        TimeCollector.__init__(self)
        ActorContainer.__init__(self)
        self.main_proc = main_proc

    def broadcast(self, msg, only_main=True, **kwargs):
        if (not only_main) or self.main_proc:
            for actor in self.actors_dct[BroadCastActor]:
                actor.act_broadcast(self, msg, **kwargs)
        return self

    def broadcast_dataframe(self, data, only_main=True, **kwargs):
        msgs = dataframe2strs(data, inter_col='\t', divider=2)
        for msg in msgs:
            self.broadcast(msg, only_main=only_main, **kwargs)
        return None


class IterBasedActorContainer(BasicActorContainer):
    TOTAL_EPOCH = 'total_epoch'
    TOTAL_ITER = 'total_iter'
    IND_EPOCH = 'ind_epoch'
    IND_ITER = 'ind_iter'
    IND_ITER_INEP = 'ind_iter_inep'

    def __init__(self, total_epoch=None, total_iter=None, main_proc=True):
        BasicActorContainer.__init__(self, main_proc=main_proc)
        self.kwargs_proc = {}
        self.ind_epoch = 0
        self.ind_iter = 0
        self.ind_iter_inep = 0
        self.total_epoch = total_epoch
        self.total_iter = total_iter

    @property
    def ind_epoch(self):
        return self.kwargs_proc.get(IterBasedActorContainer.IND_EPOCH, 0)

    @ind_epoch.setter
    def ind_epoch(self, ind_epoch):
        self.kwargs_proc[IterBasedActorContainer.IND_EPOCH] = ind_epoch

    @property
    def ind_iter(self):
        return self.kwargs_proc.get(IterBasedActorContainer.IND_ITER, 0)

    @ind_iter.setter
    def ind_iter(self, ind_iter):
        self.kwargs_proc[IterBasedActorContainer.IND_ITER] = ind_iter

    @property
    def ind_iter_inep(self):
        return self.kwargs_proc.get(IterBasedActorContainer.IND_ITER_INEP, 0)

    @ind_iter_inep.setter
    def ind_iter_inep(self, ind_iter_inep):
        self.kwargs_proc[IterBasedActorContainer.IND_ITER_INEP] = ind_iter_inep

    @property
    def total_epoch(self):
        return self.kwargs_proc.get(IterBasedActorContainer.TOTAL_EPOCH, None)

    @total_epoch.setter
    def total_epoch(self, total_epoch):
        self.kwargs_proc[IterBasedActorContainer.TOTAL_EPOCH] = total_epoch

    @property
    def total_iter(self):
        return self.kwargs_proc.get(IterBasedActorContainer.TOTAL_ITER, None)

    @total_iter.setter
    def total_iter(self, total_iter):
        self.kwargs_proc[IterBasedActorContainer.TOTAL_ITER] = total_iter

    @property
    def eta(self):
        time_start = self.time_dct.get(TIMENODE.BEFORE_CYCLE, 0)
        time_cur = self.time_dct.get(TIMENODE.AFTER_ITER, 0)
        sec_cycled = max(time_cur - time_start, 0)

        total_iter = self.total_iter
        total_epoch = self.total_epoch
        ind_iter = self.ind_iter
        ind_epoch = self.ind_epoch

        scale_epoch = 0 if total_epoch is None or ind_epoch == 0 \
            else 1 / ind_epoch * (total_epoch - ind_epoch)
        scale_iter = 0 if total_iter is None or ind_iter == 0 \
            else 1 / ind_iter * (total_iter - ind_iter)
        sec = sec_cycled * max(scale_epoch, scale_iter)
        return sec


class MultiProcessListCollector():
    def __init__(self, word_size=1):
        manager = Manager()
        self.buffer = manager.Queue(word_size)
        self.finished = manager.Event()
        self.word_size = word_size

    def __call__(self, data, main_proc=True):
        self.finished.clear()
        if main_proc:
            updated = 0
            while updated < self.word_size - 1:
                while not self.buffer.empty():
                    data += copy.deepcopy(self.buffer.get())
                    updated += 1
            self.finished.set()
        else:
            self.buffer.put(data)
            self.finished.wait()
        return data


class IterBasedTemplate(Prefetcher, IterBasedActorContainer):
    RUNNING = 'running'

    def __init__(self, loader, total_epoch=None, total_iter=None, device=DEVICE, processor=None,
                 cycle=False, collector=None, main_proc=True):
        Prefetcher.__init__(self, loader=loader, device=device, processor=processor, cycle=cycle)
        IterBasedActorContainer.__init__(self, total_epoch=total_epoch, total_iter=total_iter, main_proc=main_proc)
        self.collector = collector
        self.running = True

    @property
    def running(self):
        return self.kwargs_proc.get(IterBasedTemplate.RUNNING, True)

    @running.setter
    def running(self, running):
        self.kwargs_proc[IterBasedTemplate.RUNNING] = running

    @abstractmethod
    def act_iter(self):
        pass

    @abstractmethod
    def act_init(self, *args, **kwargs):
        pass

    @abstractmethod
    def act_ending(self):
        pass

    @abstractmethod
    def act_return(self):
        pass

    def start(self, *args, **kwargs):
        self.update_time(TIMENODE.BEFORE_PROCESS, TIMENODE.BEFORE_INIT)
        self.act_init(*args, **kwargs)
        for actor in self.actors_dct[InitialActor]:
            actor.act_init(self, )
        self.update_time(TIMENODE.AFTER_INIT, TIMENODE.BEFORE_CYCLE)
        while self.running:
            self.update_time(TIMENODE.BEFORE_EPOCH)
            for actor in self.actors_dct[BeforeEpochActor]:
                actor.act_before_epoch(self, )
            self.ind_iter_inep = 0
            iterator = iter(self)
            while self.running:
                self.update_time(TIMENODE.BEFORE_ITER)
                for actor in self.actors_dct[BeforeIterActor]:
                    actor.act_before_iter(self, )
                try:
                    self.batch_data = next(iterator)
                except StopIteration:
                    break
                self.act_iter()
                self.update_time(TIMENODE.AFTER_ITER)
                for actor in self.actors_dct[AfterIterActor]:
                    actor.act_after_iter(self, )
                self.ind_iter = self.ind_iter + 1
                self.ind_iter_inep = self.ind_iter_inep + 1
                self.running = self.running and (self.total_iter is None or self.ind_iter < self.total_iter)

            self.update_time(TIMENODE.AFTER_EPOCH)
            for actor in self.actors_dct[AfterEpochActor]:
                actor.act_after_epoch(self, )
            self.ind_epoch = self.ind_epoch + 1
            self.running = self.running and (self.total_epoch is None or self.ind_epoch < self.total_epoch)

        self.update_time(TIMENODE.AFTER_CYCLE)
        for actor in self.actors_dct[AfterCycleActor]:
            actor.act_after_cycle(self, )
        self.act_ending()
        for actor in self.actors_dct[EndingActor]:
            actor.act_ending(self, )
        self.update_time(TIMENODE.AFTER_PROCESS)
        return self.act_return()


# </editor-fold>


# <editor-fold desc='工具原型'>

class LRScheduler(Actor):
    pass


class IMScheduler(Actor):
    pass


class Broadcaster(BroadCastActor):
    pass


class PrintBasedBroadcaster(Broadcaster):

    def __init__(self, formatter=FORMATTER.BROADCAST):
        self.formatter = formatter

    def act_broadcast(self, trainer, msg, **kwargs):
        print(format_msg(msg, self.formatter))


class LogBasedBroadcaster(Broadcaster):

    def __init__(self, log_pth, with_print=True, log_name='trainer', formatter=FORMATTER.BROADCAST):
        log_pth = os.path.abspath(ensure_extend(log_pth, 'log'))
        self.with_print = with_print
        self.logger = logging.getLogger(name=log_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.parent = None
        ensure_file_dir(log_pth)
        handler = TimedRotatingFileHandler(log_pth, when='D', encoding='utf-8')
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.formatter = formatter

    def act_broadcast(self, trainer, msg, **kwargs):
        msg = format_msg(msg, self.formatter)
        if self.with_print:
            print(msg)
        self.logger.info(destylize_msg(msg))


class Recorder(Actor, DCTExtractable):

    @property
    def name(self):
        return self.__class__.__name__


# </editor-fold>


# <editor-fold desc='loss处理'>

class LossCollector():
    def __init__(self):
        self.loss_dct = OrderedDict()

    @staticmethod
    def check_losses(losses, names):
        for i, name in enumerate(names):
            loss_i = losses[i]
            if torch.isnan(loss_i):
                BROADCAST('nan in loss ' + str(name))
                raise Exception('err loss')
            if torch.isinf(loss_i):
                BROADCAST('inf in loss ' + str(name))
                raise Exception('err loss')
        return None

    @staticmethod
    def process_loss(loss, loss_gain=1):
        if isinstance(loss, dict):
            losses, names = [], []
            for name_i, loss_i in loss.items():
                losses.append(loss_i * loss_gain)
                names.append(name_i)
            loss = sum(losses)
            return loss, names, losses
        elif isinstance(loss, torch.Tensor):
            return loss * loss_gain, [], []
        else:
            raise Exception('err loss')

    def update_loss(self, name, loss):
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        self.loss_dct[name] = loss
        return self

    def update_losses(self, names, losses):
        for name, loss in zip(names, losses):
            self.update_loss(name, loss)
        return self


# </editor-fold>
# <editor-fold desc='检查'>
# 检查梯度
def check_grad(model, loader, accu_step=1, grad_norm=0, loss_gain=8, **kwargs):
    model.train()
    BROADCAST('Checking Grad')
    loader_iter = iter(loader)
    for i in range(accu_step):
        (imgs, labels) = next(loader_iter)
        target = model.labels2tars(labels, **kwargs)
        loss = model.imgs_tars2loss(imgs, target, **kwargs)
        loss, names, losses = LossCollector.process_loss(loss)
        BROADCAST('Loss ', ''.join([n + ' %-10.5f  ' % l for l, n in zip(losses, names)]))
        (loss * loss_gain / accu_step).backward()
    if grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_norm)
    for name, para in model.named_parameters():
        BROADCAST('%-100s' % name + '%10.5f' % para.grad.norm().item())
    return None


# 检查参数
def check_para(model):
    BROADCAST('Checking Para')
    for name, para in model.named_parameters():
        if torch.any(torch.isnan(para)):
            BROADCAST('nan occur in models')
            para.data = torch.where(torch.isnan(para), torch.full_like(para, 0.1), para)
        if torch.any(torch.isinf(para)):
            BROADCAST('inf occur in models')
            para.data = torch.where(torch.isinf(para), torch.full_like(para, 0.1), para)
        max = torch.max(para).item()
        min = torch.min(para).item()
        BROADCAST('Range [ %10.5f' % min + ' , ' + '%10.5f' % max + ']  --- ' + name)
    return None

# </editor-fold>
