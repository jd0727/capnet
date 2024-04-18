import pycocotools
from pycocotools.cocoeval import COCOeval

from data.coco import CoCoDataSet
from tools.interface import *
from utils import MTorchModel


# <editor-fold desc='分类任务'>


# 分类别计算AUC
def auc_per_class(chots_md, cinds_ds, num_cls):
    num_samp = len(cinds_ds)
    aucs = np.zeros(num_cls)
    for i in range(num_cls):
        mask_i_ds = cinds_ds == i
        num_pos = (mask_i_ds).sum()
        num_neg = num_samp - num_pos
        if num_pos == 0 or num_neg == 0:
            aucs[i] = 0
        else:
            # 置信度降序
            confs_cind = chots_md[:, i]
            order = np.argsort(-confs_cind)
            mask_i_ds = mask_i_ds[order]
            # 计算曲线
            tpr_curve = (mask_i_ds).cumsum() / num_pos
            fpr_curve = (~mask_i_ds).cumsum() / num_neg
            # 计算面积
            fpr_curve = np.concatenate(([0.0], fpr_curve))
            fpr_dt = fpr_curve[1:] - fpr_curve[:-1]
            aucs[i] = np.sum(fpr_dt * tpr_curve)
    aucs = np.array(aucs)
    return aucs


# 分类计算prec
def prec_recl_per_class(cinds_md, cinds_ds, num_cls):
    tgs = np.zeros(num_cls, dtype=np.int32)
    tps = np.zeros(num_cls, dtype=np.int32)
    precs = np.zeros(num_cls)
    recls = np.zeros(num_cls)
    f1s = np.zeros(num_cls)
    accs = np.zeros(num_cls)
    for cind in range(num_cls):
        mask_i_ds = cinds_ds == cind
        mask_i_md = cinds_md == cind
        tp = np.sum(mask_i_ds * mask_i_md)
        tn = np.sum(~mask_i_ds * ~mask_i_md)
        fp = np.sum(~mask_i_ds * mask_i_md)
        fn = np.sum(mask_i_ds * ~mask_i_md)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recl = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 / (1 / prec + 1 / recl) if prec > 0 and recl > 0 else 0
        acc = (tp + tn) / len(mask_i_ds) if len(mask_i_ds) > 0 else 0
        precs[cind] = prec
        recls[cind] = recl
        f1s[cind] = f1
        accs[cind] = acc
        tgs[cind] = np.sum(mask_i_ds)
        tps[cind] = tp
    return tgs, tps, precs, recls, f1s, accs


# top精度
def acc_top_nums(chots_md, cinds_ds, top_nums=(1, 5)):
    num_samp = len(cinds_ds)
    corrects = np.zeros(shape=len(top_nums))
    order = np.argsort(-chots_md, axis=-1)
    for i, num in enumerate(top_nums):
        for j, cind_ds in enumerate(cinds_ds):
            if cind_ds in order[j, :num]:
                corrects[i] += 1
    accs = corrects / num_samp
    return accs


# </editor-fold>

# <editor-fold desc='检测任务'>
# 分类别计算AP
def ap_per_class(mask_pos, mask_neg, confs_md, cinds_md, cinds_ds, num_cls=20, interp=False):
    order = np.argsort(-confs_md)
    mask_pos, mask_neg, confs_md, cinds_md = mask_pos[order], mask_neg[order], confs_md[order], cinds_md[order]
    aps = np.zeros(num_cls)
    for cind in range(num_cls):
        mask_pred_pos = cinds_md == cind
        num_gt = (cinds_ds == cind).sum()
        num_pred = mask_pred_pos.sum()
        if num_pred == 0 or num_gt == 0:
            aps[cind] = 0
        else:
            fp_nums = (mask_neg[mask_pred_pos]).cumsum()  # 累加和列表
            tp_nums = (mask_pos[mask_pred_pos]).cumsum()
            # 计算曲线
            recall_curve = tp_nums / (num_gt + 1e-16)
            precision_curve = tp_nums / (tp_nums + fp_nums + 1e-16)
            # 计算面积
            recall_curve = np.concatenate(([0.0], recall_curve, [1.0]))
            precision_curve = np.concatenate(([1.0], precision_curve, [0.0]))

            precision_curve = np.flip(np.maximum.accumulate(np.flip(precision_curve)))
            if interp:
                x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
                ap = np.trapz(np.interp(x, recall_curve, precision_curve), x)  # integrate
            else:  # 'continuous'
                ids = np.where(recall_curve[1:] != recall_curve[:-1])[0]  # points where x axis (recall) changes
                ap = np.sum((recall_curve[ids + 1] - recall_curve[ids]) * precision_curve[ids + 1])  # area under curve
            aps[cind] = ap

            # for i in range(precision_curve.size - 1, 0, -1):
            #     precision_curve[i - 1] = np.maximum(precision_curve[i - 1], precision_curve[i])
            # aps[cind] = np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:])
    aps = np.array(aps)
    return aps


# </editor-fold>

# <editor-fold desc='分割任务'>
# 分类别计算IOU
def iou_per_class(maskss_md, maskss_ds, num_cls, scale=1000):
    lbs_ds = np.zeros(num_cls, dtype=np.int32)
    lbs_md = np.zeros(num_cls, dtype=np.int32)
    precs = np.zeros(num_cls)
    recls = np.zeros(num_cls)
    f1s = np.zeros(num_cls)
    accs = np.zeros(num_cls)
    ious = np.zeros(num_cls)
    vol = maskss_md.shape[0] * maskss_md.shape[1] * maskss_md.shape[2]
    maskss_md = maskss_md > 0
    maskss_ds = maskss_ds > 0
    for cind in range(num_cls):
        mask_ds = maskss_ds[..., cind]
        mask_md = maskss_md[..., cind]
        tp = np.sum(mask_ds * mask_md) / scale
        tn = np.sum(~mask_ds * ~mask_md) / scale
        fp = np.sum(~mask_ds * mask_md) / scale
        fn = np.sum(mask_ds * ~mask_md) / scale
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recl = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 / (1 / prec + 1 / recl) if prec > 0 and recl > 0 else 0
        acc = (tp + tn) / vol if vol > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precs[cind] = prec
        recls[cind] = recl
        f1s[cind] = f1
        accs[cind] = acc
        ious[cind] = iou
        lbs_ds[cind] = np.sum(mask_ds)
        lbs_md[cind] = np.sum(mask_md)
    return lbs_ds, lbs_md, precs, recls, f1s, accs, ious


# 省内存版本
def iou_per_class_eff(tp, tn, fp, fn, num_cls, scale=1000):
    lbs_ds = np.zeros(num_cls, dtype=np.int32)
    lbs_md = np.zeros(num_cls, dtype=np.int32)
    precs = np.zeros(num_cls)
    recls = np.zeros(num_cls)
    f1s = np.zeros(num_cls)
    accs = np.zeros(num_cls)
    ious = np.zeros(num_cls)

    for cind in range(num_cls):
        tp_cind = np.sum(tp[:, cind] / scale)
        tn_cind = np.sum(tn[:, cind] / scale)
        fp_cind = np.sum(fp[:, cind] / scale)
        fn_cind = np.sum(fn[:, cind] / scale)
        vol = tp_cind + tn_cind + fp_cind + fn_cind
        prec = tp_cind / (tp_cind + fp_cind) if (tp_cind + fp_cind) > 0 else 0
        recl = tp_cind / (tp_cind + fn_cind) if (tp_cind + fn_cind) > 0 else 0
        f1 = 2 / (1 / prec + 1 / recl) if prec > 0 and recl > 0 else 0
        acc = (tp_cind + tn_cind) / vol if vol > 0 else 0
        iou = tp_cind / (tp_cind + fp_cind + fn_cind) if (tp_cind + fp_cind + fn_cind) > 0 else 0
        precs[cind] = prec
        recls[cind] = recl
        f1s[cind] = f1
        accs[cind] = acc
        ious[cind] = iou
        lbs_ds[cind] = tp_cind + fp_cind
        lbs_md[cind] = tp_cind + fn_cind
    return lbs_ds, lbs_md, precs, recls, f1s, accs, ious


# </editor-fold>

# <editor-fold desc='Evaler类封装'>


class Evaler(IterBasedTemplate):
    EXTEND = 'pkl'

    def __init__(self, loader, total_epoch=1, device=DEVICE, main_proc=True, collector=None, cache_pth='',
                 **kwargs):
        IterBasedTemplate.__init__(
            self, total_epoch=total_epoch, total_iter=total_epoch * len(loader), loader=loader,
            device=device, processor=None, cycle=False, main_proc=main_proc, collector=collector)
        self.cind2name = loader.cind2name
        self.kwargs_infer = kwargs
        self.labels_cmb = None
        self.cache_pth = cache_pth

    def act_init(self, model, *args, **kwargs):
        if isinstance(model, InferableModel):
            self.model = model
            self.device = model.device if isinstance(model, MTorchModel) else DEVICE
            self.running = True
            self.labels_cmb = []
            self.ind_iter = 0
            self.ind_epoch = 0
            self.model.act_init_infer(self)
        elif isinstance(model, list):
            self.labels_cmb = model
            self.running = False
        elif isinstance(model, str):
            self.broadcast('Using test cache at ' + model)
            buffer = load_pkl(model, extend=Evaler.EXTEND)
            self.labels_cmb = buffer['labels_cmb']
            self.running = False
        else:
            raise Exception('fmt err ' + model.__class__.__name__)
        return None

    def act_return(self):
        self.update_time(TIMENODE.BEFORE_CALC)
        val = self._eval_sum(self.labels_cmb) if self.main_proc else 0
        self.update_time(TIMENODE.AFTER_CALC)
        return val

    def act_ending(self):
        if self.collector is not None:
            self.labels_cmb = self.collector(self.labels_cmb, main_proc=self.main_proc)
        if self.cache_pth is not None and len(self.cache_pth) > 0 and self.main_proc:
            cache_pth = ensure_extend(self.cache_pth, 'pkl')
            self.broadcast('Save test cache at ' + cache_pth)
            save_pkl(file_pth=cache_pth, obj=self.labels_cmb)
        return None

    def act_iter(self):
        imgs, labels_ds = self.batch_data
        labels_md = self.model.act_iter_infer(self, imgs, labels_ds, **self.kwargs_infer)
        self.labels_md = labels_md
        self.update_time(TIMENODE.AFTER_INFER)
        for label_ds, label_md in zip(labels_ds, labels_md):
            label_md.meta = label_ds.meta
            label_md.ctx_from(label_ds)
            label_md.kwargs = label_ds.kwargs
            self.labels_cmb.append((label_ds, label_md))
        return None

    @abstractmethod
    def _eval_sum(self, labels_cmb):
        print('*** core calc opr ***')
        pass

    @staticmethod
    def Accuracy(loader, total_epoch=1, cache_pth='', save_pth='', top_nums=(1, 5), **kwargs):
        return AccuracyEvaler(loader=loader, total_epoch=total_epoch, cache_pth=cache_pth,
                              save_pth=save_pth, top_nums=top_nums, **kwargs)

    @staticmethod
    def PrecRcal(loader, total_epoch=1, cache_pth='', save_pth='', **kwargs):
        return PrecRcalEvaler(loader=loader, total_epoch=total_epoch, cache_pth=cache_pth, save_pth=save_pth, **kwargs)

    @staticmethod
    def AUC(loader, total_epoch=1, cache_pth='', save_pth='', **kwargs):
        return AUCEvaler(loader=loader, total_epoch=total_epoch, cache_pth=cache_pth, save_pth=save_pth, **kwargs)

    @staticmethod
    def BoxMAP(loader, total_epoch=1, cache_pth='', save_pth='', criterion=0.5, ignore_class=False, **kwargs):
        return BoxMAPEvaler(loader=loader, total_epoch=total_epoch, cache_pth=cache_pth, save_pth=save_pth,
                            criterion=criterion, ignore_class=ignore_class, **kwargs)

    @staticmethod
    def InstMAP(loader, total_epoch=1, cache_pth='', save_pth='', criterion=0.5, **kwargs):
        return InstMAPEvaler(loader=loader, total_epoch=total_epoch, cache_pth=cache_pth, save_pth=save_pth,
                             criterion=criterion, **kwargs)

    @staticmethod
    def MIOU(loader, total_epoch=1, cache_pth='', save_pth='', **kwargs):
        return MIOUEvaler(loader=loader, total_epoch=total_epoch, cache_pth=cache_pth, save_pth=save_pth, **kwargs)


class EvalerBroadcaster(IntervalTrigger, Broadcaster, BeforeEpochActor, AfterIterActor, AfterCycleActor,
                        InitDCTExtractable):

    def __init__(self, step=50, offset=0, first=False, last=False):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.periods_infer = []

    def get_time_msg(self, evaler, **kwargs):
        return ''.join([name + ' %-5.4f ' % val for name, val in evaler.periods.items()])

    def get_eta_msg(self, evaler, **kwargs):
        sec = evaler.eta
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)

    def act_after_iter(self, evaler, **kwargs):
        if self.trigger(evaler.ind_iter_inep, len(evaler)):
            msg = 'Iter %06d ' % (evaler.ind_iter + 1) + '[ %04d ]' % (evaler.ind_iter_inep + 1) + ' | '
            msg += self.get_time_msg(evaler, **kwargs) + ' | '
            msg += 'ETA ' + self.get_eta_msg(evaler, **kwargs) + ' | '
            evaler.broadcast(msg)
        self.periods_infer.append(evaler._collect_period(PERIOD.INFER))

    def act_before_epoch(self, evaler, **kwargs):
        msg = '< Eval > Epoch %d' % (evaler.ind_epoch + 1) + '  Data %d' % evaler.num_data + \
              '  Batch %d' % evaler.num_batch + '  BatchSize %d' % evaler.batch_size + \
              '  ImgSize ' + str(evaler.img_size) + '  ETA ' + self.get_eta_msg(evaler, **kwargs)
        evaler.broadcast(msg)

    def act_after_cycle(self, evaler, **kwargs):
        mean_infer = np.mean(self.periods_infer)
        msg = 'Average infer time %f' % (mean_infer / evaler.batch_size) \
              + ' [ %f' % mean_infer + ' / %d ]' % evaler.batch_size
        evaler.broadcast(msg)


class PrintBasedEvalerBroadcaster(PrintBasedBroadcaster, EvalerBroadcaster):
    def __init__(self, step=50, offset=0, first=True, last=True,
                 formatter=FORMATTER.BROADCAST):
        super(PrintBasedEvalerBroadcaster, self).__init__(formatter=formatter)
        super(PrintBasedBroadcaster, self).__init__(step=step, offset=offset, first=first, last=last)


class LogBasedEvalerBroadcaster(LogBasedBroadcaster, EvalerBroadcaster):
    def __init__(self, log_pth, step=50, offset=0, first=True, last=True, formatter=FORMATTER.BROADCAST,
                 with_print=True, log_name='evaler'):
        super(LogBasedEvalerBroadcaster, self).__init__(
            log_pth, with_print=with_print, log_name=log_name, formatter=formatter)
        super(LogBasedBroadcaster, self).__init__(step=step, offset=offset, first=first, last=last)


class EvalerBuffer(AfterCycleActor, InitDCTExtractable):

    def __init__(self, cache_pth):
        self.cache_pth = cache_pth

    def act_after_cycle(self, evaler, **kwargs):
        cache_pth = ensure_extend(self.cache_pth, 'pkl')
        save_pkl(obj=evaler.labels_cmb, file_pth=cache_pth)

    @property
    def name(self):
        return self.__class__.__name__


class EvalActor(Actor):
    def __init__(self, evaler):
        self.evaler = evaler

    @property
    def name(self):
        return self.__class__.__name__


class EpochBasedEvalActor(EvalActor, IntervalTrigger, AfterEpochActor, InitDCTExtractable):
    def __init__(self, evaler, step, save_pth='', offset=0, first=False, last=False, best_perfmce=0):
        EvalActor.__init__(self, evaler)
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.save_pth = save_pth
        self.best_perfmce = best_perfmce

    def act_after_epoch(self, trainer, **kwargs):
        if not self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            return None
        perfmce = self.evaler.start(trainer.model)
        if perfmce > self.best_perfmce and len(self.save_pth) > 0 and trainer.main_proc:
            self.best_perfmce = perfmce
            save_pth = os.path.splitext(self.save_pth)[0] + '_best'
            model_pth = os.path.abspath(ensure_extend(save_pth, 'pth'))
            trainer.broadcast('Save best model to ' + model_pth)
            trainer.model.save(model_pth)


class IterBasedEvalActor(EvalActor, IntervalTrigger, AfterIterActor, InitDCTExtractable):

    def __init__(self, evaler, step, save_pth='', offset=0, first=False, last=False, best_perfmce=0):
        EvalActor.__init__(self, evaler)
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.save_pth = save_pth
        self.best_perfmce = best_perfmce

    def act_after_iter(self, trainer, **kwargs):
        if not self.trigger(ind=trainer.ind_iter, total=trainer.total_iter):
            return None
        perfmce = self.evaler.start(trainer.model)
        if perfmce > self.best_perfmce and len(self.save_pth) > 0:
            save_pth = os.path.splitext(self.save_pth)[0] + '_best'
            model_pth = os.path.abspath(ensure_extend(save_pth, 'pth'))
            trainer.broadcast('Save best model to ' + model_pth)
            trainer.model.save(model_pth)


# </editor-fold>


class MIOUEvaler(Evaler):

    def __init__(self, loader, total_epoch=1, save_pth='', **kwargs):
        super().__init__(loader, total_epoch, **kwargs)
        self.save_pth = save_pth

    def _eval_sum(self, labels_cmb):
        data = MIOUEvaler.calc(labels_cmb, num_cls=self.loader.num_cls, cind2name=self.loader.cind2name)
        self.broadcast_dataframe(data)
        if self.save_pth: data.save(self.save_pth, index=False)
        return np.array(data['IOU'])[-1]

    @staticmethod
    def calc(labels_cmb, num_cls, cind2name=None, ignore_empty=True):
        tp = np.zeros(shape=(len(labels_cmb), num_cls))
        fp = np.zeros(shape=(len(labels_cmb), num_cls))
        tn = np.zeros(shape=(len(labels_cmb), num_cls))
        fn = np.zeros(shape=(len(labels_cmb), num_cls))
        for i, (label_ds, label_md) in enumerate(labels_cmb):
            masks_md = label_md.export_masksN_enc(num_cls=num_cls, img_size=label_md.img_size)
            masks_ds = label_ds.export_masksN_enc(num_cls=num_cls, img_size=label_ds.img_size)
            for cind in range(num_cls):
                mask_ds_cind = masks_ds == cind
                mask_md_cind = masks_md == cind
                tp[i, cind] = np.sum(mask_ds_cind * mask_md_cind)
                tn[i, cind] = np.sum(~mask_ds_cind * ~mask_md_cind)
                fp[i, cind] = np.sum(~mask_ds_cind * mask_md_cind)
                fn[i, cind] = np.sum(mask_ds_cind * ~mask_md_cind)
            del masks_md, masks_ds
        lbs_ds, lbs_md, precs, recls, f1s, accs, ious = iou_per_class_eff(tp, tn, fp, fn, num_cls=num_cls, scale=1000)
        data = pd.DataFrame(columns=['Class', 'Target(k)', 'Pred(k)', 'Percison', 'Recall', 'F1', 'Accuracy', 'IOU'])
        for i in range(num_cls):
            name = cind2name(i) if cind2name is not None else i
            if ignore_empty and lbs_ds[i] == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name, 'Target(k)': lbs_ds[i], 'Pred(k)': lbs_md[i],
                'Percison': precs[i], 'Recall': recls[i], 'F1': f1s[i], 'Accuracy': accs[i], 'IOU': ious[i]
            }, index=[0])])
        if num_cls > 1:
            lb_ds_sum = np.sum(np.array(lbs_ds))
            lb_md_sum = np.sum(np.array(lbs_md))
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total', 'Target(k)': np.sum(data['Target(k)']), 'Pred(k)': np.sum(data['Pred(k)']),
                'Percison': np.sum(np.array(data['Percison']) * lbs_md) / max(lb_md_sum, 1),
                'Recall': np.sum(np.array(data['Recall']) * lbs_ds) / max(lb_ds_sum, 1),
                'F1': np.sum(data['F1']) / max(np.sum(lbs_ds > 0), 1),
                'Accuracy': np.average(data['Accuracy']),
                'IOU': np.sum(data['IOU']) / max(np.sum(lbs_ds > 0), 1),
            }, index=[0])])
        return data


class AccuracyEvaler(Evaler):
    def _eval_sum(self, labels_cmb):
        data = AccuracyEvaler.calc(labels_cmb, top_nums=self.top_nums)
        self.broadcast_dataframe(data)
        if self.save_pth: data.save(self.save_pth, index=False)
        return np.array(data[0])

    def __init__(self, loader, total_epoch=1, save_pth='', top_nums=(1, 5), **kwargs):
        super().__init__(loader, total_epoch, **kwargs)
        self.top_nums = top_nums
        self.save_pth = save_pth

    @staticmethod
    def calc(labels_cmb, top_nums=(1, 5)):
        labels_ds, labels_md = zip(*labels_cmb)
        chots_md = cates2chotsN(labels_md)
        cinds_ds = cates2cindsN(labels_ds)
        accs = acc_top_nums(chots_md, cinds_ds, top_nums=top_nums)
        data = pd.DataFrame(columns=['Top%d' % n for n in top_nums], index=['Accuracy'],
                            data=['%5.5f' % a for a in accs])
        return data


class PrecRcalEvaler(Evaler):
    def _eval_sum(self, labels_cmb):
        data = PrecRcalEvaler.calc(labels_cmb, num_cls=self.loader.num_cls, cind2name=self.loader.cind2name)
        self.broadcast_dataframe(data)
        if self.save_pth: data.save(self.save_pth, index=False)
        return np.array(data['F1'])[-1]

    def __init__(self, loader, total_epoch=1, save_pth='', **kwargs):
        super(PrecRcalEvaler, self).__init__(loader, total_epoch=total_epoch, **kwargs)
        self.save_pth = save_pth

    @staticmethod
    def calc(labels_cmb, num_cls, cind2name=None, ignore_empty=True):
        labels_ds, labels_md = zip(*labels_cmb)
        chots_md = cates2chotsN(labels_md)
        cinds_md = chotsN2cindsN(chots_md)
        cinds_ds = cates2cindsN(labels_ds)
        lbs_ds, lbs_md, precs, recls, f1s, accs = prec_recl_per_class(cinds_md, cinds_ds, num_cls=num_cls)
        data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'Percison', 'Recall', 'F1', 'Accuracy'])
        for i in range(num_cls):
            name = cind2name(i) if cind2name is not None else i
            if ignore_empty and lbs_ds[i] == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name, 'Target': lbs_ds[i], 'Pred': lbs_md[i],
                'Percison': precs[i], 'Recall': recls[i], 'F1': f1s[i], 'Accuracy': accs[i]
            }, index=[0])])
        if num_cls > 1:
            lb_ds_sum = np.sum(np.array(lbs_ds))
            lb_md_sum = np.sum(np.array(lbs_md))
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total', 'Target': lb_ds_sum, 'Pred': lb_md_sum,
                'Percison': np.sum(np.array(data['Percison']) * lbs_md) / lb_md_sum,
                'Recall': np.sum(np.array(data['Recall']) * lbs_ds) / lb_ds_sum,
                'F1': np.sum(data['F1']) / np.sum(lbs_ds > 0), 'Accuracy': np.average(data['Accuracy'])
            }, index=[0])])
        return data


class F1Evaler(Evaler):
    def _eval_sum(self, labels_cmb):
        data = PrecRcalEvaler.calc(labels_cmb, num_cls=self.loader.num_cls, cind2name=self.loader.cind2name)
        self.broadcast_dataframe(data)
        if self.save_pth: data.save(self.save_pth, index=False)
        return np.array(data['F1'])[-1]

    def __init__(self, loader, total_epoch=1, save_pth='', **kwargs):
        super().__init__(loader, total_epoch, **kwargs)
        self.save_pth = save_pth


class AUCEvaler(Evaler):
    def _eval_sum(self, labels_cmb):
        data = AUCEvaler.calc(labels_cmb, num_cls=self.loader.num_cls, cind2name=self.loader.cind2name)
        self.broadcast_dataframe(data)
        if self.save_pth: data.save(self.save_pth, index=False)
        return np.array(data['AUC'])[-1]

    def __init__(self, loader, total_epoch=1, save_pth='', **kwargs):
        super().__init__(loader, total_epoch, **kwargs)
        self.save_pth = save_pth

    @staticmethod
    def calc(labels_cmb, num_cls, cind2name=None, ):
        labels_ds, labels_md = zip(*labels_cmb)
        chots_md = cates2chotsN(labels_md)
        cinds_md = chotsN2cindsN(chots_md)
        cinds_ds = cates2cindsN(labels_ds)
        aucs = auc_per_class(chots_md, cinds_ds, num_cls)
        data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'Pos', 'AUC'])
        for i in range(num_cls):
            mask_i_ds = cinds_ds == i
            mask_i_md = cinds_md == i
            name = cind2name(i) if cind2name is not None else i
            data = pd.concat([data, pd.DataFrame.from_dict({
                'Class': name, 'Target': np.sum(mask_i_ds), 'Pred': np.sum(mask_i_md),
                'Pos': np.sum(mask_i_ds * mask_i_md), 'AUC': aucs[i]
            })])
        if num_cls > 1:
            data = data.append({
                'Class': 'Total', 'Target': np.sum(data['Target']), 'Pred': np.sum(data['Pred']),
                'Pos': np.sum(data['Pos']), 'AUC': np.average(data['AUC'])
            }, ignore_index=True)
        return data


class CRITERION:
    VOC = 0.5
    COCO = tuple([0.5 + i * 0.05 for i in range(10)])
    COCO_STD = 'coco_std'


def _summarize_coco_eval(coco_eval, cind2name=None):
    def _summarize_coco(coco_eval, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        nums = np.sum(s > -1, axis=(0, 1, 3))
        sum_vals = np.sum(np.where(s == -1, np.zeros_like(s), s), axis=(0, 1, 3))
        mean_vals = np.where(nums > 0, sum_vals / nums, np.full_like(sum_vals, fill_value=0))
        total_val = np.mean(s[s > -1]) if np.any(nums > 0) else 0
        pkd_vals = np.concatenate([mean_vals, [total_val]], axis=0)
        return pkd_vals

    data = pd.DataFrame()
    cls_names = [cind2name(i) if cind2name else str(i) for i in coco_eval.params.catIds]
    data['Class'] = cls_names + ['Total']
    data['AP'] = _summarize_coco(coco_eval, iouThr=None, areaRng='all', maxDets=100)
    data['AP50'] = _summarize_coco(coco_eval, iouThr=0.5, areaRng='all', maxDets=coco_eval.params.maxDets[2])
    data['AP75'] = _summarize_coco(coco_eval, iouThr=0.75, areaRng='all', maxDets=coco_eval.params.maxDets[2])
    data['APs'] = _summarize_coco(coco_eval, iouThr=None, areaRng='small', maxDets=coco_eval.params.maxDets[2])
    data['APm'] = _summarize_coco(coco_eval, iouThr=None, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
    data['APl'] = _summarize_coco(coco_eval, iouThr=None, areaRng='large', maxDets=coco_eval.params.maxDets[2])
    return data


def eval_json(json_pth_md, json_pth_lb, eval_type='bbox'):
    json_dict_md = load_json(json_pth_md)
    if isinstance(json_dict_md, dict):
        annotations = json_dict_md['annotations']
    elif isinstance(json_dict_md, list):
        annotations = json_dict_md
    else:
        raise Exception('err json')

    for anno in annotations:
        if 'score' not in anno.keys():
            anno['score'] = 1

    json_dict_lb = load_json(json_pth_lb)
    cates = json_dict_lb['categories']
    cate_dict = dict([(cate['id'], cate['name']) for cate in cates])
    cind2name = lambda cind: cate_dict[cind]
    # cls_names = list(sorted([cate_dict['name'] for cate_dict in coco_tool.cats.values()]))
    # coco_md = CoCoDataSet.json_dct2coco_obj(json_dict_md)
    coco_lb = CoCoDataSet.json_dct2coco_obj(json_dict_lb)
    coco_md = coco_lb.loadRes(json_dict_md)

    coco_eval = pycocotools.cocoeval.COCOeval(coco_lb, coco_md, eval_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    data = _summarize_coco_eval(coco_eval, cind2name=cind2name)
    return data


class BoxMAPEvaler(Evaler):
    def __init__(self, loader, total_epoch=1, save_pth='', criterion=0.5, ignore_class=False, **kwargs):
        Evaler.__init__(self, loader=loader, total_epoch=total_epoch, **kwargs)
        self.criterion = criterion
        self.ignore_class = ignore_class
        self.save_pth = save_pth

    def _eval_sum(self, labels_cmb):
        kwargs = dict(labels_cmb=labels_cmb, label_process=BoxMAPEvaler.label_process, num_cls=self.loader.num_cls,
                      cind2name=self.loader.cind2name, ignore_class=self.ignore_class)
        if self.criterion == CRITERION.COCO_STD:
            labels_ds, labels_pd = zip(*labels_cmb)
            anno_pd = CoCoDataSet.labels2json_lst(labels_pd, with_score=True, with_rgn=False)
            coco_ds = CoCoDataSet.labels2coco_obj(labels_ds, with_score=False, with_rgn=False)
            coco_pd = coco_ds.loadRes(anno_pd)
            coco_eval = pycocotools.cocoeval.COCOeval(coco_ds, coco_pd, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            data = _summarize_coco_eval(coco_eval, cind2name=self.loader.cind2name)
        elif isinstance(self.criterion, Iterable):
            kwargs['iou_thress'] = self.criterion
            data = BoxMAPEvaler.calc_coco(**kwargs)
        else:
            kwargs['iou_thres'] = self.criterion
            data = BoxMAPEvaler.calc_voc(**kwargs)
        self.broadcast_dataframe(data)
        if self.save_pth is not None and len(self.save_pth) > 0 and self.main_proc:
            save_pth = ensure_extend(self.save_pth, 'xlsx')
            self.broadcast('Save eval result at ' + save_pth)
            data.to_excel(save_pth, index=False)
        return np.array(data['AP'])[-1]

    @staticmethod
    def label_process(label_ds, label_md):
        assert isinstance(label_ds, BoxesLabel) or isinstance(label_ds, InstsLabel), \
            'fmt err ' + label_ds.__class__.__name__
        assert isinstance(label_md, BoxesLabel) or isinstance(label_ds, InstsLabel), \
            'fmt err ' + label_ds.__class__.__name__
        confs = label_md.export_confsN()
        order = np.argsort(-confs)
        label_md = label_md[order]
        confs = confs[order]
        if np.any([isinstance(box.border, XYWHABorder) for box in label_ds]) or \
                np.any([isinstance(box.border, XYWHABorder) for box in label_md]):
            xywhas_ds, cinds_ds = label_ds.export_xywhasN(), label_ds.export_cindsN()
            xywhas_md, cinds_md = label_md.export_xywhasN(), label_md.export_cindsN()
            iou_mat = ropr_mat_xywhasN(xywhas_md, xywhas_ds, opr_type=OPR_TYPE.IOU)
        else:
            xyxys_ds, cinds_ds = label_ds.export_xyxysN(), label_ds.export_cindsN()
            xyxys_md, cinds_md = label_md.export_xyxysN(), label_md.export_cindsN()
            iou_mat = ropr_mat_xyxysN(xyxys_md, xyxys_ds, opr_type=OPR_TYPE.IOU)
        diffs = label_ds.export_ignoresN()
        return cinds_md, confs, cinds_ds, diffs, iou_mat

    @staticmethod
    def match_core(cinds_md, cinds_ds, diffs, iou_mat, iou_thres=0.5, ignore_class=False):
        mask_pos = np.full(shape=len(cinds_md), fill_value=False)
        mask_neg = np.full(shape=len(cinds_md), fill_value=False)
        if len(cinds_ds) == 0:
            return mask_pos, ~mask_neg
        elif len(cinds_md) == 0:
            return mask_pos, mask_neg
        else:
            for k in range(len(cinds_md)):
                if not ignore_class:  # 不同分类之间不能匹配比较
                    iou_mat[k, ~(cinds_md[k] == cinds_ds)] = 0
                    ind_ds = np.argmax(iou_mat[k, :])
                else:  # 动态改变类别
                    ind_ds = np.argmax(iou_mat[k, :])
                    cinds_md[k] = cinds_ds[ind_ds]
                if iou_mat[k, ind_ds] > iou_thres:
                    if not diffs[ind_ds]:
                        mask_pos[k] = True
                    iou_mat[:, ind_ds] = 0  # 防止重复匹配
                else:
                    mask_neg[k] = True
            return mask_pos, mask_neg

    @staticmethod
    def match_core_v2(cinds_md, cinds_ds, diffs, iou_mat, iou_thres=0.5, ignore_class=False):
        mask_pos = np.full(shape=len(cinds_md), fill_value=False)
        mask_neg = np.full(shape=len(cinds_md), fill_value=False)
        if len(cinds_ds) == 0:
            return mask_pos, ~mask_neg
        elif len(cinds_md) == 0:
            return mask_pos, mask_neg
        else:
            fltr_iou = (iou_mat >= iou_thres)
            if not ignore_class:
                fltr_cls = cinds_md[:, None] == cinds_ds[None, :]
                fltr_iou *= fltr_cls
            ids_md, ids_ds = np.nonzero(fltr_iou)
            ious = iou_mat[ids_md, ids_ds]
            order = np.argsort(-ious)
            ids_md, ids_ds, ious = ids_md[order], ids_ds[order], ious[order]
            fltr_repeat = np.unique(ids_ds, return_index=True)[1]
            ids_md, ids_ds, ious = ids_md[fltr_repeat], ids_ds[fltr_repeat], ious[fltr_repeat]
            fltr_repeat = np.unique(ids_md, return_index=True)[1]
            ids_md, ids_ds, ious = ids_md[fltr_repeat], ids_ds[fltr_repeat], ious[fltr_repeat]
            mask_pos[ids_md] = True
            mask_neg = ~mask_pos * ~diffs
            return mask_pos, mask_neg

    @staticmethod
    def calc_voc(labels_cmb, label_process, num_cls, cind2name=None, iou_thres=0.5, ignore_class=False,
                 ignore_empty=True):
        mask_md_pos = [np.zeros(shape=(0,), dtype=bool)]
        mask_md_neg = [np.zeros(shape=(0,), dtype=bool)]
        cinds_ds_acpt = [np.zeros(shape=(0,))]
        cinds_md_acpt = [np.zeros(shape=(0,))]
        confs_md_acpt = [np.zeros(shape=(0,))]
        for i, (label_ds, label_md) in enumerate(labels_cmb):
            cinds_md, confs, cinds_ds, diffs, iou_mat = label_process(label_ds, label_md)
            mask_pos_i, mask_neg_i = BoxMAPEvaler.match_core(
                cinds_md, cinds_ds, diffs, iou_mat, iou_thres=iou_thres, ignore_class=ignore_class)
            cinds_ds_acpt.append(cinds_ds[~diffs])
            cinds_md_acpt.append(cinds_md)
            confs_md_acpt.append(confs)
            mask_md_pos.append(mask_pos_i)
            mask_md_neg.append(mask_neg_i)
        cinds_md_acpt = np.concatenate(cinds_md_acpt, axis=0)
        confs_md_acpt = np.concatenate(confs_md_acpt, axis=0)
        cinds_ds_acpt = np.concatenate(cinds_ds_acpt, axis=0)
        mask_md_pos = np.concatenate(mask_md_pos, axis=0)
        mask_md_neg = np.concatenate(mask_md_neg, axis=0)

        aps = ap_per_class(mask_md_pos, mask_md_neg, confs_md_acpt, cinds_md_acpt, cinds_ds_acpt, num_cls=num_cls)
        data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'Pos', 'Recall', 'Precision', 'F1', 'Ign', 'AP'])
        for cind in range(num_cls):
            mask_md_cind = cinds_md_acpt == cind
            if ignore_empty and np.sum(cinds_ds_acpt == cind) == 0:
                continue
            num_pred = np.sum(mask_md_cind)
            num_pos = np.sum(mask_md_pos[mask_md_cind])
            num_neg = np.sum(mask_md_neg[mask_md_cind])
            num_ign = np.sum((~mask_md_neg[mask_md_cind]) * (~mask_md_pos[mask_md_cind]))
            name = cind2name(cind) if cind2name is not None else str(cind)
            num_tar = np.sum(cinds_ds_acpt == cind)
            prec = num_pos / max(num_pred, 1)
            recl = num_pos / max(num_tar, 1)
            f1 = 2 * recl * prec / (prec + recl + 1e-7)
            data = pd.concat([data, pd.DataFrame({
                'Class': name, 'Target': num_tar, 'Pred': num_pred, 'Precision': prec, 'F1': f1,
                'Pos': num_pos, 'Recall': recl, 'Ign': num_ign, 'AP': aps[cind]
            }, index=[0])])
        if num_cls > 1:
            num_cls_valid = np.sum(np.array(data['Target']) > 0)
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total', 'Target': np.sum(data['Target']), 'Pred': np.sum(data['Pred']),
                'Pos': np.sum(data['Pos']), 'Recall': np.sum(data['Recall']) / num_cls_valid,
                'Precision': np.sum(data['Precision']) / num_cls_valid,
                'F1': np.sum(data['F1']) / num_cls_valid,
                'Ign': np.sum(data['Ign']), 'AP': np.sum(data['AP']) / num_cls_valid
            }, index=[0])])
        return data

    @staticmethod
    def calc_coco(labels_cmb, label_process, num_cls, cind2name=None, iou_thress=(0.5, 0.55, 0.6), ignore_class=False,
                  ignore_empty=True):
        num_thres = len(iou_thress)
        mask_md_pos = [[np.zeros(shape=(0,), dtype=bool)] for _ in range(num_thres)]
        mask_md_neg = [[np.zeros(shape=(0,), dtype=bool)] for _ in range(num_thres)]
        cinds_ds_acpt = [np.zeros(shape=(0,))]
        cinds_md_acpt = [np.zeros(shape=(0,))]
        confs_md_acpt = [np.zeros(shape=(0,))]
        for i, (label_ds, label_md) in enumerate(labels_cmb):
            cinds_md, confs, cinds_ds, diffs, iou_mat = label_process(label_ds, label_md)

            for j, iou_thres in enumerate(iou_thress):
                iou_mat_j = copy.deepcopy(iou_mat)
                mask_pos_i, mask_neg_i = BoxMAPEvaler.match_core(
                    cinds_md, cinds_ds, diffs, iou_mat_j, iou_thres=iou_thres, ignore_class=ignore_class)
                mask_md_pos[j].append(mask_pos_i)
                mask_md_neg[j].append(mask_neg_i)

            cinds_ds_acpt.append(cinds_ds[~diffs])
            cinds_md_acpt.append(cinds_md)
            confs_md_acpt.append(confs)

        cinds_md_acpt = np.concatenate(cinds_md_acpt, axis=0)
        confs_md_acpt = np.concatenate(confs_md_acpt, axis=0)
        cinds_ds_acpt = np.concatenate(cinds_ds_acpt, axis=0)

        aps = []
        for j in range(num_thres):
            mask_md_pos_j = np.concatenate(mask_md_pos[j], axis=0)
            mask_md_neg_j = np.concatenate(mask_md_neg[j], axis=0)
            aps_j = ap_per_class(mask_md_pos_j, mask_md_neg_j, confs_md_acpt, cinds_md_acpt, cinds_ds_acpt,
                                 num_cls=num_cls)
            aps.append(aps_j)
        aps = np.stack(aps, axis=0)
        ap_aver = np.mean(aps, axis=0)

        names_ap = ['@ %.2f' % iou_thres for iou_thres in iou_thress]
        data = pd.DataFrame(columns=['Class', 'Target', 'AP'])
        for cind in range(num_cls):
            name = cind2name(cind) if cind2name is not None else str(cind)
            if ignore_empty and np.sum(cinds_ds_acpt == cind) == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name, 'Target': np.sum(cinds_ds_acpt == cind), 'AP': ap_aver[cind]}, index=[0])])
        for name_ap, ap in zip(names_ap, aps):
            data[name_ap] = ap
        data = data.astype(object)
        if num_cls > 1:
            num_valid = np.sum(np.array(data['Target']) > 0)
            data_dict = {'Class': 'Total', 'Target': np.sum(data['Target']), 'AP': np.sum(data['AP']) / num_valid}
            for name_ap, ap in zip(names_ap, aps):
                data_dict[name_ap] = np.sum(data[name_ap]) / num_valid
            data = pd.concat([data, pd.DataFrame(data_dict, index=[0])])
        return data


class InstMAPEvaler(Evaler):
    def __init__(self, loader, total_epoch=1, save_pth='', criterion=0.5, ignore_class=False, **kwargs):
        super().__init__(loader, total_epoch, **kwargs)
        self.criterion = criterion
        self.save_pth = save_pth
        self.ignore_class = ignore_class

    def _eval_sum(self, labels_cmb):
        kwargs = dict(labels_cmb=labels_cmb, label_process=InstMAPEvaler.label_process, num_cls=self.loader.num_cls,
                      cind2name=self.loader.cind2name, ignore_class=self.ignore_class)
        if self.criterion == CRITERION.COCO_STD:
            labels_ds, labels_pd = zip(*labels_cmb)
            anno_pd = CoCoDataSet.labels2json_lst(labels_pd, name2cind_remapper=None, with_score=True, with_rgn=True,
                                                  img_id_mapper=None, as_list=False)
            coco_ds = CoCoDataSet.labels2coco_obj(labels_ds, with_score=False, with_rgn=True, as_list=False)
            coco_pd = coco_ds.loadRes(anno_pd)
            coco_eval = COCOeval(coco_ds, coco_pd, 'segm')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            data = _summarize_coco_eval(coco_eval, cind2name=self.loader.cind2name)
        elif isinstance(self.criterion, Iterable):
            kwargs['iou_thress'] = self.criterion
            data = BoxMAPEvaler.calc_coco(**kwargs)
        else:
            kwargs['iou_thres'] = self.criterion
            data = BoxMAPEvaler.calc_voc(**kwargs)
        self.broadcast_dataframe(data)
        if self.save_pth: data.to_excel(self.save_pth, index=False)
        return np.array(data['AP'])[-1]

    @staticmethod
    def label_process(label_ds, label_md):
        assert isinstance(label_ds, InstsLabel), 'fmt err ' + label_ds.__class__.__name__
        assert isinstance(label_md, InstsLabel), 'fmt err ' + label_md.__class__.__name__

        confs = label_md.export_cindsN()
        order = np.argsort(-confs)
        label_md = label_md[order]
        confs = confs[order]
        xyxys_md, cinds_md, masks_md = label_md.export_rgn_xyxysN_cindsN_maskNs_ref()
        xyxys_ds, cinds_ds, masks_ds = label_ds.export_rgn_xyxysN_cindsN_maskNs_ref()
        iou_mat = ropr_mat_xyxysN_maskNs(xyxys1=xyxys_md, xyxys2=xyxys_ds,
                                         masks1=masks_md, masks2=masks_ds, opr_type=OPR_TYPE.IOU)
        diffs = label_ds.export_ignoresN()
        return cinds_md, confs, cinds_ds, diffs, iou_mat

    # </editor-fold>
