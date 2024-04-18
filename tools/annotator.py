from tools import EvalerBroadcaster
from tools.interface import *
from utils.visual import _pilrnd_label


class Annotator(IterBasedTemplate):
    EXTEND = 'pkl'

    def __init__(self, loader, total_epoch=1, device=DEVICE, main_proc=True, collector=None,
                 with_recover=True, **kwargs):
        IterBasedTemplate.__init__(self, total_epoch=total_epoch, total_iter=total_epoch * len(loader),
                                   loader=loader, device=device, processor=None, cycle=False, main_proc=main_proc,
                                   collector=collector)
        self.cind2name = loader.cind2name
        self.kwargs_infer = kwargs
        self.labels_cmb = None
        self.with_recover = with_recover

    def act_init(self, model, *args, **kwargs):
        self.model = model
        self.running = True
        self.labels_cmb = []
        self.ind_iter = 0
        self.ind_epoch = 0
        self.model.act_init_infer(self)
        return None

    def act_return(self):
        labels_dct = {}
        self.broadcast('Cluster %d labels' % len(self.labels_cmb))
        for label_cmb in self.labels_cmb:
            label_ds, label_md = label_cmb
            if label_ds.meta not in labels_dct.keys():
                labels_dct[label_ds.meta] = [label_cmb]
            else:
                labels_dct[label_ds.meta].append(label_cmb)
        return labels_dct

    def act_ending(self):
        if self.collector is not None:
            self.labels_cmb = self.collector(self.labels_cmb, main_proc=self.main_proc)
        return None

    def act_iter(self):
        imgs, labels_ds = self.batch_data
        self.update_time(TIMENODE.BEFORE_INFER)
        labels_md = self.model.act_iter_infer(self, imgs, labels_ds, **self.kwargs_infer)
        self.labels_md = labels_md
        self.update_time(TIMENODE.AFTER_INFER)
        for label_ds, label_md in zip(labels_ds, labels_md):
            label_md.info_from(label_ds)
            if self.with_recover:
                label_ds.recover()
                label_md.recover()
            self.labels_cmb.append((label_ds, label_md))
        return None


class AnnotatorImageSaver(AfterIterActor, InitialActor):
    def act_init(self, container, **kwargs):
        container.update_period(name='imgsave', period_pair=PERIOD.IMG_SAVE)

    def __init__(self, save_dir, only_md=True, **kwargs):
        self.save_dir = save_dir
        self.only_md = only_md
        ensure_folder_pth(save_dir)
        self.kwargs = kwargs

    def act_after_iter(self, container, **kwargs):
        imgs, labels_ds = container.batch_data
        labels_md = container.labels_md
        container.update_time(TIMENODE.BEFORE_IMG_SAVE)
        for img, label_ds, label_md in zip(imgs, labels_ds, labels_md):
            imgP_md = _pilrnd_label(img, label_md)
            if self.only_md:
                imgP_md.save(os.path.join(self.save_dir, ensure_extend(label_ds.meta, 'jpg')))
            else:
                imgP_ds = _pilrnd_label(img, label_ds)
                imgP_ds.save(os.path.join(self.save_dir, ensure_extend(label_ds.meta + '_ds', 'jpg')))
                imgP_md.save(os.path.join(self.save_dir, ensure_extend(label_ds.meta + '_md', 'jpg')))
        container.update_time(TIMENODE.AFTER_IMG_SAVE)


class AnnotatorBroadcaster(EvalerBroadcaster):

    def act_before_epoch(self, evaler, **kwargs):
        msg = '< Annotate > Epoch %d' % (evaler.ind_epoch + 1) + '  Data %d' % evaler.num_data + \
              '  Batch %d' % evaler.num_batch + '  BatchSize %d' % evaler.batch_size + \
              '  ImgSize ' + str(evaler.img_size) + '  ETA ' + self.get_eta_msg(evaler, **kwargs)
        evaler.broadcast(msg)


class PrintBasedAnnotatorBroadcaster(PrintBasedBroadcaster, AnnotatorBroadcaster):
    def __init__(self, step=50, offset=0, first=True, last=True, formatter=FORMATTER.BROADCAST):
        super(PrintBasedAnnotatorBroadcaster, self).__init__(formatter=formatter)
        AnnotatorBroadcaster.__init__(self, step=step, offset=offset, first=first, last=last)


class LogBasedAnnotatorBroadcaster(LogBasedBroadcaster, AnnotatorBroadcaster):
    def __init__(self, log_pth, step=50, offset=0, first=True, last=True, formatter=FORMATTER.BROADCAST,
                 with_print=True, log_name='annotator'):
        LogBasedBroadcaster.__init__(self, log_pth, with_print=with_print, log_name=log_name, formatter=formatter)
        AnnotatorBroadcaster.__init__(self, step=step, offset=offset, first=first, last=last)


class SimpleAnnotator(Annotator):

    def act_return(self):
        labels_dct = super(SimpleAnnotator, self).act_return()
        labels_anno = []
        for meta in labels_dct.keys():
            labels_cmb = labels_dct[meta]
            labels_anno.append(labels_cmb[0][1])
        return labels_anno


class CombineAnnotator(Annotator):

    def act_return(self):
        labels_dct = super(CombineAnnotator, self).act_return()
        labels_cmb_full = []
        for meta in labels_dct.keys():
            labels_cmb = labels_dct[meta]
            labels_cmb_full += labels_cmb
        return labels_cmb_full


class LabelMixor(metaclass=ABCMeta):
    @abstractmethod
    def mix(self, labels, **kwargs):
        pass


class NMSBoxesMixor(LabelMixor):
    def __init__(self, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU,
                 num_presv=10000, by_cls=True):
        self.iou_thres = iou_thres
        self.nms_type = nms_type
        self.iou_type = iou_type
        self.num_presv = num_presv
        self.by_cls = by_cls

    def mix(self, labels, **kwargs):
        label_sum = copy.deepcopy(labels[0])
        for i in range(1, len(labels)):
            label_sum += labels[i]
        xyxysN = label_sum.export_xyxysN()
        confsN = label_sum.export_confsN()
        presv_inds = nms_xyxysN(
            xyxysN, confsN, cindsN=label_sum.export_cindsN() if self.by_cls else None,
            iou_thres=self.iou_thres, nms_type=self.nms_type, iou_type=self.iou_type,
            num_presv=self.num_presv)
        return label_sum[presv_inds]


if __name__ == '__main__':
    pass
