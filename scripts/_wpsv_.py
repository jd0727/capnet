import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from scripts.function import *
from models import *
from data import *
from tools import *

if __name__ == '__main__':
    # img_size = (1024, 576)
    img_size = (640, 640)

    num_repeat = 20
    num_epoch = 5
    num_workers = 8
    anno_folder = 'AnnotationsLean2'
    # ds = InsulatorDI(task_type=TASK_TYPE.DETECTION)
    # dataset = ds.dataset('trainval', anno_folder='AnnotationsLean')

    ds = InsulatorUpsv(task_type=TASK_TYPE.DETECTION)
    dataset = ds.dataset('trainval', anno_folder=anno_folder)

    train_loader = ds.loader(dataset, batch_size=16, pin_memory=False, shuffle=True, num_workers=num_workers,
                             aug_seq=AugV5Cap(img_size=img_size, thres=5),
                             #    aug_seq = AugNorm(img_size=img_size)
                             )
    anno_loader = ds.loader(dataset, batch_size=16, pin_memory=False, shuffle=True, num_workers=num_workers,
                            aug_seq=AugRigid(img_size=img_size, thres=5),
                            )
    test_loader = ds.loader(set_name='val', batch_size=8, pin_memory=False, shuffle=False, drop_last=False,
                            anno_folder='Annotations', num_workers=num_workers,
                            aug_seq=AugNorm(img_size=img_size, thres=1))

    save_pth = os.path.join(PROJECT_PTH, 'ckpt/insu_bubm2')
    # model = YoloV5.Medium(device=0, pack=PACK.AUTO, num_cls=train_loader.num_cls, img_size=img_size)
    # model = PolarV1.Medium(device=3, pack=PACK.AUTO, num_cls=train_loader.num_cls, img_size=img_size, num_div=36)
    model = CapNet.Medium(device=2, pack=PACK.AUTO, num_cls=train_loader.num_cls, img_size=img_size, num_div=16)
    # model.load(save_pth)

    kwargs_infer = dict(conf_thres=None)
    kwargs_anno = dict(conf_thres=None, only_main=True)

    evaler = Evaler.InstMAP(loader=test_loader, total_epoch=1, criterion=CRITERION.COCO_STD, **kwargs_infer)
    evaler.add_actor(LogBasedEvalerBroadcaster(log_pth=save_pth))
    # evaler.start(model)

    trainner = Trainer(loader=train_loader, accu_step=1, loss_gain=1,
                       opt_builder=SGDBuilder(lr=0.1, momentum=0.9, dampening=0, weight_decay=5e-4, ), )
    trainner.add_actor(LogBasedTrainerBroadcaster(log_pth=save_pth))
    trainner.add_actor(EpochBasedLRScheduler.Cos(lr_init=0.01, lr_end=1e-6, num_epoch=num_epoch * num_repeat))
    trainner.add_actor(EpochBasedSaver(save_pth=save_pth, step=1, cover=True, last=True, offset=0))
    trainner.add_actor(EpochBasedEvalActor(evaler, step=num_epoch, save_pth=save_pth))

    annotator = SimpleAnnotator(loader=anno_loader, total_epoch=1, with_recover=True, **kwargs_anno)
    annotator.add_actor(LogBasedAnnotatorBroadcaster(log_pth=save_pth))
    #
    # labels_anno = annotator.start(model)
    # labels_anno = eval_quality(labels_anno)
    # labels_anno = intersect_labels(labels_anno)
    # dataset.dump(labels_anno, anno_folder=anno_folder)
    # dataset.create_set('cap_m', fltr=has_cap_missing)

    for n in range(0, num_repeat):
        trainner.total_epoch = (n + 1) * num_epoch
        trainner.start(model)
        # model.save(save_pth + '_%d' % n)
        labels_anno = annotator.start(model)
        labels_anno = eval_quality(labels_anno)
        labels_anno = intersect_labels(labels_anno)
        dataset.dump(labels_anno, anno_folder=anno_folder)
        dataset.create_set('cap_m', fltr=has_cap_missing)
    # 退出程序
    sys.exit(0)
