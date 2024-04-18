import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from data import *
from models import *
from tools import *

if __name__ == '__main__':
    # img_size = (1024, 576)
    img_size = (640, 640)
    num_workers = 5
    anno_folder = 'AnnotationsLean'
    # ds = InsulatorDI(task_type=TASK_TYPE.DETECTION)
    # dataset = ds.dataset('val', anno_folder=anno_folder)

    ds = InsulatorUpsv(task_type=TASK_TYPE.DETECTION)
    dataset = ds.dataset('train', anno_folder=anno_folder)

    train_loader = ds.loader(dataset, batch_size=4, pin_memory=False, shuffle=True, num_workers=num_workers,
                             # aug_seq=AugV5Cap(img_size=img_size, thres=5),
                             # aug_seq=ItemsFilt(fltr=lambda item: not (item.category.cindN == 4 and item.get('repeat', 1) <= 1)),
                             aug_seq=AugNorm(img_size=img_size, thres=1),
                             # aug_seq=RmoveCap(p=1,rmv_item=False,),
                             # aug_seq=AugTest(img_size=img_size, thres=1)
                             )
    anno_loader = ds.loader(dataset, batch_size=8, pin_memory=False, shuffle=False, num_workers=num_workers,
                            aug_seq=AugNorm(img_size=img_size, thres=1))
    test_loader = ds.loader(set_name='val', batch_size=4, pin_memory=False, shuffle=True, anno_folder='Annotations',
                            num_workers=num_workers, aug_seq=AugNorm(img_size=img_size, thres=1))

    save_pth = os.path.join(PROJECT_PTH, 'ckpt/insu_bubm')
    model = CapNet.Medium(device=0, pack=PACK.AUTO, num_cls=train_loader.num_cls, img_size=img_size, num_div=16)
    model.load(save_pth)

    kwargs_infer = dict()
    kwargs_anno = dict(only_main=True)

    evaler = Evaler.InstMAP(loader=test_loader, total_epoch=1, criterion=CRITERION.COCO_STD, **kwargs_infer)
    evaler.add_actor(PrintBasedEvalerBroadcaster())
    # evaler.start(model)

    annotator = SimpleAnnotator(loader=anno_loader, total_epoch=1, with_recover=True, **kwargs_anno)
    annotator.add_actor(PrintBasedAnnotatorBroadcaster())
    #
    # labels_anno = annotator.start(model)
    # save_pkl('./buff', labels_anno)
    # labels_anno = load_pkl('./buff')
    # labels_anno = eval_quality(labels_anno)
    # labels_anno = intersect_labels(labels_anno)
    # dataset.dump(labels_anno, anno_folder=anno_folder, with_recover=False)

    # for label in dataset.labels:
    #     for item in label:
    #         if not item.get('noise',True):
    #             print(item['fit'])

    # pieces = save_pieces(dataset)
    # save_pkl(os.path.join(PROJECT_PTH,'buff'), pieces)
    # # #
    # img, label = train_loader.dataset['001475_2920_0209_5472_3078']
    # imgs, labels = [img], [label]
    imgs, labels = next(iter(train_loader))
    # imgs, labels = zip(*train_loader.dataset[6:7])
    show_labels(imgs, labels, with_text=False)
    # labels_md = model.imgs2labels(imgs, cind2name=test_loader.cind2name, only_cinds=None, **kwargs_anno)
    labels_md = model.imgs_labels2labels(imgs, labels=labels, cind2name=test_loader.cind2name, only_main=True,
                                         verbose=True, force_cover=True, conf_thres=None, from_insu=True)
    show_labels(imgs, labels_md, with_text=False)
    # #
    plt.pause(1e5)
