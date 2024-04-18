import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from utils import *
from data import *
from futurext import imgN_xyxyN_cendet2


def xywh_split(xywh, aspect=2.5, ratio=1 / 14):
    xyxy = xywhN2xyxyN(xywh)
    area_cap = ratio * np.prod(xywh[2:4])

    if xywh[2] > xywh[3]:
        cx, cy = np.sqrt(area_cap / aspect), np.sqrt(area_cap * aspect)
    else:
        cx, cy = np.sqrt(area_cap * aspect), np.sqrt(area_cap / aspect)
    # nx, ny = max(int(xywh[2] / cx), 1), max(int(xywh[3] / cy), 1)
    nx, ny = max(np.round(xywh[2] / cx), 1), max(np.round(xywh[3] / cy), 1)

    stride = xywh[2:4] / np.array([nx, ny])
    xywhs_cap = []
    for m in range(int(nx)):
        for n in range(int(ny)):
            p1 = xyxy[0:2] + np.array([m, n]) * stride
            p2 = xyxy[0:2] + np.array([m + 1, n + 1]) * stride
            xyxy_cap = np.concatenate([p1, p2], axis=0)
            xywhs_cap.append(xyxyN2xywhN(xyxy_cap))
    xywhs_cap = np.stack(xywhs_cap, axis=0)
    return xywhs_cap


# def xywh_split(xywh, aspect=2.5, ratio=0.05):
#     xyxy = xywhN2xyxyN(xywh)
#     area_cap = ratio * np.prod(xywh[2:4])
#
#     if xywh[2] > xywh[3]:
#         cx, cy = np.sqrt(area_cap / aspect), np.sqrt(area_cap * aspect)
#     else:
#         cx, cy = np.sqrt(area_cap * aspect), np.sqrt(area_cap / aspect)
#     # nx, ny = max(int(xywh[2] / cx), 1), max(int(xywh[3] / cy), 1)
#     nx, ny = max(np.round(xywh[2] / cx), 1), max(np.round(xywh[3] / cy), 1)
#
#     stride = xywh[2:4] / np.array([nx, ny])
#     xywhs_cap = []
#     for m in range(int(nx)):
#         for n in range(int(ny)):
#             p1 = xyxy[0:2] + np.array([m, n]) * stride
#             p2 = xyxy[0:2] + np.array([m + 1, n + 1]) * stride
#             xyxy_cap = np.concatenate([p1, p2], axis=0) \
#                        + np.random.uniform(size=4, low=-0.5, high=0.5) * stride[[0, 1, 0, 1]]
#             xywhs_cap.append(xyxyN2xywhN(xyxy_cap))
#     xywhs_cap = np.stack(xywhs_cap, axis=0)
#     return xywhs_cap


def xywha_split(xywha, **kwargs):
    xywh_proj = np.concatenate([[0, 0], xywha[2:4]], axis=0)
    xywhs_proj = xywh_split(xywh_proj, **kwargs)
    mat = asN2matsN(xywha[4])
    xys = xywha[:2] + xywhs_proj[:, 0:2] @ mat
    xywhas = np.concatenate([xys, xywhs_proj[:, 2:4], np.full(fill_value=xywha[4], shape=(xys.shape[0], 1))], axis=1)
    return xywhas


def xlyl_split(xlyl, **kwargs):
    xywha = xlylN2xywhaN(xlyl)
    xywhas = xywha_split(xywha, **kwargs)
    xlyls = xywhasN2xlylsN(xywhas)
    return xlyls


def cvtor(img, label, num_div=36, num_infer=2, flexible_cen=True):
    img = np.array(img)
    xyxys = label.export_xyxysN()
    items_new = []
    for i, item in enumerate(label):
        item.border = XLYLBorder.convert(item.border)
        item['cluster'] = np.random.randint(low=0, high=10000)
        label[i] = BoxRefItem.convert(item)
        xyxy = xyxys[i]
        xlyls_cap = imgN_xyxyN_cendet2(
            img, xyxy, num_div=num_div, num_infer=num_infer, flexible_cen=flexible_cen)
        for xlyl_cap in xlyls_cap:
            border_cap = XLYLBorder(xlyl_cap, size=label.img_size)
            item_new = BoxItem(border=border_cap, category=2, name='cap', cluster=i, conf=0.5)
            items_new.append(item_new)
    label += items_new
    return img, label


def cvtor3(label, **kwargs):
    label_new = label.empty()
    for i, item in enumerate(label):
        if item['name'] not in ['insulator_normal', 'insulator_blast']:
            continue
        xywh = XYWHBorder.convert(item.border).xywhN
        item.border = XYWHABorder.convert(XYXYBorder.convert(item.border))

        cluster = np.random.randint(low=0, high=1000000)
        cluster_gol = item['cluster']
        item['cluster_gol'] = cluster_gol
        item['cluster'] = cluster

        label_new.append(item)
        xywhs_cap = xywh_split(xywh, **kwargs)
        for k, xywh_cap in enumerate(xywhs_cap):
            border_cap = XLYLBorder(xywhN2xlylN(xywh_cap), size=label.img_size)
            item_new = BoxItem(border=border_cap, category=2, name='cap', cluster=cluster,
                               cluster_gol=cluster_gol)
            label_new.append(item_new)
    return label_new


def has_blast(label):
    for item in label:
        if item['name'] == 'insulator_blast':
            return True
    return False


def only_cap(label):
    label = label.filt(lambda item: item['name'] == 'cap')
    return label


def only_insu(label):
    label = label.filt(lambda item: item['name'] == 'insulator_normal' or item['name'] == 'insulator_blast')
    return label


if __name__ == '__main__':
    ds = InsulatorUpsv(task_type=TASK_TYPE.DETECTION)
    dataset = ds.dataset('trainval', anno_folder='Annotations')
    dataset.create_set('blast', fltr=has_blast)
    anno_folder = 'AnnotationsLean2'
    for set_name in ['train', 'val']:
        dataset = ds.dataset(set_name, anno_folder='Annotations')
        dataset.label_apply(cvtor3, anno_folder=anno_folder, anno_extend='xml')
    for set_name in ['test']:
        dataset = ds.dataset(set_name, anno_folder='Annotations')
        dataset.label_apply(only_insu, anno_folder=anno_folder, anno_extend='xml')

# if __name__ == '__main__':
#     ds = InsulatorUpsv(task_type=TASK_TYPE.DETECTION)
#     for set_name in ['train', 'val', 'test']:
#         dataset = ds.dataset(set_name, anno_folder='Annotations')
#         dataset.label_apply(only_insu, anno_folder='AnnotationsBase', anno_extend='xml')
