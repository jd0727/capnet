import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)

from data.processing import piece_merge, decode_meta_xyxy
from tools import prec_recl_per_class
from utils import *


def save_pieces(dataset):
    metas = dataset.create_set('cap_m', fltr=has_cap_missing)
    pieces = []
    for meta in metas:
        img, label = dataset[meta]
        buffer = {}
        for item in label:
            if item['name'] == 'cap_missing':
                cluster = item['cluster']
                if cluster not in buffer.keys():
                    xys = [XYWHBorder.convert(item.border).xywhN[:2] for item in label
                           if item['cluster'] == cluster and item.category.cindN >= 2]
                    alpha = xysN2aN(np.array(xys))
                    buffer[cluster] = alpha
                else:
                    alpha = buffer[cluster]
                border = XLYLBorder.convert(item.border)
                border.expend(ratio=2)
                xyxy = xlylN2xyxyN(border.xlylN).astype(np.int32)
                # xyxy = xyxyN_clip_simple(xyxy, size=img2size(img))
                piece = img2imgN(img)[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                pieces.append((piece, border.xlylN - xyxy[:2], alpha))
    return pieces


def intersect_labels(labels_ptch):
    # 预处理
    labels_fltd = []
    for j, label_ptch in enumerate(labels_ptch):
        main_clusters = [item['cluster'] for item in label_ptch if item.get('main', False)]
        labels_fltd.append(label_ptch.filt(lambda item: item['cluster'] in main_clusters and item.category.cindN >= 2))

    labels_gol = piece_merge(labels_fltd, iou_thres=0)
    # 筛选
    mapper = dict([(label_gol.meta, label_gol) for label_gol in labels_gol])
    # 重新裁剪
    print('Start reclip label')
    labels_upd = []
    for j, label_ptch in MEnumerate(labels_ptch):
        meta, xyxy = decode_meta_xyxy(label_ptch.meta)
        label_gol = BoxesLabel.convert(copy.deepcopy(mapper[meta]))
        rgn_size = xyxy[2:4] - xyxy[:2]
        img_size = np.array(label_ptch.img_size)
        scale = img_size / rgn_size
        label_gol.linear_(bias=-xyxy[:2] * scale, scale=scale, size=img_size).filt_measure_(thres=1)
        label_upd = label_ptch.filt(lambda item: item.category.cindN < 2)
        cluster_mapper = dict([(item['cluster_gol'], item['cluster']) for item in label_upd])
        # 聚类重新对齐
        for item in label_gol:
            if item['cluster_gol'] in cluster_mapper.keys():
                item['cluster'] = cluster_mapper[item['cluster_gol']]
                label_upd.append(item)
        labels_upd.append(label_upd)
    return labels_upd


def has_cap_missing(label):
    for item in label:
        if item['name'] == 'cap_missing':
            return True
    return False


def has_high(label):
    for item in label:
        if item.get('high', False):
            return True
    return False


def eval_quality(labels, broadcast=BROADCAST):
    fits = []
    upds = []
    cinds_gt = []
    cinds_pd = []
    for label in labels:
        for item in label:
            if item.get('main', False) and item.category.cindN < 2:
                fits.append(item.get('fit', 0))
                upds.append(item.get('updated', False))
                cinds_gt.append(item.category.cindN)
                ind_cluster = item['cluster']
                has_blast = any([item['cluster'] == ind_cluster and item.category.cindN == 4 for item in label])
                cinds_pd.append(1 if has_blast else 0)
    tgs, tps, precs, recls, f1s, accs = prec_recl_per_class(cinds_md=np.array(cinds_pd), cinds_ds=np.array(cinds_gt),
                                                            num_cls=2)
    broadcast('* Prec %.4f' % precs[1] + ' Recl %.4f' % recls[1] + ' F1 %.4f' % f1s[1] + ' Acc %.4f' % accs[1])
    fits = np.array(fits)
    fit_thres = np.exp(np.mean(np.log(fits + 1e-7)))
    ratio_upd = np.sum(np.array(upds)) / fits.shape[0]
    ratio_high = np.sum(fits > fit_thres) / fits.shape[0]
    ratio_low = np.sum(fits < fit_thres) / fits.shape[0]
    for label in labels:
        for item in label:
            if item.get('main', False) and item.category.cindN < 2:
                item['noise'] = item['fit'] < fit_thres
    broadcast(
        '* Update %.4f' % ratio_upd + ' Thres %.4f' % fit_thres + ' High %.4f' % ratio_high + ' Low %.4f' % ratio_low)
    return labels


def xlylsN_refine(xlylsN, num_div=32, num_iter=2, tolerate=0.0, only_squeeze=True):
    num_bdr, _, _ = xlylsN.shape
    xyxysN = xlylsN2xyxysN(xlylsN)
    censN = (xyxysN[:, :2] + xyxysN[:, 2:4]) / 2
    dirs = create_dirsN(num_div)
    dists_cen = np.linalg.norm(censN[..., None, :] - censN, axis=-1)

    for i in range(num_iter):
        dlsN = censN_xlylsN2dlsN(censN, xlylsN, num_div=num_div)
        xlylsN = censN[:, None, :] + dlsN[..., None] * dirs
        dts = xlylsN[..., None, :] - censN
        dists = np.linalg.norm(dts, axis=-1)
        thetas = np.arctan2(dts[..., 1], dts[..., 0])
        ainds = np.round(thetas / (np.pi * 2) * num_div).astype(np.int32) % num_div

        dists_limt = dlsN[np.broadcast_to(np.arange(num_bdr), ainds.shape), ainds]
        xlylsN_repj = dists_limt[..., None] * dirs[ainds] + censN
        fltr_dt = (dists_limt > dists)[..., None]
        if only_squeeze:
            fltr_dt *= (dists_limt < dists_cen[:, None, :])[..., None]
        xlylsN_dt = np.where(fltr_dt, xlylsN_repj - xlylsN[..., None, :], 0)
        if tolerate > 0:
            dt_norm = np.linalg.norm(xlylsN_dt, axis=-1)
            xlylsN_dt[dt_norm / dlsN[..., None] < tolerate] = 0
        xlylsN = np.sum(xlylsN_dt, axis=-2) / (np.sum(fltr_dt, axis=-2) + 1) + xlylsN

    return xlylsN


def filt_by_area(xlyls, min_area, thres=0):
    areas = xlylsN2areasN(xlyls) + 1e-7
    areas_log = np.log(areas)
    presv_mask = areas > min_area
    if thres > 0:
        area_log_aver = np.sum(areas_log * presv_mask) / max(np.sum(presv_mask), 1)
        presv_mask *= np.abs(areas_log - area_log_aver) < thres
    return np.nonzero(presv_mask)[0]


# torch.set_anomaly_enabled(True)

def fltr_labels(labels_cmb, iou_thres=0.5, with_align=True, ratio=0.25, conf_thres=0.1, with_rmovp=True):
    labels_fltd = []
    BROADCAST('Filt label')
    for i, (label_ds, label_anno) in MEnumerate(labels_cmb):
        # 拆分
        fltr_cap = lambda item: item.category.cindN >= 2
        lbds_cap, lbds_insu = label_ds.split(fltr_cap)
        lban_cap, lban_insu = label_anno.split(fltr_cap)
        num_cluster = len(lbds_insu)
        xyxys_ds_insu = lbds_insu.export_xyxysN(aname_bdr='border_ref')
        # 匹配绝缘子
        if len(lban_insu) > 0 and len(lbds_insu) > 0:
            xyxys_an_insu = lban_insu.export_xyxysN()
            iou_mat = ropr_mat_xyxysN(xyxys_ds_insu, xyxys_an_insu, opr_type=OPR_TYPE.IOU)
            for i, insu_ds in enumerate(lbds_insu):
                iou_max = np.max(iou_mat[i])
                if iou_max > iou_thres:
                    ind = np.argmax(iou_mat[i])
                    insu_an = lban_insu[ind]
                    border = insu_an.border
                    if with_align:
                        H = xlylN2homography(xyxyN2xlylN(xyxys_an_insu[ind]), xyxyN2xlylN(xyxys_ds_insu[i]))
                        border.xlylN = xlylN_perspective(border.xlylN, H)
                    insu_ds.border = border
                    insu_ds.category.cindN = insu_an.category.cindN
                    insu_ds['name'] = insu_an['name']
                    iou_mat[:, ind] = 0
        # 匹配绝缘片
        if len(lban_cap) > 0 and len(lbds_insu) > 0:
            xyxys_an_cap = lban_cap.export_xyxysN()
            iou_mat = ropr_mat_xyxysN(xyxys_an_cap, xyxys_ds_insu, opr_type=OPR_TYPE.IOU)
            max_ious, ks = np.max(iou_mat, axis=1), np.argmax(iou_mat, axis=1)
            for k in range(num_cluster):
                insu_ds = lbds_insu[k]
                ind_cluster = insu_ds['cluster']
                inds_k = np.nonzero((ks == k) * (max_ious > 0))[0]
                caps = []
                if len(inds_k) > 0:
                    xyxyN_rgn = XYXYBorder.convert(insu_ds.border_ref).xyxyN
                    caps = lban_cap[inds_k]
                    caps = rmv_label_overlap2(xyxyN_rgn, caps, num_div=16)

                if len(inds_k) == 0:
                    caps = [item for item in lbds_cap if item['cluster'] == k]

                for cap in caps:
                    cap['cluster'] = ind_cluster
                    lbds_insu.append(cap)
        else:
            lbds_insu += lbds_cap
        labels_fltd.append(lbds_insu)
    return labels_fltd


def rmv_label_overlap2(xyxyN_rgn, items, num_div=16):
    xlyls = items.export_xlylsN()
    # xlyls = xlylsN_clip_simple(xlyls, xyxyN_rgn=xyxyN_rgn)
    xlyls = xlylsN_refine(xlyls, num_div=num_div)
    presv_inds = filt_by_area(xlyls, min_area=xyxyN2areaN(xyxyN_rgn) * 0.01, thres=1)
    items, xlyls = items[presv_inds], xlyls[presv_inds]
    for item, xlyl in zip(items, xlyls):
        item.border.xlylN = xlyl
    # np.corrcoef(x)
    return items


def rmv_label_overlap(xlylN_rgn, items, num_div=16, order_by_conf=True):
    xyxyN_rgn = xlylN2xyxyN(xlylN_rgn)
    xyxyN_rgn = xyxyN_rgn.astype(np.int32)
    patch_size = xyxyN_rgn[2:4] - xyxyN_rgn[:2]
    # buffer = xlylN2maskNb(xlylN_rgn - xyxyN_rgn[:2], size=patch_size)
    buffer = np.full(shape=patch_size[::-1], fill_value=True)
    if order_by_conf:
        measures = [-item.category.conf for item in items]
    else:
        measures = [xlylN2areaN(item.border.xlylN) for item in items]
    order = np.argsort(measures)
    items = [items[ind] for ind in order]
    items_fltd = []
    for i, item in enumerate(items):
        xlyl = item.border.xlylN - xyxyN_rgn[:2]
        cen = xlylN2xywhN(xlyl)[:2].astype(np.int32)
        cx = np.clip(cen[0], a_min=0, a_max=patch_size[0] - 1)
        cy = np.clip(cen[1], a_min=0, a_max=patch_size[1] - 1)
        maskNb = xlylN2maskNb(xlyl, size=patch_size)
        maskNb_rmv = maskNb * buffer
        # if True:
        if maskNb_rmv[cy, cx]:
            buffer = buffer * (~maskNb)
            dl_new = maskNb_xyN2dlN(maskNb_rmv, cen, num_div=num_div, erode=0)
            xlyl_new = cenN_dlN2xlylN(cen, dl_new)
            item.border.xlylN = xlyl_new + xyxyN_rgn[:2]
            items_fltd.append(item)
    return items_fltd
