from futurext import xlylsT_filt_by_area2, featsT_pnt_pool, dlsT_scan_with_pool3

from models.polar.polar2 import *
from models.template import matcher_cenpiror, SurpervisedInferable


def similar_loss(xlyls, num_div=36):
    xlyls = xlyls - xlylsT2xywhsT(xlyls.detach())[..., None, :2]
    xs, ys = xlyls[..., 0], xlyls[..., 1]
    alphas = (torch.atan2(ys, xs) % (np.pi * 2)) / (np.pi * 2) * num_div
    dists = torch.linalg.norm(xlyls, dim=-1)

    inds = alphas.long() % num_div
    dists_mean = torch.zeros(size=(num_div,), device=xlyls.device)
    nums_sum = torch.zeros(size=(num_div,), device=xlyls.device, dtype=torch.long)
    filler = torch.ones_like(inds, device=xlyls.device, dtype=torch.long)
    nums_sum.scatter_add_(dim=0, index=inds.view(-1), src=filler.view(-1)).clamp_(min=1)
    dists_mean.scatter_add_(dim=0, index=inds.view(-1), src=dists.detach().view(-1))
    dists_mean = dists_mean / nums_sum

    dists_tar = dists_mean[inds]
    ious = ropr_arr_dlsT(dists, dists_tar, opr_type=OPR_TYPE.IOU)
    return ious


def xlylsT_xywhasT2ious(xlylsT, xywhasT):
    mat = asT2matsT(-xywhasT[..., 4])
    xlylsT_proj = (xlylsT - xywhasT[:, None, :2]) @ mat
    xlylsT_max = torch.max(xlylsT_proj, dim=-2)[0]
    xlylsT_min = torch.min(xlylsT_proj, dim=-2)[0]
    xyxysT_cvt = torch.cat([-xywhasT[:, 2:4] / 2, xywhasT[:, 2:4] / 2], dim=-1)
    xyxysT_proj = torch.cat([xlylsT_min, xlylsT_max], dim=-1)
    ious = ropr_arr_xyxysT(xyxysT_proj, xyxysT_cvt, opr_type=OPR_TYPE.IOU)
    return ious


def xlylsT_metric2(img, xlylsT, insu_area=1.0, num_samp=20, num_super=2):
    num_cap, num_div, _ = xlylsT.size()
    if num_cap == 0:
        return torch.as_tensor(0)
    censT, dlsT = xlylsT2censT_dlsT(xlylsT, num_div=num_div * num_super)
    dtdl = torch.linspace(start=0, end=1.0, steps=num_samp, device=dlsT.device)
    dls_grid = dlsT[..., None].detach() * dtdl
    colors_in = featsT_pnt_pool(featsT=img[None], ids_b=torch.zeros(size=(xlylsT.size(0),)), censT=censT,
                                dlsT_grid=dls_grid)
    # colors_aver = torch.mean(colors_in, dim=(0,), keepdim=True)
    colors_aver = torch.mean(colors_in, dim=(0, 2), keepdim=True)
    dists_in = torch.norm(colors_in - colors_aver, dim=-1)
    dens = 1 / torch.mean(dists_in, dim=(1, 2)).clamp(1e-5)

    dls_normd = dlsT / np.sqrt(insu_area)
    dls_aver = torch.mean(dls_normd, dim=0, keepdim=True)
    dens_dl = 1 / torch.norm(dls_normd - dls_aver, dim=1).clamp(1e-5)

    fits = dens * torch.sum(dls_normd ** 2, dim=1) * dens_dl
    # pow_num = max(0, num_cap - 2)
    pow_num = 1 if num_cap > 2 else 0
    return torch.mean(fits) * pow_num


def mix_item(items_new, items_old):
    items_upd = items_new.empty()
    if len(items_old) == 0 and len(items_new) == 0:
        return items_upd
    elif len(items_old) == 0:
        for item_new in items_new:
            item_new['repeat'] = 1
            items_upd.append(item_new)
        return items_upd
    elif len(items_new) == 0:
        for item_old in items_old:
            item_old['repeat'] = item_old.get('repeat', 1) - 1
            if item_old['repeat'] > 0:
                items_upd.append(item_old)
        return items_upd

    xyxys_old = items_old.export_xyxysN()
    xyxys_new = items_new.export_xyxysN()

    iou_mat = ropr_mat_xyxysN(xyxys_new, xyxys_old)
    for i in range(iou_mat.shape[0]):
        item_new = items_new[i]
        if len(iou_mat[i]) > 0 and np.max(iou_mat[i]) > 0.5:
            ind = np.argmax(iou_mat[i])
            iou_mat[:, ind] = -1
            item_new['repeat'] = items_old[ind].get('repeat', 0) + 1
        else:
            item_new['repeat'] = 1
        items_upd.append(item_new)

    # 补充上一次没有检测到的
    # for j in range(iou_mat.shape[1]):
    #     item_old = items_old[j]
    #     if np.all(iou_mat[:, j] == -1):
    #         continue
    #     else:
    #         item_old['repeat'] = item_old.get('repeat', 1) - 1
    #     if item_old['repeat'] > 0:
    #         items_upd.append(item_old)
    return items_upd


def sup_item(items_new, items_ref):
    items_upd = items_new.empty()
    xyxys_ref = items_ref.export_xyxysN()
    xyxys_new = items_new.export_xyxysN()
    iou_mat = ropr_mat_xyxysN(xyxys_new, xyxys_ref)
    maskr = np.all(iou_mat < 0.5, axis=1)
    for i in range(iou_mat.shape[0]):
        if maskr[i]:
            items_upd.append(items_new[i])
    return items_upd


class CapNet(OneStageTorchModel, SurpervisedInferable, RadiusBasedCenterPrior):

    def __init__(self, backbone, device=None, pack=PACK.AUTO, radius=2.1, alpha=1, beta=6, max_match=10, **kwargs):
        OneStageTorchModel.__init__(self, backbone=backbone, device=device, pack=pack, )
        self.area_thres = 0.01
        self.layers = backbone.layers
        self.radius = radius
        self.alpha = alpha
        self.beta = beta
        self.max_match = max_match

    def xyxy2inds(self, xyxy):
        inds = []
        offset_layer = 0
        for layer in self.layers:
            stride, Wf, Hf = layer.stride, layer.Wf, layer.Hf
            xyxy_ly = (xyxy / stride).astype(np.int32)
            xys = xyxyN2ixysN(xyxy_ly, size=(Wf, Hf))
            ids = xys[:, 1] * Wf + xys[:, 0]
            inds.append(offset_layer + ids)
            offset_layer = offset_layer + Wf * Hf
        inds = np.concatenate(inds, axis=0)
        return inds

    @property
    def num_div(self):
        return self.backbone.num_div

    @property
    def num_dstr(self):
        return self.backbone.num_dstr

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def num_layer(self):
        return len(self.layers)

    @property
    def img_size(self):
        return self.backbone.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size

    #
    @torch.no_grad()
    def imgs_labels2labels(self, imgs, labels, cind2name=None, only_main=False, force_cover=False, verbose=False,
                           conf_thres=None, from_insu=False, **kwargs):
        self.eval()
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        labels = labels_rescale(copy.deepcopy(labels), img_sizes=[self.img_size] * imgsT.size(0), ratios=ratios)
        preds = self.pkd_modules['backbone'](imgsT)
        censsT, stridessT, dlssT_dstr, chotssT = preds.split((2, 1, self.num_div * self.num_dstr, self.num_cls), dim=-1)
        dlssT_dstr = dlssT_dstr.reshape(preds.size(0), preds.size(1), self.num_div, self.num_dstr)
        dlssT = dlsT_dstr2dlsT(dlssT_dstr) * stridessT
        confssT, cindssT = torch.max(chotssT, dim=-1)
        if not force_cover:
            # imgsT = featsT_gauss(imgsT, kernel_size=5)
            imgs = F.avg_pool2d(imgs, stride=1, kernel_size=3, padding=1)
        labels_new = []
        for img, label, censT, dlsT, chotsT in zip(imgsT, labels, censsT, dlssT, chotssT):
            confsT, cindsT = torch.max(chotsT, dim=-1)
            insus_lb = copy.deepcopy(label.filt(lambda item: item.category.cindN < 2))
            caps_r_lb = label.filt(lambda item: item.category.cindN == 2)
            caps_m_lb = label.filt(lambda item: item.category.cindN == 4)
            insus_lb.orderby_measure(ascend=True)
            num_insu = len(insus_lb)
            for i in range(num_insu):
                item = insus_lb[i]
                if only_main and not item.get('main', False):
                    continue
                xyxy = XYXYBorder.convert(item.border).xyxyN
                insu_area = np.prod(xyxy[2:4] - xyxy[:2])
                inds = arrsN2arrsT(self.xyxy2inds(xyxy), self.device)
                censT_j, dlsT_j, chotsT_j, confsT_j, cindsT_j = \
                    censT[inds], dlsT[inds], chotsT[inds], confsT[inds], cindsT[inds]
                if conf_thres is None:
                    pmask = confsT_j > max(torch.mean(confsT_j) + torch.std(confsT_j), 1e-4)
                    if from_insu and item.category.cindN == 1:
                        ind_m = torch.argmax(chotsT_j[:, 4])
                        pmask[ind_m] = True
                        cindsT_j[ind_m] = 4
                    elif from_insu and item.category.cindN == 0:
                        cindsT_j[:] = 2
                else:
                    pmask = confsT_j > conf_thres
                censT_j, dlsT_j, confsT_j, cindsT_j = censT_j[pmask], dlsT_j[pmask], confsT_j[pmask], cindsT_j[pmask]
                confsT[inds] = 0

                # dlsT_j = dlsT_area_clip(dlsT_j, min_area=insu_area * self.area_thres, max_area=insu_area)
                pinds, dlsT_x = nms_dlsT(censT=censT_j, dlsT=dlsT_j, confsT=confsT_j, cindsT=cindsT_j, num_presv=10000)
                censT_j, dlsT_j, confsT_j, cindsT_j = censT_j[pinds], dlsT_j[pinds], confsT_j[pinds], cindsT_j[pinds]
                # print(censT_j.size(),dlsT_x.size())
                xlylsT_j = censT_dlsT2xlylsT(censT_j, dlsT_x)
                xlylsT_j = xysT_clip(xlylsT_j, xyxyN_rgn=xyxy)

                pinds = xlylsT_filt_by_area2(xlylsT_j, min_area=insu_area * self.area_thres, max_area=insu_area)
                xlylsT_j, confsT_j, cindsT_j = xlylsT_j[pinds], confsT_j[pinds], cindsT_j[pinds]
                xlylsT_j = xlylsT_regularization(xlylsT_j, num_div=self.num_div)

                # 生成标签
                caps = BoxesLabel.from_xlylsT_confsT_cindsT(
                    xlylsT=xlylsT_j, confsT=confsT_j, cindsT=cindsT_j, cind2name=cind2name,
                    img_size=self.img_size, num_cls=self.num_cls)
                for cap in caps:
                    cap['cluster'] = item['cluster']
                    cap['cluster_gol'] = item.get('cluster_gol', item['cluster'])
                # 添加缺失cap
                caps_m = caps.filt(lambda item: item.category.cindN == 4)
                caps_r = caps.filt(lambda item: item.category.cindN == 2)
                caps_fltd = mix_item(items_new=caps_m, items_old=caps_m_lb)
                insus_lb.extend(caps_fltd)
                # 添加一般cap
                fit = xlylsT_metric2(xlylsT=xlylsT_j[cindsT_j == 2], insu_area=insu_area, img=img).item()
                if force_cover or fit >= float(item.get('fit', 0)):
                    if verbose:
                        print(label.meta, float(item.get('fit', 0)), '->', fit)
                    item['fit'] = fit
                    item['updated'] = True
                    insus_lb.extend(caps_r)
                else:
                    item['updated'] = False
                    insus_lb.extend([cap for cap in caps_r_lb if cap['cluster'] == item['cluster']])
            labels_new.append(insus_lb)
        return labels_rescale(labels_new, img_sizes=imgs2img_sizes(imgs), ratios=1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_mtch = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        xlyls_tg = []
        cinds_tg = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            if len(label) == 0:
                continue
            label = label.filt(lambda item: item.category.cindN >= 2)
            cinds_lb = label.export_cindsN()
            xlyls_lb = label.export_xlylsN()
            xywhs_lb = xlylsN2xywhsN(xlyls_lb)

            ids_lb, ids_ancr, ids_mtch = matcher_cenpiror(
                xys_lb=xywhs_lb[:, :2], layers=self.layers, offset_lb=self.offset_lb)

            inds_mtch.append(ids_mtch)
            inds_b_pos.append(np.full(fill_value=i, shape=len(ids_ancr)))
            inds_lb.append(ids_lb)
            inds_pos.append(ids_ancr)
            cinds_tg.append(cinds_lb[ids_lb])
            xlyls_tg.append(xlyls_lb[ids_lb])

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_lb = np.concatenate(inds_lb, axis=0)
        inds_mtch = np.concatenate(inds_mtch, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0).astype(np.int32)
        xlyls_tg = xlylsNs_concat(xlyls_tg, num_pnt_default=self.num_div)
        targets = (inds_b_pos, inds_pos, inds_lb, xlyls_tg, cinds_tg, inds_mtch)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        preds = self.pkd_modules['backbone'](imgs)
        inds_b_pos, inds_pos, inds_lb, xlyls_tg, cinds_tg, inds_mtch = \
            arrsN2arrsT(targets, device=preds.device)

        chots_pd = preds[..., (3 + self.num_div * self.num_dstr):]
        chots_tg = torch.zeros_like(chots_pd, device=self.device)
        weight = torch.ones_like(chots_pd, device=self.device)
        if inds_pos.size(0) > 0:
            xys, strides, dls_dstr_pd, chots_pd_pos = \
                preds[inds_b_pos, inds_pos].split((2, 1, self.num_div * self.num_dstr, self.num_cls), dim=-1)
            dls_dstr_pd = dls_dstr_pd.reshape(dls_dstr_pd.size(0), self.num_div, self.num_dstr)
            dls_pd = dlsT_dstr2dlsT(dls_dstr_pd) * strides

            with torch.no_grad():
                dls_tg = censT_xlylsT2dlsT(xysT=xys, xlylsT=xlyls_tg, num_div=self.num_div)
                dls_dstr_tg = dls_tg / strides
                fltr_valid = torch.all((dls_tg > 0) * (dls_dstr_tg < self.num_dstr - 1), dim=-1)
                ious = ropr_arr_dlsT(dls_pd.detach(), dls_tg, opr_type=IOU_TYPE.IOU)
                confs = torch.gather(chots_pd_pos.detach(), index=cinds_tg[..., None], dim=-1)[..., 0]
                scores = (confs.detach() ** self.alpha) * (ious ** self.beta) * fltr_valid

                max_lb = torch.max(inds_lb).item() + 1
                buffer = torch.zeros(size=(imgs.size(0), max_lb, self.num_oflb * self.num_layer), device=self.device)
                buffer[inds_b_pos, inds_lb, inds_mtch] = scores.detach()
                buffer_aligend = torch.topk(buffer, dim=-1, k=self.max_match)[0][inds_b_pos, inds_lb]
                fltr_presv = (buffer_aligend[:, -1] <= scores) * (scores > 0)
                scores_normd = scores / buffer_aligend[:, 0].clamp(min=1e-5)

            inds_b_pos, inds_pos, cinds_tg, xys, dls_pd, dls_tg, dls_dstr_pd, dls_dstr_tg, scores_normd = \
                inds_b_pos[fltr_presv], inds_pos[fltr_presv], cinds_tg[fltr_presv], \
                xys[fltr_presv], dls_pd[fltr_presv], dls_tg[fltr_presv], \
                dls_dstr_pd[fltr_presv], dls_dstr_tg[fltr_presv], scores_normd[fltr_presv]

            chots_tg[inds_b_pos, inds_pos, cinds_tg] = scores_normd.detach()
            # weight[inds_b_pos, inds_pos] = 5  # 补充分类权重
            iou_loss = dlslog_loss(dls_pd, dls_tg, reduction='mean')
            # 关键点修正
            imgs = F.avg_pool2d(imgs, stride=1, kernel_size=3, padding=1)
            dls_tg_heat, heats = dlsT_scan_with_pool3(
                imgs, ids_b=inds_b_pos, censT=xys, dlsT=dls_pd, low=0.0, high=2.0, num_samp=20, num_super=3)
            weight_vert = (cinds_tg == 2)[..., None]

            vert_loss = dlslog_loss(dls_pd, dls_tg_heat, reduction='mean', weight=weight_vert)
            dfl_loss = distribute_loss(dls_dstr_pd.reshape(-1, self.num_dstr), dls_dstr_tg.view(-1), reduction='mean')
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            vert_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            dfl_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)

        with autocast(enabled=False):
            cls_loss = F.binary_cross_entropy(chots_pd, chots_tg, weight=weight, reduction='sum') \
                       / max(1, inds_pos.size(0))
        return OrderedDict(cls=cls_loss, iou=iou_loss * 5, vert=vert_loss * 3, dfl=dfl_loss)

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18, num_dstr=16):
        backbone = PolarV2Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div,
                                    num_dstr=num_dstr)
        return CapNet(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18, num_dstr=16):
        backbone = PolarV2Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div,
                                     num_dstr=num_dstr)
        return CapNet(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18, num_dstr=16):
        backbone = PolarV2Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div,
                                      num_dstr=num_dstr)
        return CapNet(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18, num_dstr=16):
        backbone = PolarV2Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div,
                                     num_dstr=num_dstr)
        return CapNet(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1, num_div=18, num_dstr=16):
        backbone = PolarV2ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size, num_div=num_div,
                                    num_dstr=num_dstr)
        return CapNet(backbone=backbone, device=device, pack=PACK.NONE)
