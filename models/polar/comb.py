from torch.cuda.amp import autocast

from models.polar.polar1 import *
from models.template import RadiusBasedCenterPrior,OneBackwardTrainableModel
from models.yolo.yolov5 import YoloV5ConstMain


class Compound(OneBackwardTrainableModel, IndependentInferableModel, RadiusBasedCenterPrior):

    def __init__(self, backbone, assist, device=None, pack=PACK.AUTO, **kwargs):
        super(Compound, self).__init__(backbone=backbone, assist=assist, device=device, pack=pack)
        self.layers = backbone.layers
        self.alayers = assist.layers
        self.radius = 1.1
        self.whr_thresh = 4

    @property
    def num_div(self):
        return self.assist.num_div

    @property
    def strides(self):
        return np.array([layer.stride for layer in self.layers])

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def img_size(self):
        return self.backbone.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size

    @property
    def anchors(self):
        return torch.cat([layer.anchors for layer in self.backbone.layers], dim=0)

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, cind2name=None, with_cap=True, iou_thres=0.45, by_cls=False,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, only_cinds=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        preds_insu = self.pkd_modules['backbone'](imgsT.to(self.device))
        preds_cap = self.pkd_modules['assist'](imgsT.to(self.device))

        _, xlylssT_cap, confssT_cap, chotssT_cap = preds_cap.split(
            (self.num_div, self.num_div * 2, 1, self.num_cls), dim=-1)
        xywhssT_insu, confssT_insu, chotssT_insu = preds_insu.split((4, 1, self.num_cls), dim=-1)

        xyxyssT_insu = xyxysT_clip_simple(xywhsT2xyxysT(xywhssT_insu), img_size=self.img_size)
        xlylssT_cap = xlylssT_cap.view(xlylssT_cap.size(0), xlylssT_cap.size(1), self.num_div, 2)
        xlylssT_cap = xysT_clip_simple(xlylssT_cap, img_size=self.img_size)

        max_val, cindssT_cap = torch.max(torch.sigmoid(chotssT_cap), dim=-1)
        confssT_cap = confssT_cap[..., 0] * max_val
        max_val, cindssT_insu = torch.max(torch.sigmoid(chotssT_insu), dim=-1)
        confssT_insu = confssT_insu[..., 0] * max_val

        labels = []
        for xlylsT_cap, confsT_cap, cindsT_cap, xyxysT_isu, confsT_isu, cindsT_isu in \
                zip(xlylssT_cap, confssT_cap, cindssT_cap, xyxyssT_insu, confssT_insu, cindssT_insu):
            prsv_msks = confsT_isu > conf_thres
            xyxysT_isu, confsT_isu, cindsT_isu = xyxysT_isu[prsv_msks], confsT_isu[prsv_msks], cindsT_isu[prsv_msks]
            prsv_inds = nms_xyxysT(xyxysT=xyxysT_isu, confsT=confsT_isu, cindsT=cindsT_isu if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            xyxysT_isu, confsT_isu, cindsT_isu = xyxysT_isu[prsv_inds], confsT_isu[prsv_inds], cindsT_isu[prsv_inds]

            boxs = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT_isu, confsT=confsT_isu, cindsT=cindsT_isu, cind2name=cind2name,
                img_size=self.img_size, num_cls=self.num_cls)
            if not with_cap:
                labels.append(boxs)
                continue

            prsv_msks = confsT_cap > conf_thres
            xlylsT_cap, confsT_cap, cindsT_cap = xlylsT_cap[prsv_msks], confsT_cap[prsv_msks], cindsT_cap[prsv_msks]
            xyxysT_cap = xlylsT2xyxysT(xlylsT_cap)
            prsv_inds = nms_xyxysT(xyxysT=xyxysT_cap, confsT=confsT_cap, cindsT=cindsT_cap if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            xlylsT_cap, xyxysT_cap, confsT_cap, cindsT_cap = \
                xlylsT_cap[prsv_inds], xyxysT_cap[prsv_inds], confsT_cap[prsv_inds], cindsT_cap[prsv_inds]
            boxs += BoxesLabel.from_xlylsT_confsT_cindsT(
                xlylsT=xlylsT_cap, confsT=confsT_cap, cindsT=cindsT_cap, cind2name=cind2name,
                img_size=self.img_size, num_cls=self.num_cls)
            labels.append(boxs)
        return labels

    def labels2tars(self, labels, **kwargs):
        inds_insu = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_insu = [np.zeros(shape=0, dtype=np.int32)]
        xywhs_insu = [np.zeros(shape=(0, 4))]
        cinds_insu = [np.zeros(shape=0, dtype=np.int32)]

        inds_cap = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_cap = [np.zeros(shape=0, dtype=np.int32)]
        xlyls_cap = []
        xys_cap = [np.zeros(shape=(0, 2))]
        cinds_cap = [np.zeros(shape=0, dtype=np.int32)]

        for i, label in enumerate(labels):
            if len(label) == 0:
                continue
            label_insu, label_cap = label.split(lambda item: item.category.cindN < 2)
            xywhs = label_insu.export_xywhsN()
            cinds = label_insu.export_cindsN()
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                Na, stride, Wf, Hf, num_anchor = layer.Na, layer.stride, layer.Wf, layer.Hf, layer.num_anchor
                anchor_sizes = layer.anchor_sizes.numpy()

                ixys = (xywhs[:, None, :2] / stride + self.offset_lb).astype(np.int32)
                fltr_in = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
                ids_ancr = (ixys[:, :, None, 1] * Wf + ixys[:, :, None, 0]) * Na + np.arange(Na)
                whr_val = xywhs[:, None, 2:4] / anchor_sizes[None, :, :]
                whr_val = np.max(np.maximum(whr_val, 1 / whr_val), axis=2)
                fltr_whr = whr_val < self.whr_thresh
                fltr_whr = np.repeat(fltr_whr[:, None, :], axis=1, repeats=fltr_in.shape[1])
                fltr_valid = fltr_whr * fltr_in[:, :, None]

                ids_lb, ids_dt, ids_az = np.nonzero(fltr_valid)
                ids_ancr = ids_ancr[fltr_valid]

                inds_b_insu.append(np.full(fill_value=i, shape=len(ids_lb)))
                inds_insu.append(offset_layer + ids_ancr)
                cinds_insu.append(cinds[ids_lb])
                xywhs_insu.append(xywhs[ids_lb])
                offset_layer = offset_layer + num_anchor

            cinds = label_cap.export_cindsN()
            xlyls = label_cap.export_xlylsN()
            xywhs = xlylsN2xywhsN(xlyls)
            offset_layer = 0
            for j, layer in enumerate(self.alayers):
                stride, Wf, Hf, num_div, recp = layer.stride, layer.Wf, layer.Hf, layer.num_div, layer.recp
                fltr_scle = (np.min(xywhs[:, 2:4], axis=1) > recp[0]) * (np.max(xywhs[:, 2:4], axis=1) < recp[1])
                ixys = (xywhs[:, None, :2] / stride + self.offset_lb).astype(np.int32)
                fltr_valid = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
                fltr_valid = fltr_valid * fltr_scle[..., None]

                ids_lb, _ = np.nonzero(fltr_valid)
                ixys = ixys[fltr_valid]
                ids_ancr = ixys[..., 1] * Wf + ixys[..., 0]
                ids_ancr, repeat_filter = np.unique(ids_ancr, return_index=True)
                ids_lb, ixys = ids_lb[repeat_filter], ixys[repeat_filter]
                xys_ly = ((ixys + 0.5) * stride)
                fltr_in = isin_arr_xlylsN(xys_ly, xlyls[ids_lb])
                ids_lb, ids_ancr, xys_ly = ids_lb[fltr_in], ids_ancr[fltr_in], xys_ly[fltr_in]

                inds_b_cap.append(np.full(fill_value=i, shape=len(ids_ancr)))
                inds_cap.append(offset_layer + ids_ancr)
                cinds_cap.append(cinds[ids_lb])
                xlyls_cap.append(xlyls[ids_lb])
                xys_cap.append(xys_ly)
                offset_layer = offset_layer + Wf * Hf

        inds_b_insu = np.concatenate(inds_b_insu, axis=0)
        inds_insu = np.concatenate(inds_insu, axis=0)
        xywhs_insu = np.concatenate(xywhs_insu, axis=0)
        cinds_insu = np.concatenate(cinds_insu, axis=0)

        inds_b_cap = np.concatenate(inds_b_cap, axis=0)
        inds_cap = np.concatenate(inds_cap, axis=0)
        cinds_cap = np.concatenate(cinds_cap, axis=0)
        xys_cap = np.concatenate(xys_cap, axis=0)
        xlyls_cap = xlylsNs_concat(xlyls_cap, num_pnt_default=self.num_div)

        targets = (inds_b_insu, inds_insu, xywhs_insu, cinds_insu,
                   inds_b_cap, inds_cap, cinds_cap, xys_cap, xlyls_cap)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        inds_b_insu, inds_insu, xywhs_insu, cinds_insu, \
        inds_b_cap, inds_cap, cinds_cap, xys_cap, xlyls_cap = arrsN2arrsT(targets, device=self.device)
        loss_dct = OrderedDict()
        imgs = imgs.to(self.device)
        # 主网络
        preds = self.pkd_modules['backbone'](imgs)
        confs_pd = preds[..., 4]
        confs_tg = torch.zeros_like(confs_pd, device=preds.device)
        weight_conf = torch.full_like(confs_pd, fill_value=1, device=preds.device)
        if inds_b_insu.size(0) > 0:
            xywhs_pd, _, chots_pd = preds[inds_b_insu, inds_insu].split((4, 1, self.num_cls), dim=-1)
            ious = ropr_arr_xywhsT(xywhs_pd, xywhs_insu, opr_type=IOU_TYPE.IOU)
            confs_tg[inds_b_insu, inds_insu] = ious.detach()
            loss_dct['insu_iou'] = torch.mean((1 - ious))
            with autocast(enabled=False):
                chots_insu = F.one_hot(cinds_insu, self.num_cls).float()
                loss_dct['insu_cls'] = F.binary_cross_entropy_with_logits(
                    chots_pd, chots_insu, weight=None, reduction='mean')
        with autocast(enabled=False):
            loss_dct['insu_conf'] = F.binary_cross_entropy(
                confs_pd, confs_tg, weight=weight_conf, reduction='mean') * 50
        # 辅助
        preds = self.pkd_modules['assist'](imgs)
        confs_pd = preds[..., 3 * self.num_div]
        confs_tg = torch.full_like(confs_pd, fill_value=0, device=preds.device)
        weight_conf = torch.full_like(confs_pd, fill_value=1, device=preds.device)
        # 匹配最优尺度
        if inds_b_cap.size(0) > 0:
            dls_pd, xlyls_pd, _, chots_pd = preds[inds_b_cap, inds_cap].split(
                (self.num_div, 2 * self.num_div, 1, self.num_cls), dim=-1)
            dls_tg = censT_xlylsT2dlsT(xysT=xys_cap, xlylsT=xlyls_cap, num_div=self.num_div)
            ious = ropr_arr_dlsT(dls_pd, dls_tg, opr_type=IOU_TYPE.IOU)
            confs_tg[inds_b_cap, inds_cap] = ious.detach()
            loss_dct['cap_iou'] = 1 - torch.mean(ious)
            with autocast(enabled=False):
                chots_tg = F.one_hot(cinds_cap, self.num_cls).float()
                loss_dct['cap_cls'] = F.binary_cross_entropy_with_logits(
                    chots_pd, chots_tg, weight=None, reduction='mean')
        with autocast(enabled=False):
            loss_dct['cap_conf'] = F.binary_cross_entropy(
                confs_pd, confs_tg, weight=weight_conf, reduction='mean') * 50

        return loss_dct

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        assist = PolarV1Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        backbone = YoloV5Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size, )
        return Compound(backbone=backbone, assist=assist, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        assist = PolarV1Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        backbone = YoloV5Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size, )
        return Compound(backbone=backbone, assist=assist, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        assist = PolarV1Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        backbone = YoloV5Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size, )
        return Compound(backbone=backbone, assist=assist, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        assist = PolarV1Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        backbone = YoloV5Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size, )
        return Compound(backbone=backbone, assist=assist, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1, num_div=18):
        assist = PolarV1ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size, num_div=num_div,
                                  recps=PolarV1Main.RECPS)
        backbone = YoloV5ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size,
                                   anchor_sizess=YoloV5Main.ANCHOR_SIZESS)
        return Compound(backbone=backbone, assist=assist, device=device, pack=PACK.NONE)
