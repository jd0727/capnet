from torch.cuda.amp import autocast

from models.base.darknet import DarkNetV5Bkbn
from models.modules import *
from models.template import RadiusBasedCenterPrior
from models.yolo.modules import YoloFrame
from models.yolo.yolov5 import YoloV5Main, YoloV5DownStream, YoloV5UpStream
from utils import *


class YoloVRLayer(RotationalAnchorImgLayer):
    def __init__(self, in_channels, anchor_sizes, alphas, stride, num_cls, img_size=(0, 0)):
        RotationalAnchorImgLayer.__init__(
            self, stride=stride, alphas=alphas, anchor_sizes=anchor_sizes, img_size=img_size)
        self.num_cls = num_cls
        self.reg = Ck1(in_channels=in_channels, out_channels=self.Na * (num_cls + 6))
        init_sig(self.reg.conv.bias[5:self.Na * (num_cls + 6):(num_cls + 6)], prior_prob=0.001)

    @staticmethod
    def decode(featmap, a_offset, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(featmap.device, non_blocking=True)
        wh_offset = wh_offset.to(featmap.device, non_blocking=True)
        a_offset = a_offset.to(featmap.device, non_blocking=True)
        Hf, Wf, Na, _ = xy_offset.size()
        featmap = featmap.permute(0, 2, 3, 1)
        featmap = featmap.reshape(-1, Hf, Wf, Na, num_cls + 6).contiguous()

        reg_xy = torch.sigmoid(featmap[..., 0:2]) * 2 - 0.5
        reg_wh = (torch.sigmoid(featmap[..., 2:4]) * 2) ** 2 * (wh_offset / stride)
        xy = (reg_xy + xy_offset) * stride
        wh = reg_wh * stride
        a = featmap[..., 4:5] + a_offset

        chot = torch.sigmoid(featmap[..., 6:])
        conf = torch.sigmoid(featmap[..., 5:6])
        preds = torch.cat([xy, wh, a, conf, chot], dim=-1)
        preds = preds.reshape(-1, Wf * Hf * Na, num_cls + 6)
        return preds

    def forward(self, featmap):
        featmap = self.reg(featmap)
        pred = YoloVRLayer.decode(
            featmap=featmap, a_offset=self.a_offset, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloVRMain(DarkNetV5Bkbn, ImageONNXExportable):
    ALPHAS = (0, math.pi / 4, math.pi / 2, math.pi * 3 / 4)
    ANCHOR_SIZESS = (
        [(128, 32)] * 4,
        [(256, 64)] * 4,
        [(512, 128)] * 4
    )

    def __init__(self, Module, channels, repeat_nums, anchor_sizess, alphas, num_cls=80, act=ACT.SILU,
                 img_size=(0, 0), in_channels=3):
        super(YoloVRMain, self).__init__()
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        feat_channelss = (channels * 4, channels * 8, channels * 8)
        down_channelss = (channels * 4, channels * 8, channels * 8)
        up_channelss = (channels * 4, channels * 8, channels * 16)
        self.spp = YoloV5Main.C1C3RepeatSPP(in_channels=channels * 16, out_channels=channels * 8, act=act)
        self.down = YoloV5DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act)
        self.up = YoloV5UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act)
        self.layers = nn.ModuleList([
            YoloVRLayer(in_channels=channels * 4, anchor_sizes=anchor_sizess[0], stride=8,
                        num_cls=num_cls, img_size=img_size, alphas=alphas),
            YoloVRLayer(in_channels=channels * 8, anchor_sizes=anchor_sizess[1], stride=16,
                        num_cls=num_cls, img_size=img_size, alphas=alphas),
            YoloVRLayer(in_channels=channels * 16, anchor_sizes=anchor_sizess[2], stride=32,
                        num_cls=num_cls, img_size=img_size, alphas=alphas)
        ])

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        for layer in self.layers:
            layer.img_size = img_size

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats4 = self.spp(feats4)
        feats = (feats2, feats3, feats4)
        feats = self.down(feats)
        feats = self.up(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, img_size=(0, 0), anchor_sizess=ANCHOR_SIZESS, alphas=ALPHAS):
        return YoloVRMain(**YoloV5Main.NANO_PARA, num_cls=num_cls, act=act, img_size=img_size,
                          anchor_sizess=anchor_sizess, alphas=alphas)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, img_size=(0, 0), anchor_sizess=ANCHOR_SIZESS, alphas=ALPHAS):
        return YoloVRMain(**YoloV5Main.SMALL_PARA, num_cls=num_cls, act=act, img_size=img_size,
                          anchor_sizess=anchor_sizess, alphas=alphas)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, img_size=(0, 0), anchor_sizess=ANCHOR_SIZESS, alphas=ALPHAS):
        return YoloVRMain(**YoloV5Main.MEDIUM_PARA, num_cls=num_cls, act=act, img_size=img_size,
                          anchor_sizess=anchor_sizess, alphas=alphas)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, img_size=(0, 0), anchor_sizess=ANCHOR_SIZESS, alphas=ALPHAS):
        return YoloVRMain(**YoloV5Main.LARGE_PARA, num_cls=num_cls, act=act, img_size=img_size,
                          anchor_sizess=anchor_sizess, alphas=alphas)


class YoloVRConstLayer(RotationalAnchorImgLayer):
    def __init__(self, anchor_sizes, alphas, batch_size, stride, num_cls, img_size=(0, 0)):
        super().__init__(stride=stride, alphas=alphas, anchor_sizes=anchor_sizes, img_size=img_size)
        self.num_cls = num_cls
        self.featmaps = nn.Parameter(torch.zeros(batch_size, self.Na * (num_cls + 6), self.Hf, self.Wf))
        init_sig(bias=self.featmaps[:, 5:self.Na * (num_cls + 6):(num_cls + 6), :, :], prior_prob=0.1)

    def forward(self, featmaps):
        pred = YoloVRLayer.decode(
            featmap=self.featmaps, a_offset=self.a_offset, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloVRConstMain(nn.Module):
    def __init__(self, anchor_sizess, alphas, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloVRConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloVRConstLayer(anchor_sizes=anchor_sizess[0], alphas=alphas,
                             batch_size=batch_size, stride=8, num_cls=num_cls, img_size=img_size),
            YoloVRConstLayer(anchor_sizes=anchor_sizess[1], alphas=alphas,
                             batch_size=batch_size, stride=16, num_cls=num_cls, img_size=img_size),
            YoloVRConstLayer(anchor_sizes=anchor_sizess[2], alphas=alphas,
                             batch_size=batch_size, stride=32, num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloVR(YoloFrame, RadiusBasedCenterPrior):
    def __init__(self, backbone, device=None, pack=PACK.AUTO, **kwargs):
        super().__init__(backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.whr_thres = 4
        self.radius = 0.5
        self.alpha_thres = np.pi / 4

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True, num_presv=3000,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None, as_xyxy=True, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        preds = self.pkd_modules['backbone'](imgsT.to(self.device))
        xywhassT, confssT, chotssT = preds.split((5, 1, self.num_cls), dim=-1)
        max_vals, cindssT = torch.max(torch.sigmoid(chotssT), dim=2)
        confssT = confssT[..., 0] * max_vals
        labels = []
        for xywhasT, confsT, cindsT in zip(xywhassT, confssT, cindssT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xywhasT, confsT, cindsT = xywhasT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
            if not as_xyxy:
                prsv_inds = nms_xywhasT(xywhasT=xywhasT, confsT=confsT, cindsT=None,
                                        iou_thres=iou_thres, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.KLIOU)
                if len(prsv_inds) == 0:
                    labels.append(BoxesLabel(img_size=self.img_size))
                    continue
                xywhasT, confsT, cindsT = xywhasT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
                boxes = BoxesLabel.from_xywhasT_confsT_cindsT(
                    xywhasT=xywhasT, cindsT=cindsT, confsT=confsT, cind2name=cind2name, img_size=self.img_size,
                    num_cls=self.num_cls)
            else:
                xyxysT = xywhasT2xyxysT(xywhasT)
                prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=None,
                                       iou_thres=iou_thres, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU)
                if len(prsv_inds) == 0:
                    labels.append(BoxesLabel(img_size=self.img_size))
                    continue
                xyxysT, confsT, cindsT = xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
                boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                    xyxysT=xyxysT, cindsT=cindsT, confsT=confsT, cind2name=cind2name, img_size=self.img_size,
                    num_cls=self.num_cls)
            labels.append(boxes)
        return labels_rescale(labels, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        xywhas_tg = [np.zeros(shape=(0, 5))]
        chots_tg = [np.zeros(shape=(0, self.num_cls))]
        offset_lb = self.offset_lb

        for i, label in enumerate(labels):
            chots = label.export_chotsN(num_cls=self.num_cls)
            xywhas = label.export_xywhasN()
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                Na, stride, Wf, Hf, num_anchor = layer.Na, layer.stride, layer.Wf, layer.Hf, layer.num_anchor
                anchor_sizes = layer.anchor_sizes.numpy()
                alphas = layer.alphas.numpy()

                ixys = (xywhas[:, None, :2] / stride + offset_lb).astype(np.int32)
                fltr_in = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
                ids_ancr = (ixys[:, :, None, 1] * Wf + ixys[:, :, None, 0]) * Na + np.arange(Na)
                whr_val = xywhas[:, None, 2:4] / anchor_sizes[None, :, :]
                whr_val = np.max(np.maximum(whr_val, 1 / whr_val), axis=2)
                fltr_whr = whr_val < self.whr_thres
                fltr_whr = np.repeat(fltr_whr[:, None, :], axis=1, repeats=offset_lb.shape[0])

                dt_a = (xywhas[:, None, 4:5] % np.pi - alphas + np.pi / 2) % np.pi - np.pi / 2
                fltr_a = np.abs(dt_a) < self.alpha_thres
                fltr_valid = fltr_whr * fltr_a * fltr_in[:, :, None]

                ids_lb, ids_dt, ids_az = np.nonzero(fltr_valid)
                ids_ancr = ids_ancr[fltr_valid]

                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_lb)))
                inds_layer.append(np.full(fill_value=j, shape=len(ids_lb)))
                inds_pos.append(offset_layer + ids_ancr)
                chots_tg.append(chots[ids_lb])
                xywhas_tg.append(xywhas[ids_lb])

                offset_layer = offset_layer + num_anchor

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        xywhas_tg = np.concatenate(xywhas_tg, axis=0)
        chots_tg = np.concatenate(chots_tg, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        targets = (inds_b_pos, inds_pos, xywhas_tg, chots_tg, inds_layer)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        inds_b_pos, inds_pos, xywhas_tg, chots_tg, inds_layer = arrsN2arrsT(targets, device=self.device)
        confs_pd = preds[..., 5]
        confs_tg = torch.zeros_like(confs_pd, device=preds.device)
        weight_conf = torch.full_like(confs_pd, fill_value=1, device=preds.device)
        if inds_b_pos.size(0) > 0:
            xywhas_pd, _, chots_pd = preds[inds_b_pos, inds_pos].split((5, 1, self.num_cls), dim=1)
            ious = ropr_arr_xywhasT(xywhas_pd, xywhas_tg, opr_type=OPR_TYPE.KLIOU)
            confs_tg[inds_b_pos, inds_pos] = ious.detach()
            iou_loss = torch.mean((1 - ious))
            # 分类损失
            with autocast(enabled=False):
                cls_loss = F.binary_cross_entropy_with_logits(
                    chots_pd, chots_tg, weight=None, reduction='mean')
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        # 目标检出损失
        with autocast(enabled=False):
            conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, weight=weight_conf, reduction='mean')
        return OrderedDict(conf=conf_loss * 10, iou=iou_loss * 0.1, cls=cls_loss * 0.1)

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloVR(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloVR(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, **kwargs):
        backbone = YoloV5Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloVR(backbone=backbone, device=device, pack=pack, **kwargs)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, **kwargs):
        backbone = YoloV5Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloVR(backbone=backbone, device=device, pack=pack, **kwargs)

    @staticmethod
    def XLarge(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.XLarge(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloVR(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloVRConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size,
                                   anchor_sizess=YoloVRMain.ANCHOR_SIZESS, alphas=YoloVRMain.ALPHAS)
        return YoloVR(backbone=backbone, device=device, pack=PACK.NONE)
