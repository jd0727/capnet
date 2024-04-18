from models.yolo.yolov5 import *
from models.yolo.yolov8 import DarkNetV8Bkbn, YoloV8Main, YoloV8DownStream, YoloV8UpStream
from utils import *


class PolarLayerProtyp(PointAnchorImgLayer):
    def __init__(self, stride, num_cls, img_size=(0, 0), num_div=36):
        PointAnchorImgLayer.__init__(self, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.num_div = num_div

    @property
    def num_div(self):
        return self._num_div

    @num_div.setter
    def num_div(self, num_div):
        self._num_div = num_div
        self.thetas = divide_circleT(num_div)
        self.dxy_offset = torch.stack([torch.cos(self.thetas), torch.sin(self.thetas)], dim=1)

    @property
    def anchors(self):
        anchors = self.cens_dls2xlyls(torch.full(size=(self.Wf * self.Hf, self.num_div), fill_value=self.stride / 2))
        return anchors

    def cens_dls2xlyls(self, dls):
        xy_offset = self.xy_offset.view(-1, 2).to(dls.device) + 0.5
        dxy_offset = self.dxy_offset.to(dls.device)
        xlyls = dls[..., None] * dxy_offset + xy_offset[..., None, :] * self.stride
        return xlyls


class PolarV1Layer(PolarLayerProtyp):
    def __init__(self, in_channels, stride, num_cls, img_size=(256, 256), num_div=36, act=ACT.RELU, norm=NORM.BATCH):
        super(PolarV1Layer, self).__init__(stride, num_cls, img_size=img_size, num_div=num_div)

        self.reg_dl = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=num_div, act=act, norm=norm),
            Ck3s1NA(in_channels=num_div, out_channels=num_div, act=act, norm=norm),
            Ck1s1(in_channels=num_div, out_channels=num_div),
        )

        self.reg_chot = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=num_cls, act=act, norm=norm),
            Ck3s1NA(in_channels=num_cls, out_channels=num_cls, act=act, norm=norm),
            Ck1s1(in_channels=num_cls, out_channels=num_cls),
        )
        init_sig(bias=self.reg_chot[-1].conv.bias, prior_prob=0.001)

    @staticmethod
    def decode(reg_dl, reg_chot, stride, xy_offset):
        reg_chot = reg_chot.permute(0, 2, 3, 1)
        reg_dl = reg_dl.permute(0, 2, 3, 1)
        Hf, Wf, _ = xy_offset.size()
        dls = torch.exp(reg_dl.clamp(min=-5, max=5)) * stride
        xy_offset = (xy_offset.to(reg_dl.device) + 0.5) * stride
        xy_offset = xy_offset.to(device=reg_dl.device)[None].repeat(reg_dl.size(0), 1, 1, 1)
        chot = torch.sigmoid(reg_chot)
        pred = torch.cat([xy_offset, dls, chot], dim=-1)
        pred = torch.reshape(pred, shape=(reg_dl.size(0), Hf * Wf, -1))
        return pred

    def forward(self, feat):
        reg_dl = self.reg_dl(feat)
        reg_chot = self.reg_chot(feat)
        preds = PolarV1Layer.decode(
            reg_dl=reg_dl, reg_chot=reg_chot, stride=self.stride, xy_offset=self.xy_offset)
        return preds


class PolarV1ConstLayer(PolarLayerProtyp):
    def __init__(self, batch_size, stride, num_cls, img_size=(0, 0), num_div=36):
        super(PolarV1ConstLayer, self).__init__(stride, num_cls, img_size=img_size, num_div=num_div)
        self.feat = nn.Parameter(torch.zeros(batch_size, num_cls + num_div, self.Hf, self.Wf))
        init_sig(bias=self.feat[:, num_div:, :, :], prior_prob=0.001)

    def forward(self, feat):
        reg_dl, reg_chot = self.feat.split((self.num_div, self.num_cls), dim=1)
        preds = PolarV1Layer.decode(
            reg_dl=reg_dl, reg_chot=reg_chot, stride=self.stride, xy_offset=self.xy_offset)
        return preds


class PolarV1ConstMain(nn.Module):
    def __init__(self, num_cls=80, img_size=(256, 256), batch_size=3, num_div=36):
        super(PolarV1ConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            PolarV1ConstLayer(batch_size=batch_size, stride=8, num_cls=num_cls, img_size=img_size,
                              num_div=num_div, ),
            PolarV1ConstLayer(batch_size=batch_size, stride=16, num_cls=num_cls, img_size=img_size,
                              num_div=num_div, ),
            PolarV1ConstLayer(batch_size=batch_size, stride=32, num_cls=num_cls, img_size=img_size,
                              num_div=num_div, ),
        ])

    @property
    def num_div(self):
        return self.layers[0].num_div

    def forward(self, imgs):
        preds = [layer(None) for layer in self.layers]
        preds = torch.cat(preds, dim=1)
        return preds


class PolarV1Main(DarkNetV8Bkbn, ImageONNXExportable):

    def __init__(self, Module, channelss, repeat_nums, num_cls=80, act=ACT.SILU, norm=NORM.BATCH,
                 img_size=(256, 256), in_channels=3, num_div=36):
        DarkNetV8Bkbn.__init__(self, Module, channelss, repeat_nums, act=act, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        feat_channelss = channelss[1:]
        down_channelss = channelss[1:]
        up_channelss = channelss[1:]
        self.spp = YoloV8Main.C1C3RepeatSPP(in_channels=channelss[-1], out_channels=channelss[-1],
                                            act=act, norm=norm)
        self.down = YoloV8DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1],
                                     act=act, norm=norm)
        self.up = YoloV8UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1],
                                 act=act, norm=norm)
        self.layers = nn.ModuleList([
            PolarV1Layer(in_channels=channelss[1], stride=8, num_cls=num_cls, img_size=img_size,
                         num_div=num_div, act=act, norm=norm),
            PolarV1Layer(in_channels=channelss[2], stride=16, num_cls=num_cls, img_size=img_size,
                         num_div=num_div, act=act, norm=norm),
            PolarV1Layer(in_channels=channelss[3], stride=32, num_cls=num_cls, img_size=img_size,
                         num_div=num_div, act=act, norm=norm),
        ])

    @property
    def num_div(self):
        return self.layers[0].num_div

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        self.layer.img_size = img_size

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
    def Nano(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_div=18):
        return PolarV1Main(**DarkNetV8Bkbn.PARA_NANO, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, norm=norm)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_div=18):
        return PolarV1Main(**DarkNetV8Bkbn.PARA_SMALL, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, norm=norm)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_div=18):
        return PolarV1Main(**DarkNetV8Bkbn.PARA_MEDIUM, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, norm=norm)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_div=18):
        return PolarV1Main(**DarkNetV8Bkbn.PARA_LARGE, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, norm=norm)


class PolarV1(OneStageTorchModel, IndependentInferableModel, RadiusBasedCenterPrior):

    def __init__(self, backbone, device=None, pack=PACK.AUTO, radius=3.1, alpha=1, beta=3, max_match=10, **kwargs):
        super(PolarV1, self).__init__(backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.radius = radius
        self.alpha = alpha
        self.beta = beta
        self.max_match = max_match

    @property
    def num_div(self):
        return self.backbone.num_div

    @property
    def num_layer(self):
        return len(self.layers)

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

    @property
    def num_anchor(self):
        return np.sum([layer.num_anchor for layer in self.backbone.layers])

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, cind2name=None, as_xyxy=False, iou_thres=0.45, by_cls=False,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        preds = self.pkd_modules['backbone'](imgsT)
        censsT, dlssT, chotssT = preds.split((2, self.num_div, self.num_cls), dim=-1)
        confssT, cindssT = torch.max(chotssT, dim=-1)
        labels = []
        for censT, dlsT, confsT, cindsT in zip(censsT, dlssT, confssT, cindssT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            censT, dlsT, confsT, cindsT = censT[prsv_msks], dlsT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
            xlylsT = censT_dlsT2xlylsT(censT, dlsT)
            xlylsT = xysT_clip(xlylsT, np.array(self.img_size))
            xyxysT = xlylsT2xyxysT(xlylsT)
            prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=cindsT if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xlylsT, xyxysT, confsT, cindsT = xlylsT[prsv_inds], xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
            if as_xyxy:
                boxs = BoxesLabel.from_xyxysT_confsT_cindsT(
                    xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name,
                    img_size=self.img_size, num_cls=self.num_cls)
            else:
                boxs = BoxesLabel.from_xlylsT_confsT_cindsT(
                    xlylsT=xlylsT, confsT=confsT, cindsT=cindsT, cind2name=cind2name,
                    img_size=self.img_size, num_cls=self.num_cls)
            labels.append(boxs)
        return labels_rescale(labels, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        xlyls_tg = []
        cinds_tg = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            if len(label) == 0:
                continue
            label = label.permutation()
            cinds_lb = label.export_cindsN()
            xlyls_lb = label.export_xlylsN()
            xywhs_lb = xlylsN2xywhsN(xlyls_lb)
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                stride, Wf, Hf, num_div = layer.stride, layer.Wf, layer.Hf, layer.num_div
                ixys = (xywhs_lb[:, None, :2] / stride + self.offset_lb).astype(np.int32)
                fltr_valid = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)

                ids_lb, _ = np.nonzero(fltr_valid)
                ixys = ixys[fltr_valid]
                ids_ancr = ixys[..., 1] * Wf + ixys[..., 0]

                inds_layer.append(np.full(fill_value=j, shape=len(ids_ancr)))
                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_ancr)))
                inds_lb.append(ids_lb)
                inds_pos.append(offset_layer + ids_ancr)
                cinds_tg.append(cinds_lb[ids_lb])
                xlyls_tg.append(xlyls_lb[ids_lb])
                offset_layer = offset_layer + Wf * Hf

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_lb = np.concatenate(inds_lb, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        xlyls_tg = xlylsNs_concat(xlyls_tg, num_pnt_default=self.num_div)
        targets = (inds_b_pos, inds_pos, inds_lb, xlyls_tg, cinds_tg, inds_layer)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        inds_b_pos, inds_pos, inds_lb, xlyls_tg, cinds_tg, inds_layer = arrsN2arrsT(targets, device=self.device)

        chots_pd = preds[..., (2 + self.num_div):]
        chots_tg = torch.zeros_like(chots_pd, device=self.device)
        # 匹配最优尺度
        if inds_pos.size(0) > 0:
            xys, dls_pd, chots_pd_pos = preds[inds_b_pos, inds_pos].split((2, self.num_div, self.num_cls), dim=-1)
            dls_tg = censT_xlylsT2dlsT(xysT=xys, xlylsT=xlyls_tg, num_div=self.num_div)
            fltr_valid = torch.all((dls_tg > 0), dim=-1)
            ious = ropr_arr_dlsT(dls_pd, dls_tg, opr_type=IOU_TYPE.IOU)
            confs = torch.gather(chots_pd_pos, index=cinds_tg[..., None], dim=-1)[..., 0]
            scores = (confs.detach() ** self.alpha) * (ious.detach() ** self.beta) * fltr_valid

            max_lb = torch.max(inds_lb).item() + 1
            buffer = torch.zeros(size=(imgs.size(0), max_lb, self.num_anchor), device=self.device)
            buffer[inds_b_pos, inds_lb, inds_pos] = scores.detach()
            score_thres = torch.topk(buffer, dim=-1, k=self.max_match)[0][inds_b_pos, inds_lb, -1]
            fltr_presv = (score_thres <= scores) * (scores > 0)

            inds_b_pos, inds_pos, cinds_tg, ious = \
                inds_b_pos[fltr_presv], inds_pos[fltr_presv], cinds_tg[fltr_presv], ious[fltr_presv]

            chots_tg[inds_b_pos, inds_pos, cinds_tg] = 1
            iou_loss = torch.mean((1 - ious))
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)

        with autocast(enabled=False):
            cls_loss = F.binary_cross_entropy(chots_pd, chots_tg, weight=None, reduction='sum') \
                       / max(1, inds_pos.size(0))
        return OrderedDict(cls=cls_loss * 0.5, iou=iou_loss * 7.5)

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = PolarV1Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return PolarV1(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = PolarV1Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return PolarV1(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = PolarV1Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return PolarV1(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = PolarV1Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return PolarV1(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1, num_div=18):
        backbone = PolarV1ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size, num_div=num_div,
                                    )
        return PolarV1(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = PolarV1.Medium(img_size=(512, 512), num_cls=20)
    model.export_onnx('../../buff.onnx')
