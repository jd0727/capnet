from models.base.vit import LocalAttentionMutiHead2D
from models.polar.polar1 import *
from models.yolo.yolov8 import YoloV8Layer


class TransResidualV8(nn.Module):
    def __init__(self, channels, act=ACT.LK):
        super(TransResidualV8, self).__init__()
        self.attn = LocalAttentionMutiHead2D(in_channels=channels, out_channels=channels, act=act, kernel_size=7)
        self.conv2 = Ck3s1NA(in_channels=channels, out_channels=channels, act=act)

    def forward(self, x):
        x = x + self.conv2(self.attn(x))
        return x


class YoloV8TMain(DarkNetV8Bkbn, ImageONNXExportable):

    def __init__(self, Module, channelss, repeat_nums, num_cls=80, act=ACT.SILU, img_size=(256, 256),
                 in_channels=3, num_dstr=4):
        DarkNetV8Bkbn.__init__(self, Module, channelss, repeat_nums, act=act, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self._num_dstr = num_dstr
        feat_channelss = channelss[1:]
        down_channelss = channelss[1:]
        up_channelss = channelss[1:]
        self.down = YoloV8DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act)
        self.up = YoloV8UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act)
        self.layers = nn.ModuleList([
            YoloV8Layer(in_channels=channelss[1], inner_cahnnels=channelss[1], stride=8,
                        num_cls=num_cls, img_size=img_size, num_dstr=num_dstr),
            YoloV8Layer(in_channels=channelss[2], inner_cahnnels=channelss[1], stride=16,
                        num_cls=num_cls, img_size=img_size, num_dstr=num_dstr),
            YoloV8Layer(in_channels=channelss[3], inner_cahnnels=channelss[1], stride=32,
                        num_cls=num_cls, img_size=img_size, num_dstr=num_dstr)
        ])

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    @property
    def num_dstr(self):
        return self._num_dstr

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
        feats = (feats2, feats3, feats4)
        feats = self.down(feats)
        feats = self.up(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    PARA_NANO = dict(Module=TransResidualV8, channelss=(32, 64, 128, 256), repeat_nums=(1, 2, 2, 1))
    PARA_SMALL = dict(Module=TransResidualV8, channelss=(64, 128, 256, 512), repeat_nums=(1, 2, 2, 1))
    PARA_MEDIUM = dict(Module=TransResidualV8, channelss=(96, 192, 384, 576), repeat_nums=(2, 4, 4, 2))
    PARA_LARGE = dict(Module=TransResidualV8, channelss=(128, 256, 512, 512), repeat_nums=(3, 6, 6, 3))
    PARA_XLARGE = dict(Module=TransResidualV8, channelss=(160, 320, 640, 640), repeat_nums=(4, 8, 8, 4))

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8TMain(**YoloV8TMain.PARA_NANO, num_cls=num_cls, act=act, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8TMain(**YoloV8TMain.PARA_SMALL, num_cls=num_cls, act=act, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8TMain(**YoloV8TMain.PARA_MEDIUM, num_cls=num_cls, act=act, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8TMain(**YoloV8TMain.PARA_LARGE, num_cls=num_cls, act=act, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8TMain(**YoloV8TMain.PARA_XLARGE, num_cls=num_cls, act=act, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)


if __name__ == '__main__':
    model = YoloV8TMain.Medium()
    model.export_onnx('./buff')


class BubbleV2Main(PolarV1Main):

    def __init__(self, Module, channels, repeat_nums, recps, num_cls=80, act=ACT.SILU, img_size=(256, 256),
                 in_channels=3, num_div=36):
        super(BubbleV2Main, self).__init__(Module, channels, repeat_nums, recps, num_cls=num_cls, act=act,
                                           img_size=img_size, in_channels=in_channels, num_div=num_div)

        self.conf_early = Ck1s1(in_channels=channels * 4, out_channels=1)

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

        confs = self.conf_early(feats2)
        confs_x2 = F.avg_pool2d(confs, stride=2, kernel_size=2)
        confs_x4 = F.avg_pool2d(confs_x2, stride=2, kernel_size=2)
        confs_early = torch.cat([
            confs.view(confs.size(0), -1),
            confs_x2.view(confs_x2.size(0), -1),
            confs_x4.view(confs_x4.size(0), -1),
        ], dim=-1)
        confs_early = torch.sigmoid(confs_early)
        return preds, confs_early

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18):
        return BubbleV2Main(**DarkNetV5Bkbn.PARA_NANO, num_cls=num_cls, act=act, img_size=img_size,
                            in_channels=in_channels, num_div=num_div, recps=PolarV1Main.RECPS)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18):
        return BubbleV2Main(**DarkNetV5Bkbn.PARA_SMALL, num_cls=num_cls, act=act, img_size=img_size,
                            in_channels=in_channels, num_div=num_div, recps=PolarV1Main.RECPS)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18):
        return BubbleV2Main(**DarkNetV5Bkbn.PARA_MEDIUM, num_cls=num_cls, act=act, img_size=img_size,
                            in_channels=in_channels, num_div=num_div, recps=PolarV1Main.RECPS)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18):
        return BubbleV2Main(**DarkNetV5Bkbn.PARA_LARGE, num_cls=num_cls, act=act, img_size=img_size,
                            in_channels=in_channels, num_div=num_div, recps=PolarV1Main.RECPS)


#
# class BubbleV2ConstMain(PolarV1ConstMain):
#     def __init__(self, recps, num_cls=80, img_size=(256, 256), batch_size=3, num_div=36):
#         super(PolarV1ConstMain, self).__init__()
#         self.num_cls = num_cls
#         self.img_size = img_size
#         self.layers = nn.ModuleList([
#             PolarV1ConstLayer(batch_size=batch_size, stride=8, num_cls=num_cls, img_size=img_size, num_div=num_div,
#                             recp=recps[0]),
#             PolarV1ConstLayer(batch_size=batch_size, stride=16, num_cls=num_cls, img_size=img_size, num_div=num_div,
#                             recp=recps[1]),
#             PolarV1ConstLayer(batch_size=batch_size, stride=32, num_cls=num_cls, img_size=img_size, num_div=num_div,
#                             recp=recps[2]),
#         ])
#
#     @property
#     def num_div(self):
#         return self.layers[0].num_div
#
#     def forward(self, imgs):
#         preds = [layer(None) for layer in self.layers]
#         preds = torch.cat(preds, dim=1)
#         return preds


class BubbleV2(OneStageTorchModel, IndependentInferableModel, RadiusBasedCenterPrior):

    def __init__(self, backbone, device=None, pack=PACK.AUTO, **kwargs):
        OneStageTorchModel.__init__(self, backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.radius = 1.1

    @property
    def num_div(self):
        return self.backbone.num_div

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

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, cind2name=None, as_xyxy=False, iou_thres=0.45, by_cls=False,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, only_cinds=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        preds, confssT_early = self.pkd_modules['backbone'](imgsT)

        _, xlylssT, confssT, chotssT = preds.split(
            (self.num_div, self.num_div * 2, 1, self.num_cls), dim=-1)
        xlylssT = xlylssT.view(xlylssT.size(0), xlylssT.size(1), self.num_div, 2)
        xlylssT = xysT_clip_simple(xlylssT, img_size=self.img_size)
        max_val, cindssT = torch.max(torch.sigmoid(chotssT), dim=-1)
        confssT = confssT[..., 0] * max_val * confssT_early
        prsv_mskss = confssT > conf_thres
        if only_cinds is not None:
            prsv_mskss *= torch.any(cindssT[..., None] == torch.Tensor(only_cinds).to(cindssT.device), dim=-1)
        labels = []
        for xlylsT, confsT, cindsT, prsv_msks in zip(xlylssT, confssT, cindssT, prsv_mskss):
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xlylsT, confsT, cindsT = xlylsT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
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
        return labels

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        xlyls_tg = []
        cinds_tg = [np.zeros(shape=0, dtype=np.int32)]
        xys = [np.zeros(shape=(0, 2))]
        inds_clus = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            if len(label) == 0:
                continue
            label = label.filt(lambda item: item.category.cindN >= 2)
            label = label.permutation()
            iclus_lb = label.export_valsN(key='cluster', default=0)
            cinds_lb = label.export_cindsN()
            xlyls_lb = label.export_xlylsN()
            xywhs_lb = xlylsN2xywhsN(xlyls_lb)
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                stride, Wf, Hf, num_div, recp = layer.stride, layer.Wf, layer.Hf, layer.num_div, layer.recp
                fltr_scle = (np.min(xywhs_lb[:, 2:4], axis=1) > recp[0]) * (np.max(xywhs_lb[:, 2:4], axis=1) < recp[1])
                ixys = (xywhs_lb[:, None, :2] / stride + self.offset_lb).astype(np.int32)
                fltr_valid = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
                fltr_valid = fltr_valid * fltr_scle[..., None]

                ids_lb, _ = np.nonzero(fltr_valid)
                ixys = ixys[fltr_valid]
                ids_ancr = ixys[..., 1] * Wf + ixys[..., 0]

                xys_ly = ((ixys + 0.5) * stride)
                fltr_in = isin_arr_xlylsN(xys_ly, xlyls_lb[ids_lb])
                ids_lb, ids_ancr, xys_ly = ids_lb[fltr_in], ids_ancr[fltr_in], xys_ly[fltr_in]

                inds_layer.append(np.full(fill_value=j, shape=len(ids_ancr)))
                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_ancr)))
                inds_lb.append(ids_lb)
                inds_pos.append(offset_layer + ids_ancr)
                cinds_tg.append(cinds_lb[ids_lb])
                xlyls_tg.append(xlyls_lb[ids_lb])
                xys.append(xys_ly)
                inds_clus.append(iclus_lb[ids_lb])
                offset_layer = offset_layer + Wf * Hf

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_lb = np.concatenate(inds_lb, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        inds_clus = np.concatenate(inds_clus, axis=0)
        xys = np.concatenate(xys, axis=0)
        xlyls_tg = xlylsNs_concat(xlyls_tg, num_pnt_default=self.num_div)
        targets = (inds_b_pos, inds_pos, inds_lb, xys, xlyls_tg, cinds_tg, inds_clus, inds_layer)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        preds, confs_pd_early = self.pkd_modules['backbone'](imgs)
        inds_b_pos, inds_pos, inds_lb, xys, xlyls_tg, cinds_tg, inds_clus, inds_layer = \
            arrsN2arrsT(targets, device=self.device)
        confs_pd = preds[:, :, 3 * self.num_div]
        confs_tg = torch.full_like(confs_pd, fill_value=0, device=preds.device)
        confs_tg_early = torch.full_like(confs_pd_early, fill_value=0, device=preds.device)
        if inds_pos.size(0) > 0:
            dls_pd, xlyls_pd, _, chots_pd = preds[inds_b_pos, inds_pos].split(
                (self.num_div, 2 * self.num_div, 1, self.num_cls), dim=-1)
            xlyls_pd = xlyls_pd.reshape(-1, self.num_div, 2)
            dls_tg = censT_xlylsT2dlsT(xysT=xys, xlylsT=xlyls_tg, num_div=self.num_div)
            ious = ropr_arr_dlsT(dls_pd, dls_tg, opr_type=OPR_TYPE.IOU)
            iou_loss = 1 - torch.mean(ious)
            with autocast(enabled=False):
                chots_tg = F.one_hot(cinds_tg.long(), self.num_cls).float()
                cls_loss = F.binary_cross_entropy_with_logits(chots_pd, chots_tg, weight=None, reduction='mean')

            confs_tg[inds_b_pos, inds_pos] = torch.sqrt(ious.detach())
            confs_tg_early[inds_b_pos[cinds_tg == 2], inds_pos[cinds_tg == 2]] = 1
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            hit_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            sim_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            vert_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)

        with autocast(enabled=False):
            conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, reduction='mean')
            conf_loss_early = F.binary_cross_entropy(confs_pd_early, confs_tg_early, reduction='mean')
        return OrderedDict(conf=conf_loss * 50, iou=iou_loss, cls=cls_loss,
                           conf_early=conf_loss_early * 50,
                           # hit=hit_loss * 0.1,
                           # vert=vert_loss * 0.5,
                           # sim=sim_loss * 0.1,
                           )

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = BubbleV2Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return BubbleV2(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = BubbleV2Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return BubbleV2(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = BubbleV2Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return BubbleV2(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_div=18):
        backbone = BubbleV2Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size, num_div=num_div)
        return BubbleV2(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1, num_div=18):
        backbone = PolarV1ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size, num_div=num_div,
                                    recps=PolarV1Main.RECPS)
        return BubbleV2(backbone=backbone, device=device, pack=PACK.NONE)
