from torch.cuda.amp import autocast

from models.base import ResNetBkbn
from models.modules import *
from models.template import RadiusBasedCenterPrior, OneStageTorchModel, IndependentInferableModel
from utils.frame import *


class FCOSLayer(PointAnchorImgLayer):
    def __init__(self, stride, num_cls=20, img_size=(0, 0)):
        PointAnchorImgLayer.__init__(self, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def decode(reg_ltrd_conf, reg_chot, scale, xy_offset, stride, num_cls):
        xy_offset = xy_offset.to(reg_ltrd_conf.device, non_blocking=True)
        Hf, Wf, _ = xy_offset.size()
        reg_chot = reg_chot.permute(0, 2, 3, 1)
        reg_ltrd_conf = reg_ltrd_conf.permute(0, 2, 3, 1)
        center = torch.sigmoid(reg_ltrd_conf[..., 4:5])
        ltrd = (1 + reg_ltrd_conf[..., :4]) * torch.exp(scale.clamp(min=-5, max=5)) * stride
        xy_cen = (xy_offset + 0.5) * stride
        x1y1 = (xy_cen - ltrd[..., :2])
        x2y2 = (xy_cen + ltrd[..., 2:4])
        pred = torch.cat([x1y1, x2y2, center, reg_chot], dim=-1)
        pred = torch.reshape(pred, shape=(-1, Hf * Wf, num_cls + 5))
        return pred

    def forward(self, feat):
        ltrd_conf, chot = feat
        preds = FCOSLayer.decode(reg_ltrd_conf=ltrd_conf, reg_chot=chot, scale=self.scale, xy_offset=self.xy_offset,
                                 stride=self.stride, num_cls=self.num_cls)
        return preds


class FCOSSharePipline(nn.Module):
    def __init__(self, in_channels, repeat_num=4, num_cls=20, act=ACT.RELU,norm=NORM.BATCH):
        super(FCOSSharePipline, self).__init__()
        self.stem_chot = nn.Sequential(*[
            Ck3s1A(in_channels=in_channels, out_channels=in_channels, act=act,norm=norm)
            for _ in range(repeat_num)])

        self.stem_ltrd = nn.Sequential(*[
            Ck3s1A(in_channels=in_channels, out_channels=in_channels, act=act,norm=norm)
            for _ in range(repeat_num)])

        self.chot = Ck3s1(in_channels=in_channels, out_channels=num_cls)
        self.ltrd_conf = Ck3s1(in_channels=in_channels, out_channels=5)
        init_sig(self.ltrd_conf.conv.bias[4], prior_prob=0.001)

    def forward(self, featmaps):
        feats_procd = []
        for featmap in featmaps:
            stem_chot = self.stem_chot(featmap)
            stem_ltrd = self.stem_ltrd(featmap)
            chot = self.chot(stem_chot)
            ltrd_conf = self.ltrd_conf(stem_ltrd)
            feats_procd.append((ltrd_conf, chot))
        return feats_procd


class FCOSConstLayer(PointAnchorImgLayer):
    def __init__(self, stride, num_cls=20, img_size=(0, 0), batch_size=1):
        PointAnchorImgLayer.__init__(self, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.featmap = nn.Parameter(torch.zeros(batch_size, num_cls + 5, self.Hf, self.Wf))
        init_sig(self.featmap[:, 4], prior_prob=0.001)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        reg_chot = self.featmap[:, 5:, :, :]
        reg_ltrd_conf = self.featmap[:, :5, :, :]
        preds = FCOSLayer.decode(reg_ltrd_conf=reg_ltrd_conf, reg_chot=reg_chot, scale=self.scale,
                                 xy_offset=self.xy_offset, stride=self.stride, num_cls=self.num_cls)
        return preds


class FCOSConstMain(nn.Module):
    def __init__(self, num_cls=20, batch_size=1, img_size=(0, 0)):
        super(FCOSConstMain, self).__init__()
        self.img_size = img_size
        self.num_cls = num_cls
        self.layers = nn.ModuleList([
            FCOSConstLayer(stride=8, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            FCOSConstLayer(stride=16, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            FCOSConstLayer(stride=32, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            FCOSConstLayer(stride=64, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            FCOSConstLayer(stride=128, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
        ])

    def forward(self, imgs):
        preds = torch.cat([layer(None) for layer in self.layers], dim=1)
        return preds


class FCOSDownStreamAdd(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.RELU,norm=NORM.BATCH):
        super(FCOSDownStreamAdd, self).__init__()
        self.cvts = nn.ModuleList()
        self.adprs = nn.ModuleList()
        for i in range(len(in_channelss)):
            self.cvts.append(Ck1s1NA(in_channels=in_channelss[i], out_channels=out_channelss[i], act=act,norm=norm))
            if i < len(in_channelss) - 1:
                adpr = nn.Identity() if out_channelss[i + 1] == out_channelss[i] else \
                    Ck1s1(in_channels=out_channelss[i + 1], out_channels=out_channelss[i])
                self.adprs.append(adpr)

    def forward(self, feats):
        assert len(feats) == len(self.cvts), 'len err'
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if i == len(feats) - 1:
                feat_buff = self.cvts[i](feats[i])
            else:
                feat_buff = self.adprs[i](feat_buff)
                feat_buff = self.cvts[i](feats[i]) + F.upsample(feat_buff, scale_factor=2)
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class FCOSResNetMain(ResNetBkbn, ImageONNXExportable):

    def __init__(self, Module, repeat_nums, channels, num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        ResNetBkbn.__init__(self, Module=Module, repeat_nums=repeat_nums, channels=channels, act=act,norm=norm,
                            in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        in_channelss = (channels * 2, channels * 4, channels * 8)
        out_channelss = [256] * 3
        self.down = FCOSDownStreamAdd(in_channelss=in_channelss, out_channelss=out_channelss, act=act,norm=norm)
        self.stage5 = Ck3NA(in_channels=256, out_channels=256, stride=2, bn=False, act=act,norm=norm)
        self.stage6 = Ck3NA(in_channels=256, out_channels=256, stride=2, bn=False, act=act,norm=norm)
        self.shlayer = FCOSSharePipline(in_channels=256, repeat_num=5, num_cls=num_cls, act=act,norm=norm)
        self.layers = nn.ModuleList([
            FCOSLayer(stride=8, num_cls=num_cls, img_size=img_size),
            FCOSLayer(stride=16, num_cls=num_cls, img_size=img_size),
            FCOSLayer(stride=32, num_cls=num_cls, img_size=img_size),
            FCOSLayer(stride=64, num_cls=num_cls, img_size=img_size),
            FCOSLayer(stride=128, num_cls=num_cls, img_size=img_size),
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
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat2, feat3, feat4 = self.down((feat2, feat3, feat4))
        feat5 = self.stage5(feat4)
        feat6 = self.stage6(feat5)
        feats = (feat2, feat3, feat4, feat5, feat6)
        feats = self.shlayer(feats)
        preds = [layer(featmap) for layer, featmap in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def R18(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return FCOSResNetMain(**ResNetBkbn.PARA_R18, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                              in_channels=in_channels)

    @staticmethod
    def R34(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return FCOSResNetMain(**ResNetBkbn.PARA_R34, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                              in_channels=in_channels)

    @staticmethod
    def R50(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return FCOSResNetMain(**ResNetBkbn.PARA_R50, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                              in_channels=in_channels)

    @staticmethod
    def R101(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return FCOSResNetMain(**ResNetBkbn.PARA_R101, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                              in_channels=in_channels)

    @staticmethod
    def R152(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return FCOSResNetMain(**ResNetBkbn.PARA_R152, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                              in_channels=in_channels)


class FCOS(OneStageTorchModel, RadiusBasedCenterPrior, IndependentInferableModel):

    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        OneStageTorchModel.__init__(self, backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.radius = 1.1

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
    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        preds = self.pkd_modules['backbone'](imgsT.to(self.device))
        max_val, cindssT = torch.max(torch.sigmoid(preds[..., 5:]), dim=-1)
        xyxyssT = preds[..., :4]
        xyxyssT[..., 0:4:2] = torch.clamp(xyxyssT[..., 0:4:2], min=0, max=self.img_size[0])
        xyxyssT[..., 1:4:2] = torch.clamp(xyxyssT[..., 1:4:2], min=0, max=self.img_size[1])
        confssT = preds[..., 4] * max_val
        labels = []
        for xyxysT, confsT, cindsT in zip(xyxyssT, confssT, cindssT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
            prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=cindsT if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
            boxs = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name,
                img_size=self.img_size, num_cls=self.num_cls)
            labels.append(boxs)
        return labels

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        xyxys_tg = [np.zeros(shape=(0, 4))]
        cinds_tg = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            if len(label) == 0:
                continue
            label.orderby_measure(ascend=True)
            cinds_lb = label.export_cindsN()
            xyxys_lb = label.export_xyxysN()
            xywhs_lb = label.export_xywhsN()
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                stride, Wf, Hf = layer.stride, layer.Wf, layer.Hf
                ixys = (xywhs_lb[:, None, :2] / stride + self.offset_lb).astype(np.int32)
                fltr_valid = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
                idsids_ancr = ixys[..., 1] * Wf + ixys[..., 0]
                xys = ((ixys + 0.5) * stride)
                fltr_in = np.all((xys > xyxys_lb[:, None, :2]) * (xys < xyxys_lb[:, None, 2:4]), axis=2)
                ids_lb, ids_posi = np.nonzero(fltr_valid * fltr_in)
                ids_ancr = idsids_ancr[ids_lb, ids_posi]
                # 去除重复
                ids_ancr, repeat_filter = np.unique(ids_ancr, return_index=True)
                ids_lb = ids_lb[repeat_filter]

                inds_lb.append(ids_lb)
                inds_layer.append(np.full(fill_value=j, shape=len(ids_ancr)))
                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_ancr)))
                inds_pos.append(offset_layer + ids_ancr)
                cinds_tg.append(cinds_lb[ids_lb])
                xyxys_tg.append(xyxys_lb[ids_lb])
                offset_layer = offset_layer + Wf * Hf

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_lb = np.concatenate(inds_lb, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        xyxys_tg = np.concatenate(xyxys_tg, axis=0)
        targets = (inds_b_pos, inds_pos, inds_lb, xyxys_tg, cinds_tg, inds_layer)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        preds = self.pkd_modules['backbone'](imgs)
        inds_b_pos, inds_pos, inds_lb, xyxys_tg, cinds_tg, inds_layer = targets
        inds_b_pos = torch.as_tensor(inds_b_pos, dtype=torch.long).to(preds.device, non_blocking=True)
        inds_pos = torch.as_tensor(inds_pos, dtype=torch.long).to(preds.device, non_blocking=True)
        xyxys_tg = torch.as_tensor(xyxys_tg, dtype=torch.float).to(preds.device, non_blocking=True)
        cinds_tg = torch.as_tensor(cinds_tg, dtype=torch.long).to(preds.device, non_blocking=True)

        confs_pd = preds[:, :, 4]
        confs_tg = torch.full_like(confs_pd, fill_value=0, device=preds.device)
        weight_conf = torch.full_like(confs_pd, fill_value=1, device=preds.device)
        # 匹配最优尺度
        if inds_pos.size(0) > 0:
            xyxys_pd = preds[inds_b_pos, inds_pos, :4]
            chots_pd = preds[inds_b_pos, inds_pos, 5:]
            ious = ropr_arr_xyxysT(xyxys_pd, xyxys_tg, opr_type=OPR_TYPE.IOU)
            confs_tg[inds_b_pos, inds_pos] = ious.detach()
            iou_loss = 1 - torch.mean(ious)
            with autocast(enabled=False):
                chots_tg = F.one_hot(cinds_tg, self.num_cls).float()
                cls_loss = F.binary_cross_entropy_with_logits(chots_pd, chots_tg, weight=None, reduction='mean')
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)

        with autocast(enabled=False):
            conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, weight=weight_conf, reduction='mean')
        return OrderedDict(conf=conf_loss * 5, iou=iou_loss * 0.1, cls=cls_loss * 0.1)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = FCOSConstMain(num_cls=num_cls, img_size=img_size, batch_size=batch_size)
        return FCOS(backbone=backbone, device=device, pack=PACK.NONE)

    @staticmethod
    def ResNetR18(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = FCOSResNetMain.R18(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return FCOS(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR34(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = FCOSResNetMain.R34(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return FCOS(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR50(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = FCOSResNetMain.R50(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return FCOS(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR101(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = FCOSResNetMain.R101(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return FCOS(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR152(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = FCOSResNetMain.R152(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return FCOS(backbone=backbone, device=device, pack=pack)


if __name__ == '__main__':
    model = FCOSResNetMain.R50()
    model.export_onnx('./buff')
