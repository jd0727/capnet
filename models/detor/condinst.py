from torch.cuda.amp import autocast

from models.base import ResNetBkbn
from models.detor.fcos import FCOSSharePipline, FCOSDownStreamAdd
from models.modules import *
from models.modules import _calc_feat_size
from models.template import RadiusBasedCenterPrior, OneStageTorchModel, IndependentInferableModel
from utils import *


class CondInstSharePipline(FCOSSharePipline):
    def __init__(self, in_channels, ctr_channels, repeat_num=4, num_cls=20, act=ACT.RELU,norm=NORM.BATCH):
        FCOSSharePipline.__init__(self, in_channels, repeat_num=repeat_num, num_cls=num_cls, act=act,norm=norm, )
        self.controller = Ck1s1(in_channels=in_channels, out_channels=ctr_channels)

    def forward(self, featmaps):
        feats_procd = []
        for featmap in featmaps:
            feat_chot = self.stem_chot(featmap)
            feat_ltrd = self.stem_ltrd(featmap)
            reg_chot = self.chot(feat_chot)
            reg_ltrd_conf = self.ltrd_conf(feat_ltrd)
            fltr_wei = self.controller(feat_ltrd)
            feats_procd.append((reg_ltrd_conf, reg_chot, fltr_wei))
        return feats_procd


class CondInstLayer(PointAnchorImgLayer):
    def __init__(self, stride, num_cls=20, img_size=(0, 0)):
        PointAnchorImgLayer.__init__(self, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def decode(reg_ltrd_conf, reg_chot, ctr_wei, scale, xy_offset, stride):
        xy_offset = xy_offset.to(reg_ltrd_conf.device, non_blocking=True)
        Hf, Wf, _ = xy_offset.size()
        reg_chot = reg_chot.permute(0, 2, 3, 1)
        reg_ltrd_conf = reg_ltrd_conf.permute(0, 2, 3, 1)
        ctr_wei = ctr_wei.permute(0, 2, 3, 1)
        center = torch.sigmoid(reg_ltrd_conf[..., 4:5])
        ltrd = (1 + reg_ltrd_conf[..., :4]) * torch.exp(scale.clamp(min=-5, max=5)) * stride
        xy_cen = (xy_offset + 0.5) * stride
        x1y1 = (xy_cen - ltrd[..., :2])
        x2y2 = (xy_cen + ltrd[..., 2:4])
        pred = torch.cat([x1y1, x2y2, center, reg_chot, ctr_wei], dim=-1)
        pred = torch.reshape(pred, shape=(-1, Hf * Wf, pred.size(-1)))
        return pred

    def forward(self, feat):
        reg_ltrd_conf, reg_chot, fltr_wei = feat
        preds = CondInstLayer.decode(
            reg_ltrd_conf=reg_ltrd_conf, reg_chot=reg_chot, scale=self.scale,
            xy_offset=self.xy_offset, ctr_wei=fltr_wei, stride=self.stride)
        return preds


class CondInstConstLayer(PointAnchorImgLayer):
    def __init__(self, num_ctr, stride, num_cls=20, img_size=(256, 256), batch_size=3, ):
        PointAnchorImgLayer.__init__(self, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.num_ctr = num_ctr
        self.scale = nn.Parameter(torch.ones(1))
        self.feat = nn.Parameter(torch.zeros(batch_size, num_cls + 5 + num_ctr, self.Hf, self.Wf))
        init_sig(bias=self.feat[:, 4, :, :], prior_prob=0.001)
        nn.init.normal_(self.feat[:, num_cls + 5:, :, :], mean=0, std=0.1)

    def forward(self, feat):
        reg_ltrd_conf, reg_chot, ctr_wei = torch.split(self.feat, [5, self.num_cls, self.num_ctr], dim=1)
        preds = CondInstLayer.decode(
            reg_ltrd_conf=reg_ltrd_conf, reg_chot=reg_chot, scale=self.scale,
            xy_offset=self.xy_offset, ctr_wei=ctr_wei, stride=self.stride)
        return preds


class CondInstConstMain(nn.Module):
    CTR_STRUCT = (8, 1)

    def __init__(self, ctr_struct=CTR_STRUCT, num_cls=20, batch_size=1, img_size=(0, 0), masker_channels=8):
        super(CondInstConstMain, self).__init__()
        self.img_size = img_size
        self.num_cls = num_cls
        self.ctr_struct = ctr_struct
        num_ctr = CondInstResNetMain.calc_ctr(masker_channels, ctr_struct)
        self.num_ctr = num_ctr
        self.layers = nn.ModuleList([
            CondInstConstLayer(num_ctr=num_ctr, stride=8, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            CondInstConstLayer(num_ctr=num_ctr, stride=16, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            CondInstConstLayer(num_ctr=num_ctr, stride=32, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            CondInstConstLayer(num_ctr=num_ctr, stride=64, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
            CondInstConstLayer(num_ctr=num_ctr, stride=128, num_cls=num_cls, img_size=img_size, batch_size=batch_size),
        ])
        Wf, Hf = _calc_feat_size(img_size, 4)
        self.feats_msk = nn.Parameter(torch.rand(batch_size, masker_channels, Hf, Wf))

    def forward(self, imgs):
        preds = torch.cat([layer(None) for layer in self.layers], dim=1)
        return self.feats_msk, preds


class CondInstResNetMain(ResNetBkbn, ImageONNXExportable):
    CTR_STRUCT = (8, 8, 1)

    def __init__(self, Module, repeat_nums, channels, ctr_struct=CTR_STRUCT, num_cls=20, act=ACT.RELU,norm=NORM.BATCH,
                 img_size=(512, 512), in_channels=3, masker_channels=8):
        ResNetBkbn.__init__(self, Module=Module, repeat_nums=repeat_nums, channels=channels, act=act,norm=norm,
                            in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self._ctr_struct = ctr_struct
        self._num_ctr = CondInstResNetMain.calc_ctr(masker_channels, ctr_struct)
        in_channelss = (channels * 2, channels * 4, channels * 8)
        out_channelss = [256] * 3
        self.down = FCOSDownStreamAdd(in_channelss=in_channelss, out_channelss=out_channelss, act=act,norm=norm)
        self.stage5 = Ck3NA(in_channels=256, out_channels=256, stride=2, act=act,norm=norm)
        self.stage6 = Ck3NA(in_channels=256, out_channels=256, stride=2,  act=act,norm=norm)
        self.masker = CondInstResNetMain.MaskBranch(in_channels=256, out_channels=masker_channels, act=act,norm=norm)
        self.shlayer = CondInstSharePipline(
            in_channels=256, repeat_num=5, num_cls=num_cls, act=act,norm=norm,
            ctr_channels=self._num_ctr)
        self.layers = nn.ModuleList([
            CondInstLayer(stride=8, num_cls=num_cls, img_size=img_size),
            CondInstLayer(stride=16, num_cls=num_cls, img_size=img_size),
            CondInstLayer(stride=32, num_cls=num_cls, img_size=img_size),
            CondInstLayer(stride=64, num_cls=num_cls, img_size=img_size),
            CondInstLayer(stride=128, num_cls=num_cls, img_size=img_size),
        ])

    @property
    def ctr_struct(self):
        return self._ctr_struct

    @property
    def num_ctr(self):
        return self._num_ctr

    @staticmethod
    def calc_ctr(masker_channels=8, ctr_struct=CTR_STRUCT):
        num_ctr = 0
        last_channels = masker_channels
        for opt_channels in ctr_struct:
            num_ctr += (last_channels + 1) * opt_channels
            last_channels = opt_channels
        return num_ctr

    @staticmethod
    def MaskBranch(in_channels, out_channels, inner_channels=128, num_repeat=4, act=ACT.RELU,norm=NORM.BATCH):
        bkbn = nn.Sequential(*[
            Ck3s1NA(in_channels=in_channels if i == 0 else inner_channels,
                    out_channels=out_channels if i == num_repeat - 1 else inner_channels, act=act,norm=norm)
            for i in range(num_repeat)])
        return bkbn

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
        feats_msk = self.masker(feat2)
        feats = self.shlayer(feats)
        preds = [layer(featmap) for layer, featmap in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return feats_msk, preds

    @staticmethod
    def R18(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return CondInstResNetMain(**ResNetBkbn.PARA_R18, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                                  in_channels=in_channels)

    @staticmethod
    def R34(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return CondInstResNetMain(**ResNetBkbn.PARA_R34, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                                  in_channels=in_channels)

    @staticmethod
    def R50(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return CondInstResNetMain(**ResNetBkbn.PARA_R50, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                                  in_channels=in_channels)

    @staticmethod
    def R101(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return CondInstResNetMain(**ResNetBkbn.PARA_R101, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                                  in_channels=in_channels)

    @staticmethod
    def R152(num_cls=20, act=ACT.RELU,norm=NORM.BATCH, img_size=(512, 512), in_channels=3):
        return CondInstResNetMain(**ResNetBkbn.PARA_R152, num_cls=num_cls, act=act,norm=norm, img_size=img_size,
                                  in_channels=in_channels)


def apply_conv(feats, ctr_struct, ids_b, weights):
    N, C, H, W = feats.size()
    Ns, _ = weights.size()
    last_channels = C
    offset = 0
    buffer = feats[ids_b].view(Ns, C, H * W)
    for i, opt_channels in enumerate(ctr_struct):
        weights_i = weights[:, offset:offset + last_channels * opt_channels]
        offset += last_channels * opt_channels
        biass_i = weights[:, offset:offset + opt_channels]
        offset += opt_channels
        weights_i = weights_i.view(Ns, opt_channels, last_channels)
        buffer = torch.bmm(weights_i, buffer) + biass_i[..., None]
        buffer = F.leaky_relu(buffer, negative_slope=0.1)
        last_channels = opt_channels
    buffer = buffer.view(Ns, -1, H, W)
    return buffer


class CondInst(OneStageTorchModel, IndependentInferableModel, RadiusBasedCenterPrior):

    @property
    def img_size(self):
        return self.backbone.img_size

    @property
    def num_ctr(self):
        return self.backbone.num_ctr

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def ctr_struct(self):
        return self.backbone.ctr_struct

    def __init__(self, backbone, device=None, pack=PACK.AUTO, **kwargs):
        super(CondInst, self).__init__(backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.radius = 1.1

    # @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True, only_cinds=None,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None, with_mask=True,
                    masker_conf_thres=0.3, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        feats, preds = self.pkd_modules['backbone'](imgsT.to(self.device))
        xyxyssT, confssT, chotssT, weightss = preds.detach().split([4, 1, self.num_cls, self.num_ctr], dim=-1)
        max_val, cindssT = torch.max(torch.sigmoid(chotssT), dim=-1)
        xyxyssT[..., 0:4:2] = torch.clamp(xyxyssT[..., 0:4:2], min=0, max=self.img_size[0])
        xyxyssT[..., 1:4:2] = torch.clamp(xyxyssT[..., 1:4:2], min=0, max=self.img_size[1])
        confssT = confssT[..., 0] * max_val
        prsv_mskss = confssT > conf_thres
        if only_cinds is not None:
            prsv_mskss *= torch.any(cindssT[..., None] == torch.Tensor(only_cinds).to(cindssT.device), dim=-1)
        labels = []
        for i, (xyxysT, confsT, cindsT, prsv_msks, weights) in enumerate(
                zip(xyxyssT, confssT, cindssT, prsv_mskss, weightss)):
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT, weights = \
                xyxysT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks], weights[prsv_msks]
            prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=cindsT if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT, weights = \
                xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds], weights[prsv_inds]
            boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name,
                img_size=self.img_size, num_cls=self.num_cls)
            if not with_mask:
                labels.append(boxes)
                continue
            ids_b = torch.full(size=(weights.size(0),), fill_value=i, device=weights.device)
            masks_abs = apply_conv(feats, self.ctr_struct, ids_b=ids_b, weights=weights)
            masks_abs = F.interpolate(torch.sigmoid(masks_abs), size=(imgsT.size(2), imgsT.size(3)))
            masks_abs = masks_abs[:, 0].detach().cpu().numpy()
            rgns = [AbsValRegion(mask, conf_thres=masker_conf_thres) for mask in masks_abs]
            insts = InstsLabel.from_boxes_rgns(boxes, rgns)
            labels.append(insts)

        return labels

    def labels2tars(self, labels, **kwargs):
        masks_tg = [np.zeros(shape=(0, self.img_size[1], self.img_size[0]), dtype=np.int32)]
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        xyxys_tg = [np.zeros(shape=(0, 4))]
        cinds_tg = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            if len(label) == 0:
                continue
            label = label.permutation()
            # label.orderby_measure(ascend=True)
            masks_im = label.export_masksN_enc(self.img_size, num_cls=self.num_cls)[None, :]
            masks_tg.append(masks_im)
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
        masks_tg = np.concatenate(masks_tg, axis=0)
        targets = (inds_b_pos, inds_pos, inds_lb, xyxys_tg, cinds_tg, masks_tg, inds_layer)
        return targets

    @staticmethod
    def mask_dec(mask_enc, num_cls):
        masks_chot = torch.zeros(
            size=(mask_enc.size(0), num_cls + 1, mask_enc.size(1), mask_enc.size(2)), device=mask_enc.device)
        filler = torch.ones_like(mask_enc, device=mask_enc.device, dtype=torch.float32)
        masks_chot.scatter_(dim=1, index=mask_enc[:, None, :, :], src=filler[:, None, :, :])
        return masks_chot

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        imgs = imgs.to(self.device)
        feats, preds = self.pkd_modules['backbone'](imgs)
        inds_b_pos, inds_pos, inds_lb, xyxys_tg, cinds_tg, masks_tg, inds_layer = targets
        inds_b_pos = torch.as_tensor(inds_b_pos, dtype=torch.long).to(preds.device, non_blocking=True)
        inds_pos = torch.as_tensor(inds_pos, dtype=torch.long).to(preds.device, non_blocking=True)
        xyxys_tg = torch.as_tensor(xyxys_tg, dtype=torch.float).to(preds.device, non_blocking=True)
        cinds_tg = torch.as_tensor(cinds_tg, dtype=torch.long).to(preds.device, non_blocking=True)
        masks_tg = torch.as_tensor(masks_tg, dtype=torch.long).to(preds.device, non_blocking=True)

        masks_tg = CondInst.mask_dec(masks_tg, num_cls=self.num_cls)
        confs_pd = preds[:, :, 4]
        confs_tg = torch.full_like(confs_pd, fill_value=0, device=preds.device)
        weight_conf = torch.full_like(confs_pd, fill_value=1, device=preds.device)
        # 匹配最优尺度
        if inds_pos.size(0) > 0:
            xyxys_pd, _, chots_pd, ctr_pd = torch.split(
                preds[inds_b_pos, inds_pos], [4, 1, self.num_cls, self.num_ctr], dim=1)
            ious = ropr_arr_xyxysT(xyxys_pd, xyxys_tg, opr_type=OPR_TYPE.IOU)
            confs_tg[inds_b_pos, inds_pos] = ious.detach()
            iou_loss = 1 - torch.mean(ious)
            with autocast(enabled=False):
                chots_tg = F.one_hot(cinds_tg, self.num_cls).float()
                cls_loss = F.binary_cross_entropy_with_logits(chots_pd, chots_tg, weight=None, reduction='mean')

            # mask损失
            masks_pd = apply_conv(feats=feats, ctr_struct=self.ctr_struct, ids_b=inds_b_pos, weights=ctr_pd)
            masks_tg = F.interpolate(masks_tg, size=(masks_pd.size(2), masks_pd.size(3)))
            masks_tg_cls = masks_tg[inds_b_pos, cinds_tg]
            mask_loss = F.binary_cross_entropy_with_logits(masks_pd[:, 0], masks_tg_cls, reduction='mean')
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            mask_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)

        with autocast(enabled=False):
            conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, weight=weight_conf, reduction='mean')
        return OrderedDict(conf=conf_loss * 50, iou=iou_loss*0.01, cls=cls_loss, mask=mask_loss)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = CondInstConstMain(num_cls=num_cls, img_size=img_size, batch_size=batch_size)
        return CondInst(backbone=backbone, device=device, pack=PACK.NONE)

    @staticmethod
    def ResNetR18(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = CondInstResNetMain.R18(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return CondInst(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR34(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = CondInstResNetMain.R34(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return CondInst(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR50(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = CondInstResNetMain.R50(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return CondInst(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR101(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = CondInstResNetMain.R101(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return CondInst(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR152(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = CondInstResNetMain.R152(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, img_size=img_size)
        return CondInst(backbone=backbone, device=device, pack=pack)


if __name__ == '__main__':
    masker_channels = 3
    feats = torch.zeros(size=(2, masker_channels, 10, 20))
    ids_b = torch.Tensor([0, 0, 1, 1]).long()
    fltr_struct = CondInstResNetMain.CTR_STRUCT
    wid = CondInstResNetMain.calc_ctr(masker_channels=masker_channels, ctr_struct=fltr_struct)
    weights = torch.ones(size=(4, wid))
    a = apply_conv(feats, fltr_struct, ids_b, weights)

# if __name__ == '__main__':
#     model = CondInstResNetMain.R50()
#     model.export_onnx('./buff')
