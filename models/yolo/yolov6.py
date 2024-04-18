from torch.onnx import TrainingMode
from models.modules import *
from models.base.modules import SPPF
from models.yolo.modules import *


class RCResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, act=ACT.LK, norm=NORM.BATCH):
        super().__init__()
        self.conv1 = RCk3NA(in_channels, out_channels, stride=stride, dilation=dilation, groups=groups, act=act,
                            norm=norm)
        self.conv2 = RCk3s1NA(out_channels, out_channels, dilation=dilation, groups=groups, act=act, norm=norm)
        self.has_shortcut = in_channels == out_channels and stride == 1

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x if self.has_shortcut else out


# ConvResidualRepeat+CBA+Res
class CSPBlockV6(nn.Module):
    def __init__(self, in_channels, out_channels, repeat_num, ratio=0.5, act=ACT.LK, norm=NORM.BATCH):
        super(CSPBlockV6, self).__init__()
        inner_channels = int(ratio * out_channels)
        self.shortcut = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, act=act, norm=norm)
        backbone = [Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, act=act, norm=norm)]
        for i in range(repeat_num):
            backbone.append(RCResidual(
                in_channels=inner_channels, out_channels=inner_channels, act=act, norm=norm))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1NA(in_channels=2 * inner_channels, out_channels=out_channels, act=act, norm=norm)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


class YoloV6Bkbn(nn.Module):
    def __init__(self, channels, repeat_nums, ratio=0.5, in_channels=3, act=ACT.LK, norm=NORM.BATCH):
        super().__init__()
        self.pre = RCk3NA(in_channels=in_channels, out_channels=channels, stride=2, act=act, norm=norm)
        self.stage1 = YoloV6Bkbn.RCResidualRepeat(
            in_channels=channels, out_channels=channels * 2, ratio=ratio,
            repeat_num=repeat_nums[0], stride=2, act=act, norm=norm)
        self.stage2 = YoloV6Bkbn.RCResidualRepeat(
            in_channels=channels * 2, out_channels=channels * 4, ratio=ratio,
            repeat_num=repeat_nums[1], stride=2, act=act, norm=norm)
        self.stage3 = YoloV6Bkbn.RCResidualRepeat(
            in_channels=channels * 4, out_channels=channels * 8, ratio=ratio,
            repeat_num=repeat_nums[2], stride=2, act=act, norm=norm)
        self.stage4 = YoloV6Bkbn.RCResidualRepeat(
            in_channels=channels * 8, out_channels=channels * 16, ratio=ratio,
            repeat_num=repeat_nums[3], stride=2, act=act, norm=norm)

    @staticmethod
    def RCResidualRepeat(in_channels, out_channels, stride=2, ratio=0.5, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        convs = nn.Sequential(
            RCk3NA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act, norm=norm),
            CSPBlockV6(in_channels=out_channels, out_channels=out_channels, ratio=ratio,
                       repeat_num=repeat_num, act=act, norm=norm)
        )
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        return feats4

    PARA_NANO = dict(channels=16, ratio=2 / 3, repeat_nums=(1, 2, 3, 1))
    PARA_SMALL = dict(channels=32, ratio=2 / 3, repeat_nums=(1, 2, 3, 1))
    PARA_MEDIUM = dict(channels=48, ratio=2 / 3, repeat_nums=(2, 3, 5, 2))
    PARA_LARGE = dict(channels=64, ratio=1 / 2, repeat_nums=(3, 6, 9, 3))
    PARA_XLARGE = dict(channels=80, ratio=1 / 2, repeat_nums=(4, 8, 12, 4))

    @staticmethod
    def MEDIUM(act=ACT.LK, norm=NORM.BATCH):
        return YoloV6Bkbn(**YoloV6Bkbn.PARA_MEDIUM, act=act, norm=norm)


class YoloV6DownStream(nn.Module):
    def __init__(self, in_channelss, out_channelss, ratio=0.5, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV6DownStream, self).__init__()
        self.mixrs = nn.ModuleList()
        self.adprs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            in_channels = in_channelss[i]
            if i == len(in_channelss) - 1:
                mixr = nn.Identity() if in_channels == out_channels else \
                    Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
                adpr = nn.Identity()
            else:
                out_channels_next = out_channelss[i + 1]
                last_channels = in_channels + out_channels // 2
                adpr = Ck1s1NA(in_channels=out_channels_next, out_channels=out_channels // 2)
                mixr = CSPBlockV6(
                    in_channels=last_channels, out_channels=out_channels, ratio=ratio, repeat_num=repeat_num, act=act,
                    norm=norm)
            self.mixrs.append(mixr)
            self.adprs.append(adpr)

    def forward(self, feats):
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if i == len(feats) - 1:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](
                    torch.cat([feats[i], F.upsample(self.adprs[i](feat_buff), scale_factor=2)], dim=1))
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class YoloV6UpStream(nn.Module):
    def __init__(self, in_channelss, out_channelss, ratio=0.5, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV6UpStream, self).__init__()
        self.adprs = nn.ModuleList()
        self.mixrs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            in_channels = in_channelss[i]
            if i == 0:
                adpr = nn.Identity()
                mixr = nn.Identity() if in_channels == out_channels else \
                    Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
            else:
                out_channels_pre = out_channelss[i - 1]
                adpr = Ck3NA(in_channels=out_channels_pre, out_channels=out_channels_pre, stride=2, act=act, norm=norm)
                mixr = CSPBlockV6(in_channels=out_channels_pre + in_channels, out_channels=out_channels,
                                  ratio=ratio, repeat_num=repeat_num, act=act, norm=norm)
            self.adprs.append(adpr)
            self.mixrs.append(mixr)

    def forward(self, feats):
        feat_buff = None
        feats_out = []
        for i in range(len(feats)):
            if i == 0:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](torch.cat([feats[i], self.adprs[i](feat_buff)], dim=1))
            feats_out.append(feat_buff)
        return feats_out


class YoloV6Layer(AnchorImgLayer):
    def __init__(self, in_channels, stride, num_cls, img_size=(0, 0), act=ACT.LK, norm=NORM.BATCH, anchor_ratio=5):
        anchor_sizes = [[stride * anchor_ratio, stride * anchor_ratio]]
        super().__init__(anchor_sizes=anchor_sizes, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.stem = Ck1s1NA(in_channels=in_channels, out_channels=in_channels, act=act, norm=norm)
        self.reg_cls = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=in_channels, act=act, norm=norm),
            Ck1(in_channels=in_channels, out_channels=num_cls)
        )
        self.reg_loc = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=in_channels, act=act, norm=norm),
            Ck1(in_channels=in_channels, out_channels=4)
        )
        init_sig(self.reg_cls[1].conv.bias, prior_prob=0.01)

    def forward(self, featmap):
        featmap = self.stem(featmap)
        reg_cls = self.reg_cls(featmap)
        reg_loc = self.reg_loc(featmap)
        pred = YoloV6Layer.decode(
            reg_loc=reg_loc, reg_cls=reg_cls, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred

    @staticmethod
    def decode(reg_loc, reg_cls, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(reg_loc.device, non_blocking=True)
        wh_offset = wh_offset.to(reg_loc.device, non_blocking=True)
        Hf, Wf, _ = list(xy_offset.size())

        reg_loc = reg_loc.permute(0, 2, 3, 1)
        reg_loc = reg_loc.reshape(-1, Hf, Wf, 4)

        reg_cls = reg_cls.permute(0, 2, 3, 1)
        reg_cls = reg_cls.reshape(-1, Hf, Wf, num_cls)

        x1y1 = (xy_offset + 0.5 - reg_loc[..., :2]) * stride - wh_offset / 2
        x2y2 = (xy_offset + 0.5 + reg_loc[..., 2:4]) * stride + wh_offset / 2

        reg_cls = torch.sigmoid(reg_cls)
        pred = torch.cat([x1y1, x2y2, reg_cls], dim=-1).contiguous()
        pred = pred.reshape(-1, Wf * Hf, num_cls + 4)
        return pred


class YoloV6ConstLayer(AnchorImgLayer):
    def __init__(self, batch_size, stride, num_cls, img_size=(0, 0), anchor_ratio=5):
        anchor_sizes = [[stride * anchor_ratio, stride * anchor_ratio]]
        super(YoloV6ConstLayer, self).__init__(anchor_sizes=anchor_sizes, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.reg_cls = nn.Parameter(torch.zeros(batch_size, num_cls, self.Hf, self.Wf))
        self.reg_loc = nn.Parameter(torch.zeros(batch_size, 4, self.Hf, self.Wf))
        init_sig(self.reg_cls, prior_prob=0.01)

    def forward(self, featmap):
        pred = YoloV6Layer.decode(
            reg_loc=self.reg_loc, reg_cls=self.reg_cls, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloV6ConstMain(nn.Module):
    def __init__(self, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloV6ConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloV6ConstLayer(batch_size=batch_size, stride=8, num_cls=num_cls, img_size=img_size),
            YoloV6ConstLayer(batch_size=batch_size, stride=16, num_cls=num_cls, img_size=img_size),
            YoloV6ConstLayer(batch_size=batch_size, stride=32, num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloV6Main(YoloV6Bkbn):

    def __init__(self, channels, repeat_nums, repeat_num_stm, ratio=0.5, in_channels=3, num_cls=80, act=ACT.MISH,
                 norm=NORM.BATCH,
                 img_size=(512, 512), anchor_ratio=5):
        super(YoloV6Main, self).__init__(channels, repeat_nums, ratio=ratio, in_channels=in_channels, act=act,
                                         norm=norm)
        self.num_cls = num_cls
        self._img_size = img_size
        self.spp = SPPF(in_channels=channels * 16, out_channels=channels * 16, kernel_size=5, act=act, norm=norm)
        feat_channelss = (channels * 4, channels * 8, channels * 16)
        pan_channelss = (channels * 2, channels * 4, channels * 8)
        self.down = YoloV6DownStream(in_channelss=feat_channelss, out_channelss=pan_channelss, ratio=ratio,
                                     repeat_num=repeat_num_stm, act=act, norm=norm)
        self.up = YoloV6UpStream(in_channelss=pan_channelss, out_channelss=pan_channelss, ratio=ratio,
                                 repeat_num=repeat_num_stm, act=act, norm=norm)

        self.layers = nn.ModuleList([
            YoloV6Layer(in_channels=pan_channelss[0], stride=8, num_cls=num_cls,
                        img_size=img_size, act=act, norm=norm, anchor_ratio=anchor_ratio),
            YoloV6Layer(in_channels=pan_channelss[1], stride=16, num_cls=num_cls,
                        img_size=img_size, act=act, norm=norm, anchor_ratio=anchor_ratio),
            YoloV6Layer(in_channels=pan_channelss[2], stride=32, num_cls=num_cls,
                        img_size=img_size, act=act, norm=norm, anchor_ratio=anchor_ratio)
        ])

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
        feats = (feats2, feats3, self.spp(feats4))
        feats = self.down(feats)
        feats = self.up(feats)

        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    PARA_NANO = dict(repeat_num_stm=2, **YoloV6Bkbn.PARA_NANO)
    PARA_SMALL = dict(repeat_num_stm=2, **YoloV6Bkbn.PARA_SMALL)
    PARA_MEDIUM = dict(repeat_num_stm=3, **YoloV6Bkbn.PARA_MEDIUM)
    PARA_LARGE = dict(repeat_num_stm=6, **YoloV6Bkbn.PARA_LARGE)
    PARA_XLARGE = dict(repeat_num_stm=8, **YoloV6Bkbn.PARA_XLARGE)

    @staticmethod
    def MEDIUM(num_cls=80, img_size=(512, 512), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return YoloV6Main(**YoloV6Main.PARA_MEDIUM, num_cls=num_cls, img_size=img_size, act=act, norm=norm,
                          in_channels=in_channels)

    @staticmethod
    def LARGE(num_cls=80, img_size=(512, 512), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return YoloV6Main(**YoloV6Main.PARA_LARGE, num_cls=num_cls, img_size=img_size, act=act, norm=norm,
                          in_channels=in_channels)


class YoloV6(YoloFrame):

    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(YoloV6, self).__init__(backbone=backbone, device=device, pack=pack)
        self.layers = self.backbone.layers
        self.alpha = 1.0
        self.beta = 6.0
        self.max_mtch_num = 13

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        confssT, cindssT = torch.max(preds[..., 4:], dim=2)
        xyxyssT = preds[..., :4]
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
            boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name, img_size=self.img_size,
                num_cls=self.num_cls)
            labels.append(boxes)
        return labels_rescale(labels, imgs, 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        xyxys_tg = []
        cinds_tg = []
        for i, boxes in enumerate(labels):
            xyxys, cinds = boxes.export_xyxysN_cindsN()
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                Na, stride, Wf, Hf, num_anchor = layer.Na, layer.stride, layer.Wf, layer.Hf, layer.num_anchor
                xy_offset = layer.xy_offset.numpy() + 0.5
                xyxys_st = xyxys[:, None, None, :] / stride

                filter_in = (xy_offset > xyxys_st[..., :2]) * (xy_offset < xyxys_st[..., 2:4])
                filter_in = np.all(filter_in, axis=3)
                ids_lb, iy, ix = np.nonzero(filter_in)
                ixy = iy * Wf + ix

                inds_lb.append(ids_lb)
                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_lb)))
                inds_layer.append(np.full(fill_value=j, shape=len(ids_lb)))
                inds_pos.append(offset_layer + ixy)
                cinds_tg.append(cinds)
                xyxys_tg.append(xyxys)

                offset_layer = offset_layer + num_anchor

        # 以矩阵压缩形式表示标签
        lb_nums = [len(cinds) for cinds in cinds_tg]
        lb_num_max = max(lb_nums)
        xyxys_tg_mat = np.zeros(shape=(len(labels), lb_num_max, 4))
        cinds_tg_mat = np.full(shape=(len(labels), lb_num_max), fill_value=-1)
        for i in range(len(labels)):
            xyxys_tg_mat[i, :lb_nums[i]] = xyxys_tg[i]
            cinds_tg_mat[i, :lb_nums[i]] = cinds_tg[i]

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_lb = np.concatenate(inds_lb, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        targets = (inds_b_pos, inds_lb, inds_pos, xyxys_tg_mat, cinds_tg_mat, inds_layer)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        inds_b_pos, inds_lb, inds_pos, xyxys_tg_mat, cinds_tg_mat, inds_layer = targets
        inds_b_pos = torch.as_tensor(inds_b_pos, dtype=torch.long).to(preds.device, non_blocking=True)
        inds_lb = torch.as_tensor(inds_lb, dtype=torch.long).to(preds.device, non_blocking=True)
        inds_pos = torch.as_tensor(inds_pos, dtype=torch.long).to(preds.device, non_blocking=True)

        xyxys_tg_mat = torch.as_tensor(xyxys_tg_mat, dtype=torch.float).to(preds.device, non_blocking=True)
        cinds_tg_mat = torch.as_tensor(cinds_tg_mat, dtype=torch.long).to(preds.device, non_blocking=True)
        xyxys_tg_prep = xyxys_tg_mat[inds_b_pos, inds_lb]
        cinds_tg_prep = cinds_tg_mat[inds_b_pos, inds_lb]

        xyxys_pd = preds[inds_b_pos, inds_pos, :4]
        confs_pd = preds[inds_b_pos, inds_pos, 4 + cinds_tg_prep]

        ious = ropr_arr_xyxysT(xyxys_pd.detach(), xyxys_tg_prep, opr_type=IOU_TYPE.IOU)
        scores = torch.pow(ious, self.alpha) + torch.pow(confs_pd, self.beta)

        score_mat = torch.zeros(size=(preds.size(0), cinds_tg_mat.size(1), preds.size(1)), device=preds.device)
        score_mat[inds_b_pos, inds_lb, inds_pos] = scores

        scores_tpk, inds_pos_mtchd = torch.topk(score_mat, k=self.max_mtch_num, dim=2, largest=True)
        idx_b, idx_lb, idx_mtch = torch.nonzero(scores_tpk > 0, as_tuple=True)

        chots_pd_mtchd = preds[:, :, 4:]
        # chots_tg_mtchd = torch.zeros(size=(preds.size(0), preds.size(1), self.num_cls), device=preds.device)
        chots_tg_mtchd = (chots_pd_mtchd ** 2) * 0.75  # 原版写法？可能是为了平滑0-1标签
        if idx_b.size(0) > 0:
            inds_pos_mtchd = inds_pos_mtchd[idx_b, idx_lb, idx_mtch]
            xyxys_tg_mtchd = xyxys_tg_mat[idx_b, idx_lb]
            cinds_tg_mtchd = cinds_tg_mat[idx_b, idx_lb]
            xyxys_pd_mtchd = preds[idx_b, inds_pos_mtchd, :4]
            ious = ropr_arr_xyxysT(xyxys_pd_mtchd, xyxys_tg_mtchd, opr_type=IOU_TYPE.IOU)
            iou_loss = torch.mean(1 - ious) * 2.5
            chots_tg_mtchd[idx_b, inds_pos_mtchd, cinds_tg_mtchd] = 1
        else:
            iou_loss = torch.as_tensor(0).to(preds.device)

        cls_loss = F.binary_cross_entropy(chots_pd_mtchd, chots_tg_mtchd, reduction='sum') \
                   / max(idx_b.size(0), preds.size(0), 1)
        return OrderedDict(iou=iou_loss, cls=cls_loss)

    @staticmethod
    def MEDIUM(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV6Main.MEDIUM(num_cls=num_cls, img_size=img_size, act=ACT.LK, norm=NORM.BATCH, )
        return YoloV6(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def LARGE(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV6Main.LARGE(num_cls=num_cls, img_size=img_size, act=ACT.LK, norm=NORM.BATCH, )
        return YoloV6(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloV6ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size, )
        return YoloV6(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    # model = YoloV6Bkbn.MEDIUM()
    model = YoloV6Main.LARGE()
    img = torch.zeros(size=(1, 3, 512, 512))
    torch.onnx.export(model, img, './test.onnx', opset_version=11, training=TrainingMode.EVAL)
