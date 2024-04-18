from torch.cuda.amp import autocast

from models.base.darknet import DarkNetV5Bkbn, DarkNetV5ExtBkbn, CSPBlockV5
from models.base.modules import SPP
from models.modules import *
from models.template import RadiusBasedCenterPrior, CategoryWeightAdapter
from models.yolo.modules import *
from utils import arrsN2arrsT


class YoloV5DownStream(nn.Module):
    def __init__(self, Module, in_channelss, out_channelss, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV5DownStream, self).__init__()
        self.mixrs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            in_channels = in_channelss[i]
            if i == len(in_channelss) - 1:
                mixr = nn.Identity() if in_channels == out_channels else \
                    Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
            else:
                last_channels = in_channels + out_channelss[i + 1]
                mixr = CSPBlockV5(Module=Module, in_channels=last_channels, out_channels=out_channels,
                                  shortcut_channels=out_channels // 2, backbone_channels=out_channels // 2,
                                  backbone_inner_channels=out_channels // 2, repeat_num=repeat_num, act=act, norm=norm)
            self.mixrs.append(mixr)

    def forward(self, feats):
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if i == len(feats) - 1:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](torch.cat([feats[i], F.upsample(feat_buff, scale_factor=2)], dim=1))
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class YoloV5UpStream(nn.Module):
    def __init__(self, Module, in_channelss, out_channelss, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV5UpStream, self).__init__()
        self.adprs = nn.ModuleList()
        self.mixrs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            in_channels = in_channelss[i]
            if i == 0:
                adpr = nn.Identity()
                mixr = nn.Identity()
            else:
                out_channels_pre = out_channelss[i - 1]
                adpr = Ck3NA(in_channels=out_channels_pre, out_channels=out_channels_pre, stride=2, act=act, norm=norm)
                mixr = CSPBlockV5(Module=Module, in_channels=out_channels_pre + in_channels, out_channels=out_channels,
                                  shortcut_channels=out_channels // 2, backbone_channels=out_channels // 2,
                                  backbone_inner_channels=out_channels // 2, repeat_num=repeat_num, act=act, norm=norm)
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


class YoloV5Layer(AnchorImgLayer):
    def __init__(self, in_channels, anchor_sizes, stride, num_cls, img_size=(0, 0)):
        super().__init__(anchor_sizes=anchor_sizes, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.reg_cls = Ck1(in_channels=in_channels, out_channels=self.Na * (num_cls + 5))
        init_sig(self.reg_cls.conv.bias[4:self.Na * (num_cls + 5):(num_cls + 5)], prior_prob=0.001)

    def forward(self, featmap):
        featmap = self.reg_cls(featmap)
        pred = YoloV5Layer.decode(
            featmap=featmap, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred

    @staticmethod
    def decode(featmap, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(featmap.device, non_blocking=True)
        wh_offset = wh_offset.to(featmap.device, non_blocking=True)
        Hf, Wf, Na, _ = list(xy_offset.size())
        featmap = featmap.permute(0, 2, 3, 1)
        featmap = featmap.reshape(-1, Hf, Wf, Na, num_cls + 5).contiguous()

        # tensorRT不支持4维度以上tensor
        # x = (torch.sigmoid(featmap[..., 0]) * 2 - 0.5 + xy_offset[..., 0]) * stride
        # y = (torch.sigmoid(featmap[..., 1]) * 2 - 0.5 + xy_offset[..., 1]) * stride
        # w = (torch.sigmoid(featmap[..., 2]) * 2) ** 2 * wh_offset[..., 0]
        # h = (torch.sigmoid(featmap[..., 3]) * 2) ** 2 * wh_offset[..., 1]

        reg_xy = torch.sigmoid(featmap[..., 0:2]) * 2 - 0.5
        reg_wh = (torch.sigmoid(featmap[..., 2:4]) * 2) ** 2 * (wh_offset / stride)
        xy = (reg_xy + xy_offset) * stride
        wh = reg_wh * stride

        conf = torch.sigmoid(featmap[..., 4:5])
        # conf_cls = torch.sigmoid(featmap[..., 4:])
        chot = featmap[..., 5:]
        pred = torch.cat([xy, wh, conf, chot], dim=-1).contiguous()
        pred = pred.reshape(-1, Na * Wf * Hf, num_cls + 5)
        return pred


class YoloV5Main(DarkNetV5Bkbn, ImageONNXExportable):
    ANCHOR_SIZESS = (
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    )

    def __init__(self, Module, channels, repeat_nums, anchor_sizess, num_cls=80, act=ACT.SILU, norm=NORM.BATCH,
                 img_size=(256, 256),
                 in_channels=3):
        DarkNetV5Bkbn.__init__(self, Module, channels, repeat_nums, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        feat_channelss = (channels * 4, channels * 8, channels * 8)
        down_channelss = (channels * 4, channels * 8, channels * 8)
        up_channelss = (channels * 4, channels * 8, channels * 16)
        self.spp = YoloV5Main.C1C3RepeatSPP(in_channels=channels * 16, out_channels=channels * 8, act=act, norm=norm)
        self.down = YoloV5DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act,
                                     norm=norm)
        self.up = YoloV5UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act, norm=norm)
        self.layers = nn.ModuleList([
            YoloV5Layer(in_channels=channels * 4, anchor_sizes=anchor_sizess[0], stride=8,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 8, anchor_sizes=anchor_sizess[1], stride=16,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 16, anchor_sizes=anchor_sizess[2], stride=32,
                        num_cls=num_cls, img_size=img_size)
        ])

    @staticmethod
    def C1C3RepeatSPP(in_channels, out_channels, act=ACT.LK, norm=NORM.BATCH):
        convs = [
            Ck1s1NA(in_channels=in_channels, out_channels=in_channels // 2, act=act, norm=norm),
            SPP(kernels=(13, 9, 5), stride=1, shortcut=True),
            Ck1s1NA(in_channels=in_channels * 2, out_channels=out_channels * 2, act=act, norm=norm),
            Ck1s1NA(in_channels=out_channels * 2, out_channels=out_channels, act=act, norm=norm)
        ]
        return nn.Sequential(*convs)

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
    def Nano(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
             anchor_sizess=ANCHOR_SIZESS):
        return YoloV5Main(**DarkNetV5Bkbn.PARA_NANO, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
              anchor_sizess=ANCHOR_SIZESS):
        return YoloV5Main(**DarkNetV5Bkbn.PARA_SMALL, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
               anchor_sizess=ANCHOR_SIZESS):
        return YoloV5Main(**DarkNetV5Bkbn.PARA_MEDIUM, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
              anchor_sizess=ANCHOR_SIZESS):
        return YoloV5Main(**DarkNetV5Bkbn.PARA_LARGE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
               anchor_sizess=ANCHOR_SIZESS):
        return YoloV5Main(**DarkNetV5Bkbn.PARA_XLARGE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          anchor_sizess=anchor_sizess, in_channels=in_channels)


class YoloV5ExtMain(DarkNetV5ExtBkbn, ImageONNXExportable):
    ANCHOR_SIZESS = (
        [[20, 13], [21, 32], [41, 22]],
        [[20, 88], [48, 49], [99, 32]],
        [[52, 144], [114, 65], [143, 190], [238, 96]],
        [[85, 324], [427, 213], [40, 400], [400, 40]],
    )

    def __init__(self, Module, channels, repeat_nums, anchor_sizess, num_cls=80, act=ACT.SILU, norm=NORM.BATCH,
                 img_size=(256, 256),
                 in_channels=3):
        DarkNetV5ExtBkbn.__init__(self, Module, channels, repeat_nums, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        feat_channelss = (channels * 4, channels * 8, channels * 16, channels * 8)
        down_channelss = (channels * 4, channels * 4, channels * 8, channels * 8)
        up_channelss = (channels * 4, channels * 4, channels * 8, channels * 8)
        self.spp = YoloV5Main.C1C3RepeatSPP(in_channels=channels * 16, out_channels=channels * 8, act=act, norm=norm)
        self.down = YoloV5DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act,
                                     norm=norm)
        self.up = YoloV5UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act, norm=norm)

        self.layers = nn.ModuleList([
            YoloV5Layer(in_channels=channels * 4, anchor_sizes=anchor_sizess[0], stride=8,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 4, anchor_sizes=anchor_sizess[1], stride=16,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 8, anchor_sizes=anchor_sizess[2], stride=32,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 8, anchor_sizes=anchor_sizess[3], stride=64,
                        num_cls=num_cls, img_size=img_size)
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
        feats5 = self.stage5(feats4)
        feats5 = self.spp(feats5)
        feats = (feats2, feats3, feats4, feats5)
        feats = self.down(feats)
        feats = self.up(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
             anchor_sizess=ANCHOR_SIZESS):
        return YoloV5ExtMain(**DarkNetV5ExtBkbn.PARA_NANO, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
              anchor_sizess=ANCHOR_SIZESS):
        return YoloV5ExtMain(**DarkNetV5ExtBkbn.PARA_SMALL, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
               anchor_sizess=ANCHOR_SIZESS):
        return YoloV5ExtMain(**DarkNetV5ExtBkbn.PARA_MEDIUM, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
              anchor_sizess=ANCHOR_SIZESS):
        return YoloV5ExtMain(**DarkNetV5ExtBkbn.PARA_LARGE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             anchor_sizess=anchor_sizess, in_channels=in_channels)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3,
               anchor_sizess=ANCHOR_SIZESS):
        return YoloV5ExtMain(**DarkNetV5ExtBkbn.PARA_XLARGE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             anchor_sizess=anchor_sizess, in_channels=in_channels)


class YoloV5ConstLayer(AnchorImgLayer):
    def __init__(self, batch_size, anchor_sizes, stride, num_cls, img_size=(0, 0)):
        super().__init__(anchor_sizes=anchor_sizes, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.featmap = nn.Parameter(torch.zeros(batch_size, self.Na * (num_cls + 5), self.Hf, self.Wf))
        init_sig(self.featmap[:, 4:self.Na * (num_cls + 5):(num_cls + 5), :, :], prior_prob=0.1)

    def forward(self, featmap):
        featmap = self.featmap
        pred = YoloV5Layer.decode(featmap=featmap, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
                                  stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloV5ConstMain(nn.Module):
    def __init__(self, anchor_sizess, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloV5ConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloV5ConstLayer(batch_size=batch_size, anchor_sizes=anchor_sizess[0], stride=8,
                             num_cls=num_cls, img_size=img_size),
            YoloV5ConstLayer(batch_size=batch_size, anchor_sizes=anchor_sizess[1], stride=16,
                             num_cls=num_cls, img_size=img_size),
            YoloV5ConstLayer(batch_size=batch_size, anchor_sizes=anchor_sizess[2], stride=32,
                             num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloV5(YoloFrame, RadiusBasedCenterPrior, CategoryWeightAdapter):
    def __init__(self, backbone, device=None, pack=PACK.AUTO, **kwargs):
        YoloFrame.__init__(self, backbone=backbone, device=device, pack=pack)
        CategoryWeightAdapter.__init__(self)
        self.layers = backbone.layers
        self.whr_thresh = 4
        self.radius = 0.5

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        xywhs_tg = [np.zeros(shape=(0, 4))]
        chots_tg = [np.zeros(shape=(0, self.num_cls))]
        offset_lb = self.offset_lb

        for i, label in enumerate(labels):
            # 类别筛选
            label = label.filt(lambda item: item.category.cindN < 2)

            xywhs, chots = label.export_xywhsN_chotsN(num_cls=self.num_cls)
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                Na, stride, Wf, Hf, num_anchor = layer.Na, layer.stride, layer.Wf, layer.Hf, layer.num_anchor
                anchor_sizes = layer.anchor_sizes.numpy()

                ixys = (xywhs[:, None, :2] / stride + offset_lb).astype(np.int32)
                fltr_in = (ixys[..., 0] >= 0) * (ixys[..., 0] < Wf) * (ixys[..., 1] >= 0) * (ixys[..., 1] < Hf)
                ids_ancr = (ixys[:, :, None, 1] * Wf + ixys[:, :, None, 0]) * Na + np.arange(Na)
                whr_val = xywhs[:, None, 2:4] / anchor_sizes[None, :, :]
                whr_val = np.max(np.maximum(whr_val, 1 / whr_val), axis=2)
                fltr_whr = whr_val < self.whr_thresh
                fltr_whr = np.repeat(fltr_whr[:, None, :], axis=1, repeats=offset_lb.shape[0])
                fltr_valid = fltr_whr * fltr_in[:, :, None]

                ids_lb, ids_dt, ids_az = np.nonzero(fltr_valid)
                ids_ancr = ids_ancr[fltr_valid]

                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_lb)))
                inds_layer.append(np.full(fill_value=j, shape=len(ids_lb)))
                inds_pos.append(offset_layer + ids_ancr)
                chots_tg.append(chots[ids_lb])
                xywhs_tg.append(xywhs[ids_lb])

                offset_layer = offset_layer + num_anchor

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        xywhs_tg = np.concatenate(xywhs_tg, axis=0)
        chots_tg = np.concatenate(chots_tg, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        targets = (inds_b_pos, inds_pos, xywhs_tg, chots_tg, inds_layer)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        inds_b_pos, inds_pos, xywhs_tg, chots_tg, inds_layer = arrsN2arrsT(targets, device=self.device)

        confs_pd = preds[..., 4]
        confs_tg = torch.zeros_like(confs_pd, device=preds.device)
        weight_conf = torch.full_like(confs_pd, fill_value=1, device=preds.device)
        if inds_b_pos.size(0) > 0:
            xywhs_pd, _, chots_pd = preds[inds_b_pos, inds_pos].split((4, 1, self.num_cls), dim=-1)
            ious = ropr_arr_xywhsT(xywhs_pd, xywhs_tg, opr_type=IOU_TYPE.IOU)
            confs_tg[inds_b_pos, inds_pos] = ious.detach()
            iou_loss = 1 - torch.mean(ious)
            # 分类损失
            with autocast(enabled=False):
                weight_cls = self.get_weight_cls(chots_tg)
                cls_loss = F.binary_cross_entropy_with_logits(
                    chots_pd, chots_tg, weight=weight_cls, reduction='mean')
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        # 目标检出损失
        with autocast(enabled=False):
            conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, weight=weight_conf, reduction='mean')
        return OrderedDict(conf=conf_loss * 100, iou=iou_loss, cls=cls_loss)

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Nano(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Small(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, **kwargs):
        backbone = YoloV5Main.Medium(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack, **kwargs)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, **kwargs):
        backbone = YoloV5Main.Large(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack, **kwargs)

    @staticmethod
    def XLarge(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.XLarge(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloV5ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size,
                                   anchor_sizess=YoloV5Main.ANCHOR_SIZESS)
        return YoloV5(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = YoloV5Main.Medium(img_size=(256, 256))
    # imgs = torch.rand(2, 3, 256, 256)
    # y = model(imgs)
    # print(y.size())
    model.export_onnx('./buff2')
