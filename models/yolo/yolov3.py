from models.base.darknet import DarkNetBkbn, ParallelCpaBARepeat
from models.yolo.modules import *
from models.modules import *


class YoloV3DownStream(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV3DownStream, self).__init__()
        self.adprs = nn.ModuleList()
        self.mixrs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            if i == len(in_channelss) - 1:
                last_channels = in_channelss[i]
                adpr = nn.Identity()
            else:
                out_channels_next = out_channelss[i + 1]
                last_channels = in_channelss[i] + out_channels_next // 2
                adpr = Ck1s1NA(in_channels=out_channels_next, out_channels=out_channels_next // 2, act=act, norm=norm)
            mixr = YoloV3DownStream.C1C3Repeat(
                in_channels=last_channels, out_channels=out_channels, repeat_num=5, act=act, norm=norm)
            self.adprs.append(adpr)
            self.mixrs.append(mixr)

    @staticmethod
    def C1C3Repeat(in_channels, out_channels, repeat_num, act=ACT.LK, norm=NORM.BATCH):
        inner_channels = out_channels // 2
        convs = [Ck1s1NA(in_channels, out_channels, act=act, norm=norm)]
        for i in range((repeat_num - 1) // 2):
            convs.append(Ck3s1NA(out_channels, inner_channels, act=act, norm=norm))
            convs.append(Ck1s1NA(inner_channels, out_channels, act=act, norm=norm))
        return nn.Sequential(*convs)

    def forward(self, feats):
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if i == len(feats) - 1:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](torch.cat(
                    [feats[i], F.upsample(self.adprs[i](feat_buff), scale_factor=2)], dim=1))
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class YoloV3Layer(AnchorImgLayer):
    def __init__(self, in_channels, anchor_sizes, stride, num_cls=1, img_size=(256, 256)):
        super().__init__(anchor_sizes=anchor_sizes, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.reg = Ck1s1(in_channels=in_channels, out_channels=self.Na * (num_cls + 5))
        init_sig(self.reg.conv.bias[4:self.Na * (num_cls + 5):(num_cls + 5)], prior_prob=0.1)

    def forward(self, featmap):
        featmap = self.reg(featmap)
        pred = YoloV3Layer.decode(
            featmap=featmap, xy_offset=self.xy_offset, wh_offset=self.wh_offset, stride=self.stride,
            num_cls=self.num_cls)
        return pred

    @staticmethod
    def decode(featmap, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(featmap.device, non_blocking=True)
        wh_offset = wh_offset.to(featmap.device, non_blocking=True)
        Hf, Wf, Na, _ = xy_offset.size()
        featmap = featmap.permute(0, 2, 3, 1)  # (Nb,Hf,Wf,C)
        featmap = featmap.reshape(-1, Hf, Wf, Na, num_cls + 5).contiguous()

        x = (torch.sigmoid(featmap[..., 0]) + xy_offset[..., 0]) * stride
        y = (torch.sigmoid(featmap[..., 1]) + xy_offset[..., 1]) * stride
        w = torch.exp(featmap[..., 2]) * wh_offset[..., 0]
        h = torch.exp(featmap[..., 3]) * wh_offset[..., 1]

        conf_cind = torch.sigmoid(featmap[..., 4:])
        preds = torch.cat([x[..., None], y[..., None], w[..., None], h[..., None], conf_cind], dim=-1).contiguous()
        preds = preds.reshape(-1, Na * Wf * Hf, num_cls + 5)
        return preds


class YoloV3Main(DarkNetBkbn):
    ANCHOR_SIZESS = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ]

    def __init__(self, anchor_sizess, num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(0, 0)):
        super(YoloV3Main, self).__init__(**DarkNetBkbn.PARA_R53, act=act, norm=norm)
        self.num_cls = num_cls
        self._img_size = img_size

        self.down = YoloV3DownStream(in_channelss=(256, 512, 1024), out_channelss=(128, 256, 512), act=act, norm=norm)
        self.pconvs = ParallelCpaBARepeat(in_channelss=(128, 256, 512), out_channelss=(256, 512, 1024), kernel_size=3,
                                          num_repeat=1, act=act, norm=norm)
        self.layers = nn.ModuleList([
            YoloV3Layer(in_channels=256, anchor_sizes=anchor_sizess[0], stride=8,
                        num_cls=num_cls, img_size=img_size),
            YoloV3Layer(in_channels=512, anchor_sizes=anchor_sizess[1], stride=16,
                        num_cls=num_cls, img_size=img_size),
            YoloV3Layer(in_channels=1024, anchor_sizes=anchor_sizess[2], stride=32,
                        num_cls=num_cls, img_size=img_size)
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
        feats5 = self.stage5(feats4)
        feats = [feats3, feats4, feats5]
        feats = self.down(feats)
        feats = self.pconvs(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def DefaultAnchor(num_cls=80, img_size=(256, 256), act=ACT.LK, norm=NORM.BATCH):
        return YoloV3Main(YoloV3Main.ANCHOR_SIZESS, num_cls=num_cls, img_size=img_size, act=act, norm=norm)


class YoloV3TinyMain(nn.Module):
    ANCHOR_SIZESS = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
    ]

    def __init__(self, anchor_sizess, num_cls=20, img_size=(0, 0), act=ACT.LK, norm=NORM.BATCH):
        super(YoloV3TinyMain, self).__init__()
        self.num_cls = num_cls
        self._img_size = img_size
        self.stage1 = nn.Sequential(
            Ck3s1NA(in_channels=3, out_channels=16, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1NA(in_channels=16, out_channels=32, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1NA(in_channels=32, out_channels=64, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1NA(in_channels=64, out_channels=128, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1NA(in_channels=128, out_channels=256, act=act, norm=norm),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1NA(in_channels=256, out_channels=512, act=act, norm=norm),
            # nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            # nn.MaxPool2d(kernel_size=2, stride=1),#原版
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 可部署版本
            Ck3s1NA(in_channels=512, out_channels=1024, act=act, norm=norm),
            Ck1s1NA(in_channels=1024, out_channels=256, act=act, norm=norm),
        )
        self.c1_2 = nn.Sequential(
            Ck1s1NA(in_channels=256, out_channels=128, act=act, norm=norm),
            nn.Upsample(scale_factor=2)
        )
        self.c1 = Ck3s1NA(in_channels=384, out_channels=256, act=act, norm=norm)
        self.c2 = Ck3s1NA(in_channels=256, out_channels=512, act=act, norm=norm)
        self.layers = nn.ModuleList([
            YoloV3Layer(in_channels=256, anchor_sizes=anchor_sizess[0], stride=16,
                        num_cls=num_cls, img_size=img_size),
            YoloV3Layer(in_channels=512, anchor_sizes=anchor_sizess[1], stride=32,
                        num_cls=num_cls, img_size=img_size)
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
        feat1 = self.stage1(imgs)
        feat2 = self.stage2(feat1)
        feat2x = self.c2(feat2)
        feat1x = self.c1(torch.cat([self.c1_2(feat2), feat1], dim=1))
        pred = [layer(feat) for layer, feat in zip(self.layers, [feat1x, feat2x])]
        pred = torch.cat(pred, dim=1)
        return pred


class YoloV3ConstLayer(AnchorImgLayer):
    def __init__(self, batch_size, anchor_sizes, stride, num_cls, img_size=(0, 0)):
        super().__init__(anchor_sizes=anchor_sizes, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.featmap = nn.Parameter(torch.zeros(batch_size, self.Na * (num_cls + 5), self.Hf, self.Wf))
        init_sig(self.featmap[:, 4:self.Na * (num_cls + 5):(num_cls + 5), :, :], prior_prob=0.1)

    def forward(self, featmap):
        pred = YoloV3Layer.decode(
            featmap=self.featmap, xy_offset=self.xy_offset, wh_offset=self.wh_offset, stride=self.stride,
            num_cls=self.num_cls)
        return pred


class YoloV3ConstMain(nn.Module):
    def __init__(self, anchor_sizess, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloV3ConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloV3ConstLayer(batch_size=batch_size, anchor_sizes=anchor_sizess[0], stride=8,
                             num_cls=num_cls, img_size=img_size),
            YoloV3ConstLayer(batch_size=batch_size, anchor_sizes=anchor_sizess[1], stride=16,
                             num_cls=num_cls, img_size=img_size),
            YoloV3ConstLayer(batch_size=batch_size, anchor_sizes=anchor_sizess[2], stride=32,
                             num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloV3(YoloFrame):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super().__init__(backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.whr_thresh = 4

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
    def num_anchor(self):
        return np.sum([layer.num_anchor for layer in self.backbone.layers])

    @property
    def anchors(self):
        return torch.cat([layer.anchors for layer in self.backbone.layers], dim=0)

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        xywhs_tg = [np.zeros(shape=(0, 4))]
        chots_tg = [np.zeros(shape=(0, self.num_cls))]
        for i, boxes in enumerate(labels):
            xywhs, chots = boxes.export_xywhsN_chotsN(num_cls=self.num_cls)
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                Na, stride, Wf, Hf, num_anchor = layer.Na, layer.stride, layer.Wf, layer.Hf, layer.num_anchor
                anchor_sizes = layer.anchor_sizes.numpy()

                ixy = (xywhs[:, :2] // stride).astype(np.int32)
                ids_ancr = (np.clip(ixy[:, None, 1], a_min=0, a_max=Hf) * Wf
                            + np.clip(ixy[:, None, 0], a_min=0, a_max=Wf)) * Na + np.arange(Na)

                whr_val = xywhs[:, None, 2:4] / anchor_sizes[None, :, :]
                whr_val = np.max(np.maximum(whr_val, 1 / whr_val), axis=2)
                whr_filter = whr_val < self.whr_thresh

                ids_lb, _ = np.nonzero(whr_filter)
                ids_ancr = ids_ancr[whr_filter]
                # 去除重复
                ids_ancr, repeat_filter = np.unique(ids_ancr, return_index=True)
                ids_lb = ids_lb[repeat_filter]

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
        inds_b_pos, inds_pos, xywhs_tg, chots_tg, inds_layer = targets
        xywhs_tg = torch.as_tensor(xywhs_tg, dtype=torch.float).to(preds.device, non_blocking=True)
        chots_tg = torch.as_tensor(chots_tg, dtype=torch.float).to(preds.device, non_blocking=True)
        area = self.img_size[0] * self.img_size[1]

        preds_pos = preds[inds_b_pos, inds_pos]
        xywhs_pd = preds_pos[:, :4]
        chots_pd = preds_pos[:, 5:]
        confs_pd = preds[:, :, 4]
        confs_tg = torch.zeros_like(confs_pd, device=preds.device)
        weight = torch.full_like(confs_pd, fill_value=0.5, device=preds.device)

        if xywhs_tg.size(0) > 0:
            ious = ropr_arr_xywhsT(xywhs_pd, xywhs_tg, opr_type=IOU_TYPE.IOU)
            confs_tg[inds_b_pos, inds_pos] = ious.detach().clamp_(min=0, max=1)
            weight[inds_b_pos, inds_pos] = 3
            # 边框损失
            xy_err = torch.sum(torch.pow(xywhs_pd[:, :2] - xywhs_tg[:, :2], 2) / area, dim=1)
            wh_err = torch.sum(torch.pow(xywhs_pd[:, 2:4] - xywhs_tg[:, 2:4], 2) / area, dim=1)
            wh_power = 2 - xywhs_tg[:, 2] * xywhs_tg[:, 3] / area
            xywh_loss = (torch.mean(wh_err * wh_power) + torch.mean(xy_err)) * 5
            # 分类损失
            cls_loss = F.binary_cross_entropy(chots_pd, chots_tg, reduction='mean') * chots_tg.size(1)
        else:
            xywh_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        # 目标检出损失
        conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, weight=weight, reduction='mean')

        return OrderedDict(conf=conf_loss, xywh=xywh_loss, cls=cls_loss)

    @staticmethod
    def Std(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV3Main(num_cls=num_cls, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                              anchor_sizess=YoloV3Main.ANCHOR_SIZESS)
        return YoloV3(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Tiny(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV3TinyMain(num_cls=num_cls, img_size=img_size, act=ACT.LK, norm=NORM.BATCH,
                                  anchor_sizess=YoloV3TinyMain.ANCHOR_SIZESS)
        return YoloV3(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloV3ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size,
                                   anchor_sizess=YoloV3Main.ANCHOR_SIZESS)
        return YoloV3(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = YoloV3Main.DefaultAnchor(img_size=(416, 416))
    imgs = torch.rand(2, 3, 416, 416)
    y = model(imgs)
    print(y.size())
