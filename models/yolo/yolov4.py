from models.modules import *
from models.base.darknet import DarkNetV4Bkbn, CSPDarkNetV4TinyBkbn, DarkNetBkbn
from models.base.modules import SPP, ParallelCpaBARepeat
from models.yolo.modules import *
from models.yolo.yolov3 import YoloV3, YoloV3Layer, YoloV3ConstLayer, YoloV3DownStream


# ParallelCpaBARepeat
class YoloV4DownStream(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV4DownStream, self).__init__()
        self.cvtrs = nn.ModuleList()
        self.adprs = nn.ModuleList()
        self.mixrs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            in_channels = in_channelss[i]
            if i == len(in_channelss) - 1:
                adpr = nn.Identity()
                cvtr = nn.Identity()
                mixr = YoloV4DownStream.C1C3RepeatSPP(
                    in_channels=in_channels, out_channels=out_channels, repeat_num_pre=3, repeat_num_app=3, act=act,
                    norm=norm)
            else:
                cvtr = Ck1s1NA(in_channels=in_channels, out_channels=in_channels // 2, act=act, norm=norm)
                out_channels_next = out_channelss[i + 1]
                adpr = Ck1s1NA(in_channels=out_channels_next, out_channels=out_channels_next // 2, act=act, norm=norm)
                last_channels = in_channels // 2 + out_channels_next // 2
                mixr = YoloV3DownStream.C1C3Repeat(
                    in_channels=last_channels, out_channels=out_channels, repeat_num=5, act=act, norm=norm)

            self.adprs.append(adpr)
            self.mixrs.append(mixr)
            self.cvtrs.append(cvtr)

    @staticmethod
    def C1C3RepeatSPP(in_channels, out_channels, repeat_num_pre, repeat_num_app, act=ACT.LK, norm=NORM.BATCH):
        convs_pre = [Ck1s1NA(in_channels, out_channels, act=act, norm=norm)]
        inner_channels = out_channels // 2
        for i in range((repeat_num_pre - 1) // 2):
            convs_pre.append(Ck3s1NA(out_channels, inner_channels, act=act, norm=norm))
            convs_pre.append(Ck1s1NA(inner_channels, out_channels, act=act, norm=norm))
        convs_app = [Ck1s1NA(out_channels * 4, out_channels, act=act, norm=norm)]
        for i in range((repeat_num_app - 1) // 2):
            convs_app.append(Ck3s1NA(out_channels, inner_channels, act=act, norm=norm))
            convs_app.append(Ck1s1NA(inner_channels, out_channels, act=act, norm=norm))
        convs = convs_pre + [SPP(kernels=(13, 9, 5), stride=1, shortcut=True)] + convs_app
        return nn.Sequential(*convs)

    def forward(self, feats):
        feat_buff = None
        feats_out = []
        for i in range(len(feats) - 1, -1, -1):
            if i == len(feats) - 1:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](torch.cat(
                    [self.cvtrs[i](feats[i]), F.upsample(self.adprs[i](feat_buff), scale_factor=2)], dim=1))
            feats_out.append(feat_buff)
        feats_out = list(reversed(feats_out))
        return feats_out


class YoloV4UpStream(nn.Module):
    def __init__(self, in_channelss, out_channelss, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV4UpStream, self).__init__()
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
                adpr = Ck3NA(in_channels=out_channels_pre, out_channels=out_channels_pre * 2, stride=2, act=act,
                             norm=norm)
                mixr = YoloV3DownStream.C1C3Repeat(
                    in_channels=out_channels_pre * 2 + in_channels, out_channels=out_channels, repeat_num=5, act=act,
                    norm=norm)
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


class YoloV4Main(DarkNetV4Bkbn):
    ANCHOR_SIZESS = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ]

    def __init__(self, anchor_sizess, num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(0, 0)):
        super(YoloV4Main, self).__init__(**DarkNetBkbn.PARA_R53, act=act, norm=norm)
        self.num_cls = num_cls
        self._img_size = img_size
        self.down = YoloV4DownStream(in_channelss=(256, 512, 1024), out_channelss=(128, 256, 512), act=act, norm=norm)
        self.up = YoloV4UpStream(in_channelss=(128, 256, 512), out_channelss=(128, 256, 512), act=act, norm=norm)

        self.pconvs = ParallelCpaBARepeat(in_channelss=(128, 256, 512), out_channelss=(256, 512, 1024),
                                          kernel_size=3, num_repeat=1, act=act, norm=norm)

        self.layers = nn.ModuleList([
            YoloV3Layer(in_channels=256, anchor_sizes=anchor_sizess[0], stride=8, num_cls=num_cls,
                        img_size=img_size),
            YoloV3Layer(in_channels=512, anchor_sizes=anchor_sizess[1], stride=16, num_cls=num_cls,
                        img_size=img_size),
            YoloV3Layer(in_channels=1024, anchor_sizes=anchor_sizess[2], stride=32, num_cls=num_cls,
                        img_size=img_size)
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
        feats = (feats3, feats4, feats5)
        feats = self.down(feats)
        feats = self.up(feats)
        feats = self.pconvs(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def DefaultAnchor(num_cls=80, img_size=(256, 256), act=ACT.LK, norm=NORM.BATCH):
        return YoloV4Main(YoloV4Main.ANCHOR_SIZESS, num_cls=num_cls, img_size=img_size, act=act, norm=norm)


class YoloV4TinyMain(CSPDarkNetV4TinyBkbn):
    ANCHOR_SIZESS = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
    ]

    def __init__(self, anchor_sizess, num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(0, 0), in_channels=3):
        super(YoloV4TinyMain, self).__init__(channels=32, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self.layers = nn.ModuleList([
            YoloV3Layer(in_channels=256, anchor_sizes=anchor_sizess[0], stride=16, num_cls=num_cls,
                        img_size=img_size),
            YoloV3Layer(in_channels=512, anchor_sizes=anchor_sizess[1], stride=32, num_cls=num_cls,
                        img_size=img_size)
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
        c1 = torch.cat([feat1, self.c1_2(feat2)], dim=1)
        c1 = self.c1(c1)
        c2 = self.c2(feat2)
        feats = (c1, c2)
        pred = [layer(feat) for layer, feat in zip(self.layers, feats)]
        pred = torch.cat(pred, dim=1)
        return pred


class YoloV4ConstMain(nn.Module):
    def __init__(self, anchor_sizess, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloV4ConstMain, self).__init__()
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


class YoloV4(YoloV3):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super().__init__(backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.whr_thresh = 4

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
            balance = torch.Tensor([4, 1, 0.4]).to(device=preds.device, non_blocking=True)[inds_layer]
            weight[inds_b_pos, inds_pos] = balance * 3

            iou_power = 2 - xywhs_tg[:, 2] * xywhs_tg[:, 3] / area
            iou_loss = torch.mean((1 - ious) * iou_power)
            # 分类损失
            cls_loss = F.binary_cross_entropy(chots_pd, chots_tg, reduction='mean') * chots_tg.size(1)
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        # 目标检出损失
        conf_loss = F.binary_cross_entropy(confs_pd, confs_tg, weight=weight, reduction='mean')
        return OrderedDict(conf=conf_loss, iou=iou_loss, cls=cls_loss)

    @staticmethod
    def Std(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV4Main(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size,
                              anchor_sizess=YoloV4Main.ANCHOR_SIZESS)
        return YoloV4(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Tiny(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV4TinyMain(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size,
                                  anchor_sizess=YoloV4TinyMain.ANCHOR_SIZESS)
        return YoloV4(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloV4ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size,
                                   anchor_sizess=YoloV4Main.ANCHOR_SIZESS)
        return YoloV4(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = YoloV4Main.DefaultAnchor(img_size=(416, 416))
    imgs = torch.rand(2, 3, 416, 416)
    # print(model)

    y = model(imgs)
    print(y.size())
