from models.base import VGGBkbn
from models.base.fpn import *
from utils.frame import *


# <editor-fold desc='主干'>
class RPNLayer(AnchorImgLayer):
    def __init__(self, in_channels=512, img_size=(0, 0), stride=16, scales=(8, 16, 32), wh_ratios=(0.5, 1, 2), base=14):
        anchor_sizes = generate_anchor_sizes(scales=scales, wh_ratios=wh_ratios, base=base)
        super(RPNLayer, self).__init__(anchor_sizes, stride, img_size=img_size)
        self.xywh = Ck1s1(in_channels=in_channels, out_channels=4 * self.Na)
        self.conf = Ck1s1(in_channels=in_channels, out_channels=self.Na)
        init_sig(self.conf.conv.bias, prior_prob=0.1)

    @staticmethod
    def decode(xywh, conf, xy_offset, wh_offset, stride):
        xy_offset = xy_offset.to(xywh.device, non_blocking=True)
        wh_offset = wh_offset.to(xywh.device, non_blocking=True)
        Hf, Wf, Na, _ = xy_offset.size()
        xywh = xywh.permute(0, 2, 3, 1)
        xywh = xywh.reshape(-1, Hf, Wf, Na, 4).clamp(min=-5, max=5)

        x = xywh[..., 0] * wh_offset[..., 0] + (xy_offset[..., 0] + 0.5) * stride
        y = xywh[..., 1] * wh_offset[..., 1] + (xy_offset[..., 1] + 0.5) * stride
        w = torch.exp(xywh[..., 2]) * wh_offset[..., 0] + 1e-7
        h = torch.exp(xywh[..., 3]) * wh_offset[..., 1] + 1e-7

        conf = conf.permute(0, 2, 3, 1)
        conf = conf.reshape(-1, Hf, Wf, Na, 1)
        conf = torch.sigmoid(conf)

        pred = torch.cat([x[..., None], y[..., None], w[..., None], h[..., None], conf], dim=-1).contiguous()
        pred = pred.view(-1, Na * Wf * Hf, 5)
        return pred

    def forward(self, featmap):
        xywh = self.xywh(featmap)
        conf = self.conf(featmap)
        preds = RPNLayer.decode(xywh=xywh, conf=conf, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
                                stride=self.stride)
        return preds


class RPNLayerConst(AnchorImgLayer):
    def __init__(self, batch_size, img_size=(0, 0), stride=16, scales=(8, 16, 32), wh_ratios=(0.5, 1, 2), base=14):
        anchors = generate_anchor_sizes(scales=scales, wh_ratios=wh_ratios, base=base)
        super(RPNLayerConst, self).__init__(anchors, stride, img_size=img_size)
        self.featmap = nn.Parameter(torch.zeros(batch_size, self.Na * 5, self.Hf, self.Wf))
        init_sig(self.featmap[:, self.Na * 4:, :, :], prior_prob=0.01)

    def forward(self, featmap):
        xywh = self.featmap[:, :self.Na * 4, :, :]
        conf = self.featmap[:, self.Na * 4:, :, :]
        pred = RPNLayer.decode(xywh=xywh, conf=conf, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
                               stride=self.stride)
        return pred


class RPNOneScaleVGGBkbn(VGGBkbn):
    def __init__(self, repeat_nums=(1, 1, 1, 1), act=ACT.RELU,norm=NORM.BATCH, img_size=(0, 0)):
        super(RPNOneScaleVGGBkbn, self).__init__(repeat_nums=repeat_nums, act=act,norm=norm)
        self.layer = RPNLayer(in_channels=512, img_size=img_size, stride=16,
                              scales=(8, 16, 32), wh_ratios=(0.5, 1, 2), base=16)
        self.layers = [self.layer]

    @property
    def anchors(self):
        return self.rpn.anchors

    def forward(self, imgs):
        featmap = super(RPNOneScaleVGGBkbn, self).forward(imgs)
        rps = self.layer(featmap)
        return featmap, rps

    @staticmethod
    def A(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleVGGBkbn(**VGGBkbn.PARA_A, act=act,norm=norm, img_size=img_size)

    @staticmethod
    def B(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleVGGBkbn(**VGGBkbn.PARA_B, act=act,norm=norm, img_size=img_size)

    @staticmethod
    def D(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleVGGBkbn(**VGGBkbn.PARA_D, act=act,norm=norm, img_size=img_size)

    @staticmethod
    def E(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleVGGBkbn(**VGGBkbn.PARA_E, act=act,norm=norm, img_size=img_size)


class RPNOneScaleResNetBkbn(ResNetBkbnMutiOut3):
    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        super(RPNOneScaleResNetBkbn, self).__init__(Module, repeat_nums, channels, act=act,norm=norm)
        self._img_size = img_size
        self.layer = RPNLayer(in_channels=channels * 4, img_size=img_size, stride=16)
        self.layers = [self.layer]

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        for rpn in self.layers:
            rpn.img_size = img_size

    @property
    def anchors(self):
        return self.rpn.anchors

    def forward(self, imgs):
        feats1, feats2, feats3 = super(RPNOneScaleResNetBkbn, self).forward(imgs)
        preds = self.layer(feats3)
        return feats3, preds

    @staticmethod
    def R18(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleResNetBkbn(**ResNetBkbn.PARA_R18, act=act,norm=norm, img_size=img_size)

    @staticmethod
    def R34(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleResNetBkbn(**ResNetBkbn.PARA_R34, act=act,norm=norm, img_size=img_size)

    @staticmethod
    def R50(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleResNetBkbn(**ResNetBkbn.PARA_R50, act=act,norm=norm, img_size=img_size)

    @staticmethod
    def R101(act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224)):
        return RPNOneScaleResNetBkbn(**ResNetBkbn.PARA_R101, act=act,norm=norm, img_size=img_size)


class RPNMutiScaleResNetBkbn(ResNetBkbnMutiOut4):
    def __init__(self, Module, repeat_nums, channels=64, img_size=(512, 512), anchor_ratio=8, act=ACT.RELU,norm=NORM.BATCH):
        super(RPNMutiScaleResNetBkbn, self).__init__(Module, repeat_nums, channels, act=act,norm=norm)

        self._img_size = img_size
        self.anchor_ratio = anchor_ratio
        in_channelss = [channels, channels * 2, channels * 4, channels * 8]
        out_channelss = [channels] * 4
        self.down = FPNDownStreamAdd(in_channelss=in_channelss, out_channelss=out_channelss)
        out_channelss += [out_channelss[-1]]
        self.pconvs = ParallelCpaBARepeat(in_channelss=out_channelss, out_channelss=out_channelss,
                                          kernel_size=3, num_repeat=2, act=act,norm=norm)
        self.layers = nn.ModuleList(
            [RPNLayer(in_channels=channels, img_size=img_size, stride=stride,
                      scales=(1,), wh_ratios=(0.5, 1, 2), base=stride * anchor_ratio) for stride in [4, 8, 16, 32, 64]])

    @property
    def anchor_sizes(self):
        return torch.cat([rpn.anchor_sizes for rpn in self.layers], dim=0)

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        for rpn in self.layers:
            rpn.img_size = img_size

    @property
    def anchors(self):
        return torch.cat([rpn.anchors for rpn in self.layers], dim=0)

    def forward(self, imgs):
        feats = super(RPNMutiScaleResNetBkbn, self).forward(imgs)
        feats = self.down(feats)
        feats.append(F.max_pool2d(feats[-1], kernel_size=1, stride=2))
        feats = self.pconvs(feats)
        preds = torch.cat([rpn(feat) for feat, rpn in zip(feats, self.layers)], dim=1)
        return feats, preds

    @staticmethod
    def R18(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return RPNMutiScaleResNetBkbn(**ResNetBkbn.PARA_R18, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R34(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return RPNMutiScaleResNetBkbn(**ResNetBkbn.PARA_R34, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R50(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return RPNMutiScaleResNetBkbn(**ResNetBkbn.PARA_R50, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R101(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return RPNMutiScaleResNetBkbn(**ResNetBkbn.PARA_R101, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R152(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return RPNMutiScaleResNetBkbn(**ResNetBkbn.PARA_R152, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)


class PANetMutiScaleResNetBkbn(RPNMutiScaleResNetBkbn):
    def __init__(self, Module, repeat_nums, channels=64, img_size=(512, 512), anchor_ratio=4, act=ACT.RELU,norm=NORM.BATCH):
        super(PANetMutiScaleResNetBkbn, self).__init__(Module, repeat_nums, channels=channels, img_size=img_size,
                                                       anchor_ratio=anchor_ratio, act=act,norm=norm)
        out_channelss = [channels] * 5
        self.up = FPNUpStreamAdd(in_channelss=out_channelss, out_channelss=out_channelss, act=act,norm=norm)

    def forward(self, imgs):
        feats = super(RPNMutiScaleResNetBkbn, self).forward(imgs)
        feats = self.down(feats)
        feats.append(F.max_pool2d(feats[-1], kernel_size=1, stride=2))
        feats = self.pconvs(feats)
        feats = self.up(feats)
        preds = torch.cat([rpn(feat) for feat, rpn in zip(feats, self.layers)], dim=1)
        return feats, preds

    @staticmethod
    def R18(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return PANetMutiScaleResNetBkbn(**ResNetBkbn.PARA_R18, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R34(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return PANetMutiScaleResNetBkbn(**ResNetBkbn.PARA_R34, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R50(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return PANetMutiScaleResNetBkbn(**ResNetBkbn.PARA_R50, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R101(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return PANetMutiScaleResNetBkbn(**ResNetBkbn.PARA_R101, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)

    @staticmethod
    def R152(img_size=(512, 512), act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=8):
        return PANetMutiScaleResNetBkbn(**ResNetBkbn.PARA_R152, act=act,norm=norm, img_size=img_size, anchor_ratio=anchor_ratio)


# </editor-fold>

# <editor-fold desc='分类器'>

class FasterRCNNResNetClassifier(nn.Module):
    def __init__(self, Module, repeat_nums, channels=64, num_cls=20, act=ACT.RELU,norm=NORM.BATCH, feat_size=(14, 14)):
        super(FasterRCNNResNetClassifier, self).__init__()
        self.feat_size = feat_size
        self.num_cls = num_cls
        self.stage4 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 4, out_channels=channels * 8, stride=2,
                                              repeat_num=repeat_nums[3], act=act,norm=norm, with_pool=False)
        self.in_channels = channels * 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reg_chot = nn.Linear(channels * 8, num_cls + 1)
        self.reg_detla = nn.Linear(channels * 8, (num_cls + 1) * 4)

    @staticmethod
    def decode(xywh, reg_detla, reg_chot, num_cls):
        reg_detla = reg_detla.view(-1, num_cls + 1, 4)
        reg_detla = reg_detla.clamp_(min=-4, max=4)

        xywh = xywh[:, None, :].expand(xywh.size(0), num_cls + 1, 4)
        xy, wh = xywh[..., :2].detach(), xywh[..., 2:4].detach()
        xy_regd = xy + reg_detla[..., :2] * wh
        wh_regd = wh * torch.exp(reg_detla[..., 2:4])
        rois_regd = torch.cat([xy_regd, wh_regd, reg_chot[:, :, None]], dim=-1)
        return rois_regd

    def forward(self, feat, xywh):
        feat = self.stage4(feat)
        feat = self.pool(feat)
        feat = feat.reshape(-1, self.in_features)
        reg_chot = self.reg_chot(feat)
        reg_detla = self.reg_detla(feat)
        rois_regd = FasterRCNNResNetClassifier.decode(xywh, reg_detla, reg_chot, self.num_cls)
        return rois_regd

    @staticmethod
    def R18(act=ACT.RELU,norm=NORM.BATCH, num_cls=20, feat_size=(14, 14)):
        return FasterRCNNResNetClassifier(**ResNetBkbn.PARA_R18, act=act,norm=norm, num_cls=num_cls, feat_size=feat_size)

    @staticmethod
    def R34(act=ACT.RELU,norm=NORM.BATCH, num_cls=20, feat_size=(14, 14)):
        return FasterRCNNResNetClassifier(**ResNetBkbn.PARA_R34, act=act,norm=norm, num_cls=num_cls, feat_size=feat_size)

    @staticmethod
    def R50(act=ACT.RELU,norm=NORM.BATCH, num_cls=20, feat_size=(14, 14)):
        return FasterRCNNResNetClassifier(**ResNetBkbn.PARA_R50, act=act,norm=norm, num_cls=num_cls, feat_size=feat_size)

    @staticmethod
    def R101(act=ACT.RELU,norm=NORM.BATCH, num_cls=20, feat_size=(14, 14)):
        return FasterRCNNResNetClassifier(**ResNetBkbn.PARA_R101, act=act,norm=norm, num_cls=num_cls, feat_size=feat_size)


class FasterRCNNMLPClassifier(nn.Module):
    def __init__(self, in_channels=64, num_cls=20, inner_features=1024, act=ACT.RELU,norm=NORM.BATCH, feat_size=(7, 7)):
        super(FasterRCNNMLPClassifier, self).__init__()
        self.num_cls = num_cls
        self.feat_size = feat_size
        self.in_channels = in_channels
        self.in_features = in_channels * feat_size[0] * feat_size[1]
        self.stem = nn.Sequential(
            nn.Linear(self.in_features, inner_features),
            ACT.build(act),
            nn.Linear(inner_features, inner_features),
            ACT.build(act),
        )
        self.reg_chot = nn.Linear(inner_features, num_cls + 1)
        self.reg_detla = nn.Linear(inner_features, (num_cls + 1) * 4)

    def forward(self, feat, xywh):
        feat = feat.view(-1, self.in_features)
        feat = self.stem(feat)
        reg_chot = self.reg_chot(feat)
        reg_detla = self.reg_detla(feat)
        rois_regd = FasterRCNNResNetClassifier.decode(xywh, reg_detla, reg_chot, self.num_cls)
        return rois_regd


class PANetMixer(nn.Module):
    def __init__(self, in_channels=64, num_layers=5, act=ACT.RELU,norm=NORM.BATCH):
        super().__init__()
        channels = in_channels * num_layers
        self.cvter = nn.Sequential(
            Ck3s1NA(in_channels=channels, out_channels=channels, groups=num_layers, act=act,norm=norm),
            Ck1s1NA(in_channels=channels, out_channels=channels, groups=num_layers, act=act,norm=norm),
        )
        self.num_layers = num_layers
        self.in_channels = in_channels

    def forward(self, feats):
        feats = self.cvter(feats)
        feats = feats.view(feats.size(0), self.num_layers, self.in_channels, feats.size(2), feats.size(3))
        feats, _ = torch.max(feats, dim=1)
        return feats


class PANetResNetClassifier(FasterRCNNResNetClassifier):
    def __init__(self, Module, repeat_nums, channels=64, num_cls=20, num_layer=5, act=ACT.RELU,norm=NORM.BATCH, feat_size=(7, 7)):
        super(PANetResNetClassifier, self).__init__(Module=Module, repeat_nums=repeat_nums, channels=channels,
                                                    num_cls=num_cls, act=act,norm=norm, feat_size=feat_size)
        self.mixer = PANetMixer(in_channels=channels * 8, num_layers=num_layer, act=act,norm=norm)

    def forward(self, feats, xywh):
        feat = self.mixer(feats)
        rois_regd = super(PANetResNetClassifier, self).forward(feat, xywh)
        return rois_regd


class PANetMLPClassifier(FasterRCNNMLPClassifier):
    def __init__(self, in_channels=64, num_cls=20, inner_features=1024, num_layer=5, act=ACT.RELU,norm=NORM.BATCH, feat_size=(7, 7)):
        super(PANetMLPClassifier, self).__init__(in_channels=in_channels, num_cls=num_cls,
                                                 inner_features=inner_features, act=act,norm=norm, feat_size=feat_size)
        self.mixer = PANetMixer(in_channels=in_channels, num_layers=num_layer, act=act,norm=norm)

    def forward(self, feats, xywh):
        feat = self.mixer(feats)
        rois_regd = super(PANetMLPClassifier, self).forward(feat, xywh)
        return rois_regd


# </editor-fold>

# <editor-fold desc='掩码器'>

# 输入一定尺度特征图，输出两倍大小分割结果
class MaskRCNNRepeatConvMasker(nn.Module):
    def __init__(self, in_channels=64, num_cls=20, act=ACT.RELU,norm=NORM.BATCH, feat_size=(14, 14), repeat_num=4):
        super(MaskRCNNRepeatConvMasker, self).__init__()
        self.feat_size = feat_size
        self.mask_size = (feat_size[0] * 2, feat_size[1] * 2)
        self.num_cls = num_cls
        self.in_channels = in_channels
        stem = []
        for i in range(repeat_num):
            stem.append(Ck3s1NA(in_channels=in_channels, out_channels=in_channels, act=act,norm=norm)),
        self.stem = nn.Sequential(*stem)
        self.mask_head = nn.Sequential(
            CTk3NA(in_channels=in_channels, out_channels=in_channels, stride=2, act=act,norm=norm),
            Ck1s1(in_channels=in_channels, out_channels=num_cls)
        )

    def forward(self, feat):
        feat = self.stem(feat)
        mask = self.mask_head(feat)
        mask = torch.sigmoid(mask)
        return mask


class PANetRepeatConvMasker(MaskRCNNRepeatConvMasker):
    def __init__(self, in_channels=64, num_cls=20, act=ACT.RELU,norm=NORM.BATCH, feat_size=(14, 14), repeat_num=4, num_layer=5):
        super(PANetRepeatConvMasker, self).__init__(in_channels=in_channels, num_cls=num_cls, act=act,norm=norm,
                                                    feat_size=feat_size, repeat_num=repeat_num)
        self.mixer = PANetMixer(in_channels=in_channels, num_layers=num_layer, act=act,norm=norm)
        self.linear_head = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=in_channels // 2, act=act,norm=norm),
            Ck3s1NA(in_channels=in_channels // 2, out_channels=in_channels // 4, act=act,norm=norm)
        )
        self.in_features = in_channels // 4 * self.feat_size[0] * self.feat_size[1]
        out_features = self.mask_size[0] * self.mask_size[1]
        self.linear = nn.Linear(in_features=self.in_features, out_features=out_features)

    def forward(self, feats):
        feat = self.mixer(feats)
        feat = self.stem(feat)
        mask = self.mask_head(feat)
        gol = self.linear_head(feat)
        gol = self.linear(gol.view(gol.size(0), self.in_features))
        gol = gol.view(gol.size(0), 1, self.mask_size[1], self.mask_size[0])
        mask = torch.sigmoid(mask + gol)
        return mask


# </editor-fold>

# <editor-fold desc='常量模块'>
class FasterRCNNConstBkbn(nn.Module):
    def __init__(self, batch_size, img_size=(0, 0), out_channels=512):
        super(FasterRCNNConstBkbn, self).__init__()
        self.layer = RPNLayerConst(batch_size=batch_size, img_size=img_size, stride=16)
        self.layers = [self.layer]
        self.featmap = nn.Parameter(torch.zeros(batch_size, out_channels, self.rpn.Hf, self.rpn.Wf))

    @property
    def anchors(self):
        return self.rpn.anchors

    def forward(self, imgs):
        assert imgs.size(0) == self.featmap.size(0), 'size err'
        rois = self.rpn(None)
        return self.featmap, rois


class MaskRCNNConstBkbn(nn.Module):
    def __init__(self, batch_size, img_size=(0, 0), out_channels=512, anchor_ratio=4):
        super(MaskRCNNConstBkbn, self).__init__()
        self.img_size = img_size
        self.anchor_ratio = anchor_ratio
        self.layers = nn.ModuleList([])
        self.featmaps = nn.ParameterList([])
        for stride in [4, 8, 16, 32, 64]:
            rpn = RPNLayerConst(batch_size=batch_size, img_size=img_size, stride=stride,
                                scales=(1,), wh_ratios=(0.5, 1, 2), base=stride * anchor_ratio)
            self.layers.append(rpn)
            self.featmaps.append(nn.Parameter(torch.zeros(batch_size, out_channels, rpn.Hf, rpn.Wf)))

    @property
    def anchor_sizes(self):
        return torch.cat([rpn.anchor_sizes for rpn in self.layers], dim=0)

    @property
    def anchors(self):
        return torch.cat([rpn.anchors for rpn in self.layers], dim=0)

    def forward(self, imgs):
        for featmap in self.featmaps:
            assert imgs.size(0) == featmap.size(0), 'size err'
        preds = torch.cat([rpn(None) for rpn in self.layers], dim=1)
        return self.featmaps, preds


class FasterRCNNConstClassifier(nn.Module):
    def __init__(self, in_channels=64, num_cls=20, feat_size=(7, 7)):
        super(FasterRCNNConstClassifier, self).__init__()
        assert in_channels == (num_cls + 1) * 5 + num_cls, 'len err'
        self.num_cls = num_cls
        self.feat_size = feat_size
        self.in_channels = in_channels

    def forward(self, featmap, xywh):
        featmap = F.adaptive_max_pool2d(featmap, output_size=(1, 1))
        featmap = featmap.squeeze(dim=-1).squeeze(dim=-1)
        cls = featmap[:, :(self.num_cls + 1)]
        creg = featmap[:, (self.num_cls + 1):(self.num_cls + 1) * 5]
        rois_regd = FasterRCNNResNetClassifier.decode(xywh, creg, cls, self.num_cls)
        return rois_regd


class MaskRCNNConstMasker(nn.Module):
    def __init__(self, in_channels=64, num_cls=20, feat_size=(14, 14)):
        super(MaskRCNNConstMasker, self).__init__()
        assert in_channels == (num_cls + 1) * 5 + num_cls, 'len err'
        self.in_channels = in_channels
        self.num_cls = num_cls
        self.feat_size = tuple(feat_size)
        self.scale = 2
        self.mask_size = (feat_size[0] * self.scale, feat_size[1] * self.scale)

    def forward(self, featmap):
        featmap = featmap[:, (self.num_cls + 1) * 5:]
        featmap = F.upsample(featmap, scale_factor=self.scale)
        featmap = torch.sigmoid(featmap)
        return featmap


class PANetConstClassifier(FasterRCNNConstClassifier):
    def __init__(self, in_channels=64, num_cls=20, num_layer=5, feat_size=(7, 7)):
        super(PANetConstClassifier, self).__init__(in_channels=in_channels, num_cls=num_cls, feat_size=feat_size)
        self.num_layer = num_layer

    @staticmethod
    def feats_mix(feats, num_layer=5):
        feats = torch.chunk(feats, chunks=num_layer, dim=1)
        feat_sum = None
        for feat in feats:
            feat_sum = feat if feat_sum is None else torch.max(feat_sum, feat)
        return feat_sum

    def forward(self, featmaps, xywh):
        featmap = PANetConstClassifier.feats_mix(featmaps, num_layer=self.num_layer)
        rois_regd = super(PANetConstClassifier, self).forward(featmap, xywh)
        return rois_regd


class PANetConstMasker(MaskRCNNConstMasker):
    def __init__(self, in_channels=64, num_cls=20, num_layer=5, feat_size=(14, 14)):
        super(PANetConstMasker, self).__init__(in_channels=in_channels, num_cls=num_cls, feat_size=feat_size)
        self.num_layer = num_layer

    def forward(self, featmaps):
        featmap = PANetConstClassifier.feats_mix(featmaps, num_layer=self.num_layer)
        mask = super(PANetConstMasker, self).forward(featmap)
        return mask


# </editor-fold>


if __name__ == '__main__':
    model = PANetMutiScaleResNetBkbn.R101(img_size=(512, 512))
    export_pth = '/home/user/JD/Public/export/'
    torch.onnx.export(model, torch.zeros(size=(1, 3, 512, 512)), f=export_pth + 'mskr101_bkbn.onnx')

# if __name__ == '__main__':
#     model = FasterRCNNMLPClassifier(in_channels=256, num_cls=20, inner_features=1024, feat_size=(7, 7))
#     export_pth = '/home/user/JD/Public/export/'
#     torch.onnx.export(model, (torch.zeros(size=(1, 256, 7, 7)), torch.zeros(size=(1, 4))),
#                       f=export_pth + 'mskr101_clfr.onnx', opset_version=11)
