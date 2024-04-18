from torch.cuda.amp import autocast

from models.base.darknet import SPP
from models.base.modules import SeModule, CpaBASelfAttentionMutiHead
from models.modules import *
from models.template import CategoryWeightAdapter
from models.yolo.modules import *


class DarkNetResidualV8(nn.Module):
    def __init__(self, channels, act=ACT.LK, norm=NORM.BATCH):
        super(DarkNetResidualV8, self).__init__()
        self.conv1 = Ck3s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)
        self.conv2 = Ck3s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)

    def forward(self, x):
        x = x + self.conv2(self.conv1(x))
        return x


class DarkNetResidualV8SE(nn.Module):
    def __init__(self, channels, act=ACT.LK, norm=NORM.BATCH):
        super(DarkNetResidualV8SE, self).__init__()
        self.conv1 = Ck3s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)
        self.conv2 = Ck3s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)
        self.se = SeModule(channels=channels)

    def forward(self, x):
        x = x + self.se(self.conv2(self.conv1(x)))
        return x


class CSPBlockV8(nn.Module):
    def __init__(self, Module, in_channels, out_channels, repeat_num, act=ACT.LK, norm=NORM.BATCH):
        super(CSPBlockV8, self).__init__()

        self.inter = Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
        inner_channels = out_channels // 2
        self.inner_channels = inner_channels
        self.backbone = nn.ModuleList([])
        for i in range(repeat_num):
            self.backbone.append(Module(channels=inner_channels, act=act, norm=norm))
        self.outer = Ck1s1NA(in_channels=inner_channels * (repeat_num + 2), out_channels=out_channels, act=act,
                             norm=norm)

    def forward(self, x):
        x = self.inter(x)
        _, buffer = x.split(self.inner_channels, dim=1)
        xs = [x]
        for module in self.backbone:
            buffer = module(buffer)
            xs.append(buffer)
        xs = torch.cat(xs, dim=1)
        xs = self.outer(xs)
        return xs


class DarkNetV8Bkbn(nn.Module):
    def __init__(self, Module, channelss, repeat_nums, act=ACT.SILU, norm=NORM.BATCH, in_channels=3):
        super(DarkNetV8Bkbn, self).__init__()
        pre_channels = channelss[0] // 2
        self.pre = Ck3NA(in_channels=in_channels, out_channels=pre_channels, stride=2, act=act, norm=norm)
        self.stage1 = DarkNetV8Bkbn.ModuleRepeat(Module=Module, in_channels=pre_channels, out_channels=channelss[0],
                                                 repeat_num=repeat_nums[0], stride=2, act=act, norm=norm)
        self.stage2 = DarkNetV8Bkbn.ModuleRepeat(Module=Module, in_channels=channelss[0], out_channels=channelss[1],
                                                 repeat_num=repeat_nums[1], stride=2, act=act, norm=norm)
        self.stage3 = DarkNetV8Bkbn.ModuleRepeat(Module=Module, in_channels=channelss[1], out_channels=channelss[2],
                                                 repeat_num=repeat_nums[2], stride=2, act=act, norm=norm)
        self.stage4 = DarkNetV8Bkbn.ModuleRepeat(Module=Module, in_channels=channelss[2], out_channels=channelss[3],
                                                 repeat_num=repeat_nums[3], stride=2, act=act, norm=norm)

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK, norm=NORM.BATCH):
        convs = nn.Sequential(
            Ck3NA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act, norm=norm),
            CSPBlockV8(Module=Module, in_channels=out_channels, out_channels=out_channels,
                       repeat_num=repeat_num, act=act, norm=norm))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        return feats4

    PARA_NANO = dict(Module=DarkNetResidualV8, channelss=(32, 64, 128, 256), repeat_nums=(1, 2, 2, 1))
    PARA_SMALL = dict(Module=DarkNetResidualV8, channelss=(64, 128, 256, 512), repeat_nums=(1, 2, 2, 1))
    PARA_MEDIUM = dict(Module=DarkNetResidualV8, channelss=(96, 192, 384, 576), repeat_nums=(2, 4, 4, 2))
    PARA_LARGE = dict(Module=DarkNetResidualV8, channelss=(128, 256, 512, 512), repeat_nums=(3, 6, 6, 3))
    PARA_XLARGE = dict(Module=DarkNetResidualV8, channelss=(160, 320, 640, 640), repeat_nums=(4, 8, 8, 4))

    PARA_NANO_SE = dict(Module=DarkNetResidualV8SE, channelss=(32, 64, 128, 256), repeat_nums=(1, 2, 2, 1))
    PARA_SMALL_SE = dict(Module=DarkNetResidualV8SE, channelss=(64, 128, 256, 512), repeat_nums=(1, 2, 2, 1))
    PARA_MEDIUM_SE = dict(Module=DarkNetResidualV8SE, channelss=(96, 192, 384, 576), repeat_nums=(2, 4, 4, 2))
    PARA_LARGE_SE = dict(Module=DarkNetResidualV8SE, channelss=(128, 256, 512, 512), repeat_nums=(3, 6, 6, 3))
    PARA_XLARGE_SE = dict(Module=DarkNetResidualV8SE, channelss=(160, 320, 640, 640), repeat_nums=(4, 8, 8, 4))

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetV8Bkbn(**DarkNetV8Bkbn.PARA_NANO, act=act, norm=norm)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetV8Bkbn(**DarkNetV8Bkbn.PARA_SMALL, act=act, norm=norm)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetV8Bkbn(**DarkNetV8Bkbn.PARA_MEDIUM, act=act, norm=norm)

    @staticmethod
    def Large(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetV8Bkbn(**DarkNetV8Bkbn.PARA_LARGE, act=act, norm=norm)

    @staticmethod
    def XLarge(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetV8Bkbn(**DarkNetV8Bkbn.PARA_XLARGE, act=act, norm=norm)


class YoloV8DownStream(nn.Module):
    def __init__(self, Module, in_channelss, out_channelss, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV8DownStream, self).__init__()
        self.mixrs = nn.ModuleList()
        for i in range(len(in_channelss)):
            out_channels = out_channelss[i]
            in_channels = in_channelss[i]
            if i == len(in_channelss) - 1:
                mixr = nn.Identity() if in_channels == out_channels else \
                    Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
            else:
                last_channels = in_channels + out_channelss[i + 1]
                mixr = CSPBlockV8(Module=Module, in_channels=last_channels, out_channels=out_channels,
                                  repeat_num=repeat_num, act=act, norm=norm)
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


class YoloV8UpStream(nn.Module):
    def __init__(self, Module, in_channelss, out_channelss, repeat_num=1, act=ACT.LK, norm=NORM.BATCH):
        super(YoloV8UpStream, self).__init__()
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
                mixr = CSPBlockV8(Module=Module, in_channels=out_channels_pre + in_channels, out_channels=out_channels,
                                  repeat_num=repeat_num, act=act, norm=norm)
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


class YoloV8ConstLayer(PointAnchorImgLayer):
    def __init__(self, batch_size, stride, num_cls, num_dstr=4, img_size=(512, 512)):
        super().__init__(stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.num_dstr = num_dstr
        self.reg_lrtd = nn.Parameter(torch.zeros(batch_size, 4 * num_dstr, self.Hf, self.Wf))
        self.reg_chot = nn.Parameter(torch.zeros(batch_size, num_cls, self.Hf, self.Wf))
        init_sig(self.reg_chot, prior_prob=0.01)

    def forward(self, featmap):
        pred = YoloV8Layer.decode(
            reg_lrtd=self.reg_lrtd, reg_chot=self.reg_chot, xy_offset=self.xy_offset, num_dstr=self.num_dstr,
            stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloV8ConstMain(nn.Module):
    def __init__(self, num_cls=80, img_size=(416, 352), batch_size=3, num_dstr=4):
        super(YoloV8ConstMain, self).__init__()
        self.num_cls = num_cls
        self.num_dstr = num_dstr
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloV8ConstLayer(batch_size=batch_size, stride=8, num_dstr=self.num_dstr,
                             num_cls=num_cls, img_size=img_size),
            YoloV8ConstLayer(batch_size=batch_size, stride=16, num_dstr=self.num_dstr,
                             num_cls=num_cls, img_size=img_size),
            YoloV8ConstLayer(batch_size=batch_size, stride=32, num_dstr=self.num_dstr,
                             num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloV8Layer(PointAnchorImgLayer):
    def __init__(self, in_channels, inner_cahnnels, stride, num_cls, num_dstr=4, img_size=(512, 512),
                 act=ACT.SILU, norm=NORM.BATCH, ):
        super().__init__(stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.num_dstr = num_dstr
        reg_channels = 4 * num_dstr
        self.reg_lrtd = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=reg_channels, act=act, norm=norm),
            Ck3s1NA(in_channels=reg_channels, out_channels=reg_channels, act=act, norm=norm),
            Ck1s1(in_channels=reg_channels, out_channels=reg_channels),
        )
        self.reg_chot = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=inner_cahnnels, act=act, norm=norm),
            Ck3s1NA(in_channels=inner_cahnnels, out_channels=inner_cahnnels, act=act, norm=norm),
            Ck1s1(in_channels=inner_cahnnels, out_channels=num_cls),
        )
        init_sig(self.reg_chot[-1].conv.bias, prior_prob=0.001)

    def forward(self, featmap):
        reg_chot = self.reg_chot(featmap)
        reg_lrtd = self.reg_lrtd(featmap)
        pred = YoloV8Layer.decode(
            reg_chot=reg_chot, reg_lrtd=reg_lrtd, xy_offset=self.xy_offset,
            stride=self.stride, num_cls=self.num_cls, num_dstr=self.num_dstr)
        return pred

    @staticmethod
    def decode(reg_lrtd, reg_chot, xy_offset, stride, num_cls, num_dstr):
        xy_offset = xy_offset.to(reg_lrtd.device, non_blocking=True)
        Hf, Wf, _ = list(xy_offset.size())
        reg_lrtd = reg_lrtd.permute(0, 2, 3, 1)
        reg_chot = reg_chot.permute(0, 2, 3, 1)
        cens = ((xy_offset + 0.5) * stride).expand(reg_lrtd.size(0), Hf, Wf, 2)
        strides = torch.full(size=(reg_lrtd.size(0), Hf, Wf, 1), fill_value=stride, device=reg_lrtd.device)
        chot = torch.sigmoid(reg_chot)
        pred = torch.cat([cens, strides, reg_lrtd, chot], dim=-1).contiguous()
        pred = pred.reshape(-1, Wf * Hf, 3 + num_dstr * 4 + num_cls)
        return pred


def lrtds_dstr2xyxys(cens, lrtds_dstr, strides):
    lrtd = dlsT_dstr2dlsT(lrtds_dstr) * strides
    xyxys = torch.cat([cens - lrtd[..., :2], cens + lrtd[..., 2:4]], dim=-1)
    return xyxys


class YoloV8Main(DarkNetV8Bkbn, ImageONNXExportable):

    def __init__(self, Module, channelss, repeat_nums, num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256),
                 in_channels=3, num_dstr=4):
        DarkNetV8Bkbn.__init__(self, Module, channelss, repeat_nums, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self._num_dstr = num_dstr
        feat_channelss = channelss[1:]
        down_channelss = channelss[1:]
        up_channelss = channelss[1:]
        self.spp = YoloV8Main.C1C3RepeatSPP(in_channels=channelss[-1], out_channels=channelss[-1], act=act, norm=norm)
        self.down = YoloV8DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act,
                                     norm=norm)
        self.up = YoloV8UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act, norm=norm)
        self.layers = nn.ModuleList([
            YoloV8Layer(in_channels=channelss[1], inner_cahnnels=channelss[1], stride=8,
                        num_cls=num_cls, img_size=img_size, num_dstr=num_dstr),
            YoloV8Layer(in_channels=channelss[2], inner_cahnnels=channelss[1], stride=16,
                        num_cls=num_cls, img_size=img_size, num_dstr=num_dstr),
            YoloV8Layer(in_channels=channelss[3], inner_cahnnels=channelss[1], stride=32,
                        num_cls=num_cls, img_size=img_size, num_dstr=num_dstr)
        ])

    @staticmethod
    def C1C3RepeatSPP(in_channels, out_channels, act=ACT.LK, norm=NORM.BATCH):
        convs = [
            Ck1s1NA(in_channels=in_channels, out_channels=in_channels // 2, act=act, norm=norm),
            SPP(kernels=(13, 9, 5), stride=1, shortcut=True),
            Ck1s1NA(in_channels=in_channels * 2, out_channels=out_channels, act=act, norm=norm),
        ]
        return nn.Sequential(*convs)

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
        feats4 = self.spp(feats4)
        feats = (feats2, feats3, feats4)
        feats = self.down(feats)
        feats = self.up(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8Main(**DarkNetV8Bkbn.PARA_NANO, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8Main(**DarkNetV8Bkbn.PARA_SMALL, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8Main(**DarkNetV8Bkbn.PARA_MEDIUM, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8Main(**DarkNetV8Bkbn.PARA_LARGE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8Main(**DarkNetV8Bkbn.PARA_XLARGE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                          in_channels=in_channels, num_dstr=num_dstr)


class YoloV8ExtMain(DarkNetV8Bkbn, ImageONNXExportable):

    def __init__(self, Module, channelss, repeat_nums, num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256),
                 in_channels=3, num_dstr=4):
        DarkNetV8Bkbn.__init__(self, Module, channelss, repeat_nums, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self._num_dstr = num_dstr
        feat_channelss = channelss[1:]
        down_channelss = channelss[1:]
        up_channelss = channelss[1:]
        self.spp = CpaBASelfAttentionMutiHead(in_channels=channelss[-1], out_channels=channelss[-1],
                                              qk_channels=channelss[-1], act=act, norm=norm)
        self.down = YoloV8DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act,
                                     norm=norm)
        self.up = YoloV8UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act, norm=norm)
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
        feats4 = self.spp(feats4)
        feats = (feats2, feats3, feats4)
        feats = self.down(feats)
        feats = self.up(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8ExtMain(**DarkNetV8Bkbn.PARA_NANO_SE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8ExtMain(**DarkNetV8Bkbn.PARA_SMALL_SE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8ExtMain(**DarkNetV8Bkbn.PARA_MEDIUM_SE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8ExtMain(**DarkNetV8Bkbn.PARA_LARGE_SE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             in_channels=in_channels, num_dstr=num_dstr)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.SILU, norm=NORM.BATCH, img_size=(256, 256), in_channels=3, num_dstr=16):
        return YoloV8ExtMain(**DarkNetV8Bkbn.PARA_XLARGE_SE, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                             in_channels=in_channels, num_dstr=num_dstr)


class YoloV8(OneStageTorchModel, IndependentInferableModel, CategoryWeightAdapter):
    def __init__(self, backbone, device=None, pack=PACK.AUTO, **kwargs):
        OneStageTorchModel.__init__(self, backbone=backbone, device=device, pack=pack)
        CategoryWeightAdapter.__init__(self)
        self.layers = backbone.layers
        self.alpha = 1
        self.beta = 6
        self.max_match = 10

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def num_dstr(self):
        return self.backbone.num_dstr

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

    @property
    def cens(self):
        return torch.cat([layer.cens for layer in self.backbone.layers], dim=0)

    @property
    def strides(self):
        return torch.Tensor([layer.stride for layer in self.backbone.layers])

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.5, iou_thres=0.4, by_cls=True, num_presv=3000,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        preds = self.pkd_modules['backbone'](imgsT)
        censsT, stridessT, lrtdss_dstrT, chotssT = preds.split((2, 1, self.num_dstr * 4, self.num_cls), dim=-1)
        lrtdss_dstrT = lrtdss_dstrT.reshape(preds.size(0), preds.size(1), 4, self.num_dstr)
        xyxyssT = lrtds_dstr2xyxys(censsT, lrtds_dstr=lrtdss_dstrT, strides=stridessT)
        confssT, cindssT = torch.max(chotssT, dim=-1)
        xyxyssT = xyxysT_clip(xyxyssT, xyxyN_rgn=np.array(self.img_size))
        labels = []
        for xyxysT, confsT, cindsT in zip(xyxyssT, confssT, cindssT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
            prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=cindsT if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type, num_presv=num_presv)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
            boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name, img_size=self.img_size,
                num_cls=self.num_cls)
            labels.append(boxes)
        return labels_rescale(labels, imgs2img_sizes(imgs), 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_lb = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        xyxys_tg = [np.zeros(shape=(0, 4))]
        cinds_tg = [np.zeros(shape=0, dtype=np.int32)]

        for i, label in enumerate(labels):
            xyxys = label.export_xyxysN()
            cinds = label.export_cindsN()
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                stride, Wf, Hf, num_anchor = layer.stride, layer.Wf, layer.Hf, layer.num_anchor
                xy_offset = layer.xy_offset.numpy()
                xy_offset = (xy_offset + 0.5).reshape(-1, 2) * stride
                fltr_in = np.all((xyxys[:, None, :2] < xy_offset) * (xyxys[:, None, 2:4] > xy_offset), axis=-1)
                ids_lb, ids_ancr = np.nonzero(fltr_in)

                inds_b_pos.append(np.full(fill_value=i, shape=len(ids_lb)))
                inds_layer.append(np.full(fill_value=j, shape=len(ids_lb)))
                inds_pos.append(offset_layer + ids_ancr)
                cinds_tg.append(cinds[ids_lb])
                xyxys_tg.append(xyxys[ids_lb])
                inds_lb.append(ids_lb)

                offset_layer = offset_layer + num_anchor

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        xyxys_tg = np.concatenate(xyxys_tg, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        inds_lb = np.concatenate(inds_lb, axis=0)

        targets = (inds_b_pos, inds_pos, xyxys_tg, cinds_tg, inds_layer, inds_lb)
        return targets

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        inds_b_pos, inds_pos, xyxys_tg, cinds_tg, inds_layer, inds_lb = arrsN2arrsT(targets, device=self.device)
        chots_pd = preds[..., (3 + self.num_dstr * 4):]
        chots_tg = torch.zeros_like(chots_pd, device=self.device)
        if inds_b_pos.size(0) > 0:
            cens, strides, lrtds_dstr, chots_pd_pos = preds[inds_b_pos, inds_pos]. \
                split((2, 1, self.num_dstr * 4, self.num_cls), dim=-1)
            lrtds_dstr = lrtds_dstr.reshape(inds_b_pos.size(0), 4, self.num_dstr)
            xyxys_pd = lrtds_dstr2xyxys(cens, lrtds_dstr=lrtds_dstr, strides=strides)

            with torch.no_grad():
                lrtds_tg = torch.cat([cens - xyxys_tg[..., :2], xyxys_tg[..., 2:4] - cens], dim=-1) / strides
                fltr_valid = torch.all(lrtds_tg < self.num_dstr - 1, dim=-1)
                # 动态分配
                ious = ropr_arr_xyxysT(xyxys_pd, xyxys_tg, opr_type=IOU_TYPE.DIOU)
                confs = torch.gather(chots_pd_pos, index=cinds_tg[..., None], dim=-1)[..., 0]
                scores = (confs.detach() ** self.alpha) * (ious ** self.beta) * fltr_valid
                max_lb = torch.max(inds_lb).item() + 1
                buffer = torch.zeros(size=(imgs.size(0), max_lb, self.num_anchor), device=self.device)
                buffer[inds_b_pos, inds_lb, inds_pos] = scores.detach() * fltr_valid
                buffer_aligend = torch.topk(buffer, dim=-1, k=self.max_match)[0][inds_b_pos, inds_lb]
                fltr_presv = (buffer_aligend[:, -1] <= scores) * (scores > 0)
                scores_normd = scores / buffer_aligend[:, 0].clamp(min=1e-5)

            inds_b_pos, inds_pos, lrtds_dstr, lrtds_tg, xyxys_pd, xyxys_tg, cinds_tg, scores_normd, ious = \
                inds_b_pos[fltr_presv], inds_pos[fltr_presv], lrtds_dstr[fltr_presv], lrtds_tg[fltr_presv], \
                xyxys_pd[fltr_presv], xyxys_tg[fltr_presv], cinds_tg[fltr_presv], scores_normd[fltr_presv], \
                ious[fltr_presv]
            # DFL损失
            dfl_loss = distribute_loss(lrtds_dstr.reshape(-1, self.num_dstr), lrtds_tg.view(-1), reduction='mean')
            chots_tg[inds_b_pos, inds_pos, cinds_tg] = scores_normd.detach()
            iou_loss = 1 - torch.mean(ious)
        else:
            iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
            dfl_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        # 分类损失
        with autocast(enabled=False):
            weight_cls = self.get_weight_cls(chots_tg)
            cls_loss = F.binary_cross_entropy(chots_pd, chots_tg, weight=weight_cls, reduction='sum')
            cls_loss = cls_loss / max(1, inds_pos.size(0))
        return OrderedDict(cls=cls_loss, iou=iou_loss * 5, dfl=dfl_loss)

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_dstr=32):
        backbone = YoloV8Main.Nano(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size, num_dstr=num_dstr)
        return YoloV8(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_dstr=32):
        backbone = YoloV8Main.Small(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size,
                                    num_dstr=num_dstr)
        return YoloV8(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_dstr=32):
        backbone = YoloV8Main.Medium(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size,
                                     num_dstr=num_dstr)
        return YoloV8(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_dstr=32):
        backbone = YoloV8Main.Large(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size,
                                    num_dstr=num_dstr)
        return YoloV8(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def XLarge(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, num_dstr=32):
        backbone = YoloV8Main.XLarge(num_cls=num_cls, act=ACT.SILU, norm=NORM.BATCH, img_size=img_size,
                                     num_dstr=num_dstr)
        return YoloV8(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1, num_dstr=32):
        backbone = YoloV8ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size, num_dstr=num_dstr)
        return YoloV8(backbone=backbone, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = YoloV8ExtMain.Medium(img_size=(640, 640), num_cls=80)
    model.export_onnx('./buff')
