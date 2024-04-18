from models.base.resnet import np
from models.base.resunet import ResUNetEnc, ImageONNXExportable
from models.modules import *
from models.modules import _pair, _int2, _auto_pad


class GatedCpaBA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bn: bool = True, device=None, dtype=None, act=ACT.NONE):
        super(GatedCpaBA, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_auto_pad(kernel_size, dilation), dilation=_pair(dilation),
            bias=not bn, groups=groups, device=device, dtype=dtype)
        self.gate = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_auto_pad(kernel_size, dilation), dilation=_pair(dilation),
            bias=not bn, groups=groups, device=device, dtype=dtype)
        self.act = ACT.build(act)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x) * torch.sigmoid(self.gate(x))
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


class GatedGenerator(ImageONNXExportable):
    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def __init__(self, in_channels, channels, out_channels, act=ACT.RELU, img_size=(256, 256)):
        super(GatedGenerator, self).__init__()
        self._in_channels = in_channels
        self._img_size = img_size
        self.coarse = nn.Sequential(
            GatedCpaBA(in_channels=in_channels, out_channels=channels, kernel_size=7, stride=1, act=act),
            GatedCpaBA(in_channels=channels, out_channels=channels * 2, kernel_size=3, stride=2, act=act),
            GatedCpaBA(in_channels=channels * 2, out_channels=channels * 4, kernel_size=3, stride=1, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=2, act=act),
            # 变换
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1,
                       dilation=2, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1,
                       dilation=4, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1,
                       dilation=8, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1,
                       dilation=16, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1, act=act),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=1, act=act),
            # 解码
            nn.Upsample(scale_factor=2),
            GatedCpaBA(in_channels=channels * 4, out_channels=channels * 2, kernel_size=3, stride=1, act=act),
            GatedCpaBA(in_channels=channels * 2, out_channels=channels * 2, kernel_size=3, stride=1, act=act),
            nn.Upsample(scale_factor=2),
            GatedCpaBA(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, act=act),
            GatedCpaBA(in_channels=channels, out_channels=out_channels, kernel_size=7, stride=1, act=act),

        )

    def forward(self, imgs_masks):
        return self.coarse(imgs_masks)


# if __name__ == '__main__':
#     model = GatedGenerator(in_channels=4, channels=64, out_channels=3)
#     model.export_onnx('./gen')


class ConvDilations(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 2, 4, 8), act=ACT.RELU):
        super(ConvDilations, self).__init__()
        groups = len(dilations)
        assert out_channels % groups == 0
        self.convs = nn.ModuleList([
            CpaNA(in_channels=in_channels, out_channels=out_channels // groups, kernel_size=kernel_size,
                  stride=stride, dilation=dilation, act=act)
            for dilation in dilations])

    def forward(self, feat):
        output = torch.cat([conv(feat) for conv in self.convs], dim=1)
        return output


class InPResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act=ACT.RELU):
        super(InPResidual, self).__init__()
        self.conv1 = Ck3NA(in_channels=in_channels, out_channels=out_channels,
                           stride=stride, dilation=1, bn=True, act=act)
        self.conv2 = Ck3NA(in_channels=out_channels, out_channels=out_channels,
                           stride=1, dilation=2, bn=True, act=act)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1NA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, bn=True, act=None)
        else:
            self.shortcut = None
        self.gate = Ck1s1(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        gate = torch.sigmoid(self.gate(out1))
        out = self.act(out2 * gate + residual)
        return out


class PatchDiscriminator(ImageONNXExportable):

    def __init__(self, channels, in_channels=4, out_channels=1, act=ACT.RELU, img_size=(256, 256)):
        super(PatchDiscriminator, self).__init__()
        self._img_size = img_size
        self._in_channels = in_channels
        backbone = nn.Sequential(
            CpaNA(in_channels=in_channels, out_channels=channels, kernel_size=7, stride=1, act=act),
            Ck3NA(in_channels=channels, out_channels=channels * 2, stride=2, act=act),
            Ck3NA(in_channels=channels * 2, out_channels=channels * 4, stride=2, act=act),
            Ck3NA(in_channels=channels * 4, out_channels=channels * 4, stride=2, act=act),
            Ck3NA(in_channels=channels * 4, out_channels=channels * 4, stride=2, act=act),
            Ck3NA(in_channels=channels * 4, out_channels=out_channels, stride=2, act=ACT.TANH)
        )
        self.backbone = model_spectral_norm(backbone, show_detial=False)

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def forward(self, imgs_masks):
        return self.backbone(imgs_masks)


# if __name__ == '__main__':
#     model = PatchDiscriminator(channels=64)
#     model.export_onnx('./dis')


class AttnLoc(nn.Module):
    def __init__(self, low_channels, high_channels, qk_channels, out_channels, num_head=8):
        super(AttnLoc, self).__init__()
        self.q = Ck1s1(in_channels=high_channels, out_channels=qk_channels)
        self.k = Ck1s1(in_channels=low_channels, out_channels=qk_channels)
        self.v = Ck1s1(in_channels=low_channels, out_channels=out_channels)
        self.num_head = num_head
        self.out_channels = out_channels
        self.outh_channels = out_channels // num_head
        self.qkh_channels = qk_channels // num_head

    def forward(self, feat_low, feat_high, mask):
        N, _, H, W = feat_low.size()
        feat_high = F.interpolate(feat_high, size=(H, W))
        mask = F.interpolate(mask, size=(H, W))
        k = self.k(feat_low).view(N, self.num_head, self.qkh_channels, H * W)
        v_p = self.v(feat_low)
        v = v_p.view(N, self.num_head, self.outh_channels, H * W).permute(0, 1, 3, 2)
        q = self.q(feat_high).view(N, self.num_head, self.qkh_channels, H * W).permute(0, 1, 3, 2)
        mask_flt = mask.view(N, 1, H * W)

        pows = torch.matmul(q, k) / np.sqrt(self.qkh_channels)
        pows = pows * torch.where((mask_flt[..., None] > 0.5) * (mask_flt[..., None, :] < 0.5), 1, -100)
        pows = torch.softmax(pows, dim=-1)
        val = torch.matmul(pows, v).permute(0, 1, 3, 2)
        val = val.reshape(N, self.out_channels, H, W)
        return torch.where(mask > 0.5, val, v_p)


class ResUNetInPDec(nn.Module):
    def __init__(self, Module, repeat_nums, in_channelss=(32, 64), out_channelss=(32, 64), act=ACT.RELU,
                 strides=(2, 2), flags_attn=(False, True)):
        super(ResUNetInPDec, self).__init__()
        self.stages = nn.ModuleList([])
        self.cvtors = nn.ModuleList([])
        self.attns = nn.ModuleList([])
        self.strides = strides
        for i, repeat_num in enumerate(repeat_nums):
            if i < len(repeat_nums) - 1:
                cvt_channels = in_channelss[i] + out_channelss[i + 1]
                in_channels = out_channelss[i]
                high_channels = out_channelss[i + 1]
            else:
                cvt_channels = in_channelss[i]
                in_channels = in_channelss[i]
                high_channels = in_channelss[i]

            self.cvtors.append(Ck1s1NA(in_channels=cvt_channels, out_channels=out_channelss[i], act=act))
            self.attns.append(AttnLoc(
                low_channels=in_channelss[i], high_channels=high_channels,
                qk_channels=in_channelss[i], out_channels=in_channelss[i]) if flags_attn[i] else None)
            self.stages.append(ResUNetEnc.ModuleRepeat(
                Module, in_channels=in_channels, out_channels=out_channelss[i], stride=1,
                repeat_num=repeat_num, act=act, with_pool=False))
        a = 3

    def forward(self, feats, mask):
        feat = None
        fests_ret = [None] * len(feats)
        for i in range(len(feats) - 1, -1, -1):
            feat_i = feats[i]
            feat_high = feat if i < len(feats) - 1 else feat_i
            if self.attns[i] is not None:
                feat_i = self.attns[i](feat_i, feat_high, mask)
            if i < len(feats) - 1:
                feat = torch.cat([F.interpolate(feat, scale_factor=self.strides[i + 1]), feat_i], dim=1)
            else:
                feat = feat_i
            feat = self.cvtors[i](feat)
            feat = self.stages[i](feat)
            fests_ret[i] = feat
        return fests_ret


class ResUNetInPMain(ImageONNXExportable):
    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, pre_channels, flags_attn,
                 channelss, strides, out_channels=20, img_size=(224, 224), act=ACT.RELU, in_channels=4):
        super(ResUNetInPMain, self).__init__()
        self._img_size = img_size
        self._in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.pre = CpaNA(in_channels=in_channels, out_channels=pre_channels, kernel_size=1, stride=1, bn=True, act=act)
        self.enc = ResUNetEnc(Module, repeat_nums_enc, in_channels=pre_channels, out_channelss=channelss,
                              strides=strides, act=act)
        self.cvt = nn.Sequential(
            ConvDilations(in_channels=channelss[-1], out_channels=channelss[-1], act=act, kernel_size=3),
            ConvDilations(in_channels=channelss[-1], out_channels=channelss[-1], act=act, kernel_size=3)
        )

        self.dec = ResUNetInPDec(Module, repeat_nums_dec, in_channelss=channelss, out_channelss=channelss,
                                 strides=strides, act=act, flags_attn=flags_attn)
        self.out = Ck1s1(in_channels=channelss[0] + pre_channels, out_channels=out_channels)

        self.projs = nn.ModuleList(
            [Ck1s1(in_channels=channels, out_channels=out_channels) for channels in channelss])

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def forward(self, imgs_masks):
        feat = self.pre(imgs_masks)
        masks = imgs_masks[:, 3:4]
        feats = self.enc(feat)
        feats[-1] = self.cvt(feats[-1])
        feats_rec = self.dec(feats, masks)
        feat_rec = feats_rec[0]
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        imgs_rec = self.out(torch.cat([feat, feat_rec], dim=1))
        imgs_rec = torch.sigmoid(imgs_rec)
        # return imgs_rec
        imgs_proj = [torch.sigmoid(proj(feat_rec)) for proj, feat_rec in zip(self.projs, feats_rec)]
        return imgs_rec, imgs_proj

    PARA_LV6 = dict(Module=InPResidual, repeat_nums_enc=[1] * 6, repeat_nums_dec=[1] * 6, pre_channels=32,
                    channelss=(32, 64, 128, 256, 512, 1024), strides=[2] * 6,
                    flags_attn=(False, False, False, True, True, True))
    PARA_LV5 = dict(Module=InPResidual, repeat_nums_enc=[2] * 5, repeat_nums_dec=[2] * 5, pre_channels=32,
                    channelss=(32, 64, 128, 256, 512), strides=[2] * 5, flags_attn=(False, False, False, True, True))
    PARA_R34 = dict(Module=InPResidual, repeat_nums_enc=(2, 3, 4, 6, 3), repeat_nums_dec=[1, 2, 2, 3, 2],
                    channelss=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2), pre_channels=32,
                    flags_attn=(False, False, False, True, True))

    @staticmethod
    def LV6(img_size=(256, 256), out_channels=20, act=ACT.RELU, in_channels=4):
        return ResUNetInPMain(**ResUNetInPMain.PARA_LV6, img_size=img_size, out_channels=out_channels, act=act,
                              in_channels=in_channels)

    @staticmethod
    def LV5(img_size=(256, 256), out_channels=20, act=ACT.RELU, in_channels=4):
        return ResUNetInPMain(**ResUNetInPMain.PARA_LV5, img_size=img_size, out_channels=out_channels, act=act,
                              in_channels=in_channels)

    @staticmethod
    def R34(img_size=(256, 256), out_channels=20, act=ACT.RELU, in_channels=4):
        return ResUNetInPMain(**ResUNetInPMain.PARA_R34, img_size=img_size, out_channels=out_channels, act=act,
                              in_channels=in_channels)


if __name__ == '__main__':
    model = ResUNetInPMain.LV5()
    model.export_onnx('./gen')

# if __name__ == '__main__':
#     layer = AttnLoc(low_channels=32, high_channels=64, qk_channels=32, out_channels=16, num_head=8)
#     feat_low = torch.rand(size=(1, 32, 10, 10))
#     feat_high = torch.rand(size=(1, 64, 5, 5))
#     mask = torch.rand((1, 1, 20, 20))
#     y = layer(feat_low, feat_high, mask)
#     print(y.size())
