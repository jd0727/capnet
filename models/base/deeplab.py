from models.base import ResNetBkbn, VGGBkbn
from models.modules import *
from models.template import OneStageSegmentor
from utils.frame import *


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 3, 5), with_conv1=False):
        super(ASPP, self).__init__()
        convs = []
        for i, dilation in enumerate(dilations):
            if with_conv1 and dilation == 1:
                convs.append(Ck1s1(in_channels=in_channels, out_channels=out_channels))
            else:
                convs.append(Ck3(in_channels=in_channels, out_channels=out_channels, dilation=dilation))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        return sum([conv(x) for conv in self.convs])


class DeepLabV2ResNetMain(ImageONNXExportable):
    DILATIONS = (1, 6, 12, 18)

    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224),
                 in_channels=3):
        nn.Module.__init__(self)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self.pre = CpaNA(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, act=act, norm=norm)
        self.stage1 = ResNetBkbn.ModuleRepeat(Module, in_channels=64, out_channels=channels, stride=1, dilation=1,
                                              repeat_num=repeat_nums[0], act=act, norm=norm, with_pool=False)
        self.stage2 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels, out_channels=channels * 2, stride=2,
                                              dilation=1, repeat_num=repeat_nums[1], act=act, norm=norm,
                                              with_pool=False)
        self.stage3 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 2, out_channels=channels * 4, stride=1,
                                              dilation=2, repeat_num=repeat_nums[2], act=act, norm=norm,
                                              with_pool=False)
        self.stage4 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 4, out_channels=channels * 8, stride=1,
                                              dilation=4, repeat_num=repeat_nums[3], act=act, norm=norm,
                                              with_pool=False)
        self.aspp = ASPP(in_channels=channels * 8, out_channels=num_cls,
                         dilations=DeepLabV2ResNetMain.DILATIONS, with_conv1=False)

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=1, dilation=1, act=ACT.RELU,
                     norm=NORM.BATCH):
        backbone = []
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(Module(in_channels=last_channels, out_channels=out_channels,
                                   dilation=dilation, stride=stride, act=act, norm=norm))
            last_channels = out_channels
            stride = 1
            dilation = 1
        backbone = nn.Sequential(*backbone)
        return backbone

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats4 = self.aspp(feats4)
        feats4 = F.interpolate(feats4, scale_factor=4, mode='bilinear', align_corners=False)
        return feats4

    @staticmethod
    def R18(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV2ResNetMain(**ResNetBkbn.PARA_R18, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R34(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV2ResNetMain(**ResNetBkbn.PARA_R34, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R50(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV2ResNetMain(**ResNetBkbn.PARA_R50, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R101(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV2ResNetMain(**ResNetBkbn.PARA_R101, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R152(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV2ResNetMain(**ResNetBkbn.PARA_R152, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)


class ASPPParal(ASPP):
    def __init__(self, in_channels, part_channels, dilations=(1, 3, 5), with_conv1=False):
        super(ASPPParal, self).__init__(in_channels, part_channels, dilations=dilations, with_conv1=with_conv1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = Ck1s1NA(in_channels=in_channels, out_channels=part_channels)

    def forward(self, x):
        feats_gol = self.linear(self.pool(x)).repeat(1, 1, x.size(2), x.size(3))
        return torch.cat([conv(x) for conv in self.convs] + [feats_gol], dim=1)


class DeepLabV3ResNetMain(nn.Module):
    DILATIONS = (1, 6, 12, 18)

    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224),
                 in_channels=3):
        super(DeepLabV3ResNetMain, self).__init__()
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self.pre = CpaNA(in_channels=3, out_channels=64, kernel_size=7, stride=2, act=act, norm=norm)
        self.stage1 = ResNetBkbn.ModuleRepeat(Module, in_channels=64, out_channels=channels,
                                              stride=1, dilation=1, repeat_num=repeat_nums[0], act=act, norm=norm,
                                              with_pool=True)
        self.stage2 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels, out_channels=channels * 2,
                                              stride=2, dilation=1, repeat_num=repeat_nums[1], act=act, norm=norm,
                                              with_pool=False)
        self.stage3 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 2, out_channels=channels * 4,
                                              stride=1, dilation=2, repeat_num=repeat_nums[2], act=act, norm=norm,
                                              with_pool=False)
        self.stage4 = ResNetBkbn.ModuleRepeat(Module, in_channels=channels * 4, out_channels=channels * 8,
                                              stride=1, dilation=4, repeat_num=repeat_nums[3], act=act, norm=norm,
                                              with_pool=False)
        self.aspp = ASPPParal(in_channels=channels * 8, part_channels=channels,
                              dilations=DeepLabV2ResNetMain.DILATIONS, with_conv1=True)

        self.conv5 = Ck1s1NA(in_channels=channels * 5, out_channels=channels, act=act, norm=norm)
        self.sampler2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv5_1 = Ck1s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)

        self.stage6 = nn.Sequential(
            Ck3s1NA(in_channels=channels * 2, out_channels=channels, act=act, norm=norm),
            Ck3s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm),
            Ck1(in_channels=channels, out_channels=num_cls),
        )
        self.sampler4 = nn.UpsamplingBilinear2d(scale_factor=4)

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)

        feats4 = self.aspp(feats4)
        feats5 = self.conv5(feats4)
        feats5 = self.sampler2(feats5)

        feats5_1 = self.conv5_1(feats1)

        feats5 = torch.cat([feats5, feats5_1], dim=1)
        feats6 = self.stage6(feats5)

        feats6 = self.sampler4(feats6)
        return feats6

    @staticmethod
    def R18(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV3ResNetMain(**ResNetBkbn.PARA_R18, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R34(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV3ResNetMain(**ResNetBkbn.PARA_R34, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R50(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV3ResNetMain(**ResNetBkbn.PARA_R50, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R101(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV3ResNetMain(**ResNetBkbn.PARA_R101, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)

    @staticmethod
    def R152(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3):
        return DeepLabV3ResNetMain(**ResNetBkbn.PARA_R152, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)


class DeepLabConstMain(nn.Module):
    def __init__(self, batch_size=1, num_cls=20, img_size=(224, 224)):
        super().__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.featmaps = nn.Parameter(torch.zeros(batch_size, num_cls, img_size[1], img_size[0]))
        init_sig(self.featmaps, prior_prob=0.01)

    def forward(self, imgs):
        return torch.sigmoid(self.featmaps)


class DeepLabV1VGGMain(ImageONNXExportable):
    def __init__(self, repeat_nums=(1, 1, 2, 2, 2), act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224),
                 in_channels=3):
        super(DeepLabV1VGGMain, self).__init__()
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels
        self.stage1 = DeepLabV1VGGMain.Ck3Repeat(in_channels=in_channels, out_channels=64, dilation=1,
                                                 repeat_num=repeat_nums[0], act=act, norm=norm, pool_stride=0)
        self.stage2 = DeepLabV1VGGMain.Ck3Repeat(in_channels=64, out_channels=128, dilation=1,
                                                 repeat_num=repeat_nums[1], act=act, norm=norm, pool_stride=2)
        self.stage3 = DeepLabV1VGGMain.Ck3Repeat(in_channels=128, out_channels=256, dilation=1,
                                                 repeat_num=repeat_nums[2], act=act, norm=norm, pool_stride=2)
        self.stage4 = DeepLabV1VGGMain.Ck3Repeat(in_channels=256, out_channels=512, dilation=1,
                                                 repeat_num=repeat_nums[3], act=act, norm=norm, pool_stride=1)
        self.stage5 = DeepLabV1VGGMain.Ck3Repeat(in_channels=512, out_channels=512, dilation=2,
                                                 repeat_num=repeat_nums[4], act=act, norm=norm, pool_stride=1)
        self.stage6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Ck3NA(in_channels=512, out_channels=1024, dilation=12, act=act, norm=norm),
            nn.Dropout2d(0.5),
            Ck1s1NA(in_channels=1024, out_channels=1024, act=act, norm=norm),
            nn.Dropout2d(0.5),
            Ck1s1NA(in_channels=1024, out_channels=num_cls, act=None),
        )

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    @staticmethod
    def Ck3Repeat(in_channels, out_channels, repeat_num=1, pool_stride=1, dilation=1, act=ACT.RELU, norm=NORM.BATCH):
        backbone = nn.Sequential()
        if pool_stride > 0:
            backbone.add_module('pool', nn.MaxPool2d(
                kernel_size=3, stride=pool_stride, padding=1, ceil_mode=False))
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.add_module(
                str(i),
                Ck3NA(in_channels=last_channels, out_channels=out_channels, stride=1, dilation=dilation,
                      act=act, norm=norm))
            last_channels = out_channels
        return backbone

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        feats6 = self.stage6(feats5)
        feats6 = F.interpolate(feats6, scale_factor=4, mode='bilinear', align_corners=False)
        return feats6

    @staticmethod
    def A(num_cls=20, img_size=(224, 224), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DeepLabV1VGGMain(**VGGBkbn.PARA_A, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                in_channels=in_channels)

    @staticmethod
    def B(num_cls=20, img_size=(224, 224), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DeepLabV1VGGMain(**VGGBkbn.PARA_B, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                in_channels=in_channels)

    @staticmethod
    def D(num_cls=20, img_size=(224, 224), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DeepLabV1VGGMain(**VGGBkbn.PARA_D, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                in_channels=in_channels)

    @staticmethod
    def E(num_cls=20, img_size=(224, 224), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DeepLabV1VGGMain(**VGGBkbn.PARA_E, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                                in_channels=in_channels)


class DeepLab(OneStageSegmentor):
    @staticmethod
    def V1A(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV1VGGMain.A(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                      in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V1D(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV1VGGMain.D(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                      in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V2R18(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV2ResNetMain.R18(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                           in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V2R34(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV2ResNetMain.R34(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                           in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V2R50(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV2ResNetMain.R50(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                           in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V2R101(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV3ResNetMain.R101(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                            in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V3R18(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV3ResNetMain.R18(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                           in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V3R34(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV3ResNetMain.R34(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                           in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V3R50(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV3ResNetMain.R50(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                           in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V3R101(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabV3ResNetMain.R101(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                            in_channels=in_channels)
        return DeepLab(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def Const(device=None, batch_size=1, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = DeepLabConstMain(batch_size=batch_size, num_cls=num_cls, img_size=img_size)
        return DeepLab(backbone=backbone, device=device, pack=PACK.NONE, num_cls=num_cls, img_size=img_size)


if __name__ == '__main__':
    model = DeepLabV1VGGMain.A(img_size=(224, 224), num_cls=20)
    model.export_onnx('./buff')
