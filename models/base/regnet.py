from models.base.modules import *


class BottleneckSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, neck_ratio=1, group_channels=1, has_se=False, act=ACT.RELU,
                 norm=NORM.BATCH):
        super(BottleneckSE, self).__init__()
        inner_channels = int(out_channels * neck_ratio)
        self.conv1 = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, act=act, norm=norm)
        self.conv2 = Ck3NA(in_channels=inner_channels, out_channels=inner_channels,
                           groups=inner_channels // min(inner_channels, group_channels), stride=stride, act=act,
                           norm=norm)
        self.conv3 = Ck1s1NA(in_channels=inner_channels, out_channels=out_channels, act=None, norm=norm)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = CpaNA(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, act=None, norm=norm)
        else:
            self.shortcut = None
        self.se = SeModule(channels=out_channels, ratio=0.25) if has_se else None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out) if self.se is not None else out
        out = self.act(out + residual)
        return out


class RegNetBkbn(nn.Module):
    def __init__(self, channelss, repeat_nums, group_channels=1, has_se=True, act=ACT.RELU, norm=NORM.BATCH,
                 in_channels=3):
        super(RegNetBkbn, self).__init__()
        self.pre = CpaNA(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, act=act, norm=norm)
        self.stage1 = RegNetBkbn.BottleneckSERepeat(
            in_channels=32, out_channels=channelss[0], repeat_num=repeat_nums[0], stride=2,
            group_channels=group_channels, has_se=has_se, act=act, norm=norm)
        self.stage2 = RegNetBkbn.BottleneckSERepeat(
            in_channels=channelss[0], out_channels=channelss[1], repeat_num=repeat_nums[1], stride=2,
            group_channels=group_channels, has_se=has_se, act=act, norm=norm)
        self.stage3 = RegNetBkbn.BottleneckSERepeat(
            in_channels=channelss[1], out_channels=channelss[2], repeat_num=repeat_nums[2], stride=2,
            group_channels=group_channels, has_se=has_se, act=act, norm=norm)
        self.stage4 = RegNetBkbn.BottleneckSERepeat(
            in_channels=channelss[2], out_channels=channelss[3], repeat_num=repeat_nums[3], stride=2,
            group_channels=group_channels, has_se=has_se, act=act, norm=norm)

    @staticmethod
    def BottleneckSERepeat(in_channels, out_channels, repeat_num=1, stride=1, group_channels=1, has_se=False,
                           neck_ratio=1, act=ACT.RELU, norm=NORM.BATCH):
        backbone = []
        backbone.append(BottleneckSE(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                     group_channels=group_channels, has_se=has_se, neck_ratio=neck_ratio, act=act,
                                     norm=norm))
        for i in range(1, repeat_num):
            backbone.append(BottleneckSE(in_channels=out_channels, out_channels=out_channels, stride=1,
                                         group_channels=group_channels, has_se=has_se, neck_ratio=neck_ratio, act=act,
                                         norm=norm))
        backbone = nn.Sequential(*backbone)
        return backbone

    def forward(self, imgs):
        feat = self.pre(imgs)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        return feat

    X002_PARA = dict(repeat_nums=[1, 1, 4, 7], channelss=[24, 56, 152, 368], group_channels=8, has_se=True)
    X004_PARA = dict(repeat_nums=[1, 2, 7, 12], channelss=[32, 64, 160, 384], group_channels=16, has_se=True)
    X006_PARA = dict(repeat_nums=[1, 3, 5, 7], channelss=[48, 96, 240, 528], group_channels=24, has_se=True)
    X008_PARA = dict(repeat_nums=[1, 3, 7, 5], channelss=[64, 128, 288, 672], group_channels=16, has_se=True)
    X016_PARA = dict(repeat_nums=[2, 4, 10, 2], channelss=[72, 168, 408, 912], group_channels=24, has_se=True)
    X032_PARA = dict(repeat_nums=[2, 6, 15, 2], channelss=[96, 192, 432, 1008], group_channels=48, has_se=True)
    X040_PARA = dict(repeat_nums=[2, 5, 14, 2], channelss=[80, 240, 560, 1360], group_channels=40, has_se=True)
    X064_PARA = dict(repeat_nums=[2, 4, 10, 1], channelss=[168, 392, 784, 1624], group_channels=56, has_se=True)
    X080_PARA = dict(repeat_nums=[2, 5, 15, 1], channelss=[80, 240, 720, 1920], group_channels=120, has_se=True)
    X120_PARA = dict(repeat_nums=[2, 5, 11, 1], channelss=[224, 448, 896, 2240], group_channels=112, has_se=True)
    X160_PARA = dict(repeat_nums=[2, 6, 13, 1], channelss=[256, 512, 896, 2048], group_channels=128, has_se=True)
    X320_PARA = dict(repeat_nums=[2, 7, 13, 1], channelss=[336, 672, 1344, 2520], group_channels=168, has_se=True)


class RegNetMain(RegNetBkbn, ImageONNXExportable):
    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def __init__(self, channelss, repeat_nums, group_channels=1, has_se=True, act=ACT.RELU, norm=NORM.BATCH, num_cls=10,
                 img_size=(224, 224), in_channels=3):
        super(RegNetMain, self).__init__(channelss=channelss, repeat_nums=repeat_nums, group_channels=group_channels,
                                         has_se=has_se, act=act, norm=norm, in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=channelss[3], out_features=num_cls)
        self.num_cls = num_cls
        self._img_size = img_size
        self._in_channels = in_channels

    def forward(self, imgs):
        feat = super(RegNetMain, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        feat = self.linear(feat)
        return feat

    @staticmethod
    def X002(act=ACT.RELU, norm=NORM.BATCH, num_cls=10, img_size=(224, 224), in_channels=3):
        return RegNetMain(**RegNetBkbn.X002_PARA, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                          in_channels=in_channels)

    @staticmethod
    def X004(act=ACT.RELU, norm=NORM.BATCH, num_cls=10, img_size=(224, 224), in_channels=3):
        return RegNetMain(**RegNetBkbn.X004_PARA, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                          in_channels=in_channels)

    @staticmethod
    def X006(act=ACT.RELU, norm=NORM.BATCH, num_cls=10, img_size=(224, 224), in_channels=3):
        return RegNetMain(**RegNetBkbn.X006_PARA, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                          in_channels=in_channels)

    @staticmethod
    def X008(act=ACT.RELU, norm=NORM.BATCH, num_cls=10, img_size=(224, 224), in_channels=3):
        return RegNetMain(**RegNetBkbn.X008_PARA, act=act, norm=norm, num_cls=num_cls, img_size=img_size,
                          in_channels=in_channels)


class RegNet(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=None, img_size=(512, 512), num_cls=10):
        super(RegNet, self).__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def X002(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512), in_channels=3):
        backbone = RegNetMain.X002(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)
        return RegNet(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def X004(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512), in_channels=3):
        backbone = RegNetMain.X004(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)
        return RegNet(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def X006(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512), in_channels=3):
        backbone = RegNetMain.X006(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)
        return RegNet(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def X008(device=None, pack=PACK.AUTO, num_cls=20, img_size=(512, 512), in_channels=3):
        backbone = RegNetMain.X008(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                                   in_channels=in_channels)
        return RegNet(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)


if __name__ == '__main__':
    model = RegNetMain.X006(img_size=(224, 224), num_cls=20)
    model.export_onnx('./buff')
