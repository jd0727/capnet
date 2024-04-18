from models.base.modules import *


# CBA+CBA+Res
class DarkNetResidual(nn.Module):
    def __init__(self, channels, inner_channels, act=ACT.LK, norm=NORM.BATCH):
        super(DarkNetResidual, self).__init__()
        self.conv1 = Ck1s1NA(in_channels=channels, out_channels=inner_channels, act=act, norm=norm)
        self.conv2 = Ck3s1NA(in_channels=inner_channels, out_channels=channels, act=act, norm=norm)

    def forward(self, x):
        x = x + self.conv2(self.conv1(x))
        return x


class DarkNetResidualSE(nn.Module):
    def __init__(self, channels, inner_channels, act=ACT.LK, norm=NORM.BATCH):
        super(DarkNetResidualSE, self).__init__()
        self.conv1 = Ck1s1NA(in_channels=channels, out_channels=inner_channels, act=act, norm=norm)
        self.se = SeModule(channels=inner_channels, )
        self.conv2 = Ck3s1NA(in_channels=inner_channels, out_channels=channels, act=act, norm=norm)

    def forward(self, x):
        x = x + self.conv2(self.se(self.conv1(x)))
        return x


class DarkNetTinyBkbn(nn.Module):
    def __init__(self, channels, act=ACT.LK, norm=NORM.BATCH, in_channels=3):
        super(DarkNetTinyBkbn, self).__init__()
        self.stage1 = nn.Sequential(
            Ck3s1NA(in_channels=in_channels, out_channels=channels, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Ck3s1NA(in_channels=channels, out_channels=channels * 2, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarkNetTinyBkbn.C3C1Repeat(in_channels=channels * 2, out_channels=channels * 4, repeat_num=3, act=act,
                                       norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarkNetTinyBkbn.C3C1Repeat(in_channels=channels * 4, out_channels=channels * 8, repeat_num=3, act=act,
                                       norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarkNetTinyBkbn.C3C1Repeat(in_channels=channels * 8, out_channels=channels * 16, repeat_num=5, act=act,
                                       norm=norm),
        )
        self.stage2 = DarkNetTinyBkbn.C3C1Repeat(
            in_channels=channels * 16, out_channels=channels * 32, repeat_num=5, act=act, norm=norm)
        self.stage3 = nn.Sequential(
            Ck1NA(in_channels=channels * 32, out_channels=channels * 32, act=act, norm=norm),
            Ck1NA(in_channels=channels * 32, out_channels=channels * 32, act=act, norm=norm)
        )

    @staticmethod
    def C3C1Repeat(in_channels, out_channels, repeat_num, act=ACT.LK, norm=NORM.BATCH):
        inner_channels = out_channels // 2
        convs = [Ck3s1NA(in_channels, out_channels, act=act, norm=norm)]
        for i in range((repeat_num - 1) // 2):
            convs.append(Ck1s1NA(out_channels, inner_channels, act=act, norm=norm))
            convs.append(Ck3s1NA(inner_channels, out_channels, act=act, norm=norm))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        return feats1, feats2, feats3


class DarkNetBkbn(nn.Module):
    def __init__(self, Module, channels, repeat_nums, act=ACT.LK, norm=NORM.BATCH, in_channels=3):
        super(DarkNetBkbn, self).__init__()
        self.pre = Ck3s1NA(in_channels=in_channels, out_channels=channels, act=act, norm=norm)
        self.stage1 = DarkNetBkbn.ModuleRepeat(Module=Module, in_channels=channels, out_channels=channels * 2,
                                               repeat_num=repeat_nums[0], stride=2, act=act, norm=norm)
        self.stage2 = DarkNetBkbn.ModuleRepeat(Module=Module, in_channels=channels * 2, out_channels=channels * 4,
                                               repeat_num=repeat_nums[1], stride=2, act=act, norm=norm)
        self.stage3 = DarkNetBkbn.ModuleRepeat(Module=Module, in_channels=channels * 4, out_channels=channels * 8,
                                               repeat_num=repeat_nums[2], stride=2, act=act, norm=norm)
        self.stage4 = DarkNetBkbn.ModuleRepeat(Module=Module, in_channels=channels * 8, out_channels=channels * 16,
                                               repeat_num=repeat_nums[3], stride=2, act=act, norm=norm)
        self.stage5 = DarkNetBkbn.ModuleRepeat(Module=Module, in_channels=channels * 16, out_channels=channels * 32,
                                               repeat_num=repeat_nums[4], stride=2, act=act, norm=norm)

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK, norm=NORM.BATCH):
        convs = [Ck3NA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act, norm=norm)]
        for i in range(repeat_num):
            convs.append(Module(channels=out_channels, inner_channels=out_channels // 2, act=act, norm=norm))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    PARA_R53 = dict(Module=DarkNetResidual, channels=32, repeat_nums=(1, 2, 8, 8, 4))

    @staticmethod
    def R53(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetBkbn(**DarkNetBkbn.PARA_R53, act=act, norm=norm)


# ConvResidualRepeat+CBA+Res
class CSPBlockV4(nn.Module):
    def __init__(self, Module, in_channels, out_channels, shortcut_channels, backbone_channels, backbone_inner_channels,
                 repeat_num, act=ACT.LK, norm=NORM.BATCH):
        super(CSPBlockV4, self).__init__()
        self.shortcut = Ck1NA(in_channels=in_channels, out_channels=shortcut_channels, act=act, norm=norm)
        backbone = [Ck1s1NA(in_channels=in_channels, out_channels=backbone_channels, act=act, norm=norm)]
        for i in range(repeat_num):
            backbone.append(Module(
                channels=backbone_channels, inner_channels=backbone_inner_channels, act=act, norm=norm))
        backbone.append(Ck1s1NA(in_channels=backbone_channels, out_channels=backbone_channels, act=act, norm=norm))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1NA(in_channels=shortcut_channels + backbone_channels, out_channels=out_channels, act=act,
                                norm=norm)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


# ConvResidualRepeat+CBA+Res
class CSPBlockV5(nn.Module):
    def __init__(self, Module, in_channels, out_channels, shortcut_channels, backbone_channels, backbone_inner_channels,
                 repeat_num, act=ACT.LK, norm=NORM.BATCH):
        super(CSPBlockV5, self).__init__()
        self.shortcut = Ck1s1NA(in_channels=in_channels, out_channels=shortcut_channels, act=act, norm=norm)
        backbone = [Ck1s1NA(in_channels=in_channels, out_channels=backbone_channels, act=act, norm=norm)]
        for i in range(repeat_num):
            backbone.append(Module(
                channels=backbone_channels, inner_channels=backbone_inner_channels, act=act, norm=norm))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1NA(in_channels=shortcut_channels + backbone_channels, out_channels=out_channels, act=act,
                                norm=norm)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


class DarkNetV4Bkbn(nn.Module):
    def __init__(self, Module, channels, repeat_nums, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        super(DarkNetV4Bkbn, self).__init__()
        self.pre = Ck3s1NA(in_channels=in_channels, out_channels=channels, act=act, norm=norm)
        self.stage1 = DarkNetV4Bkbn.ModuleRepeat(Module=Module, in_channels=channels, out_channels=channels * 2,
                                                 repeat_num=repeat_nums[0], stride=2, act=act, norm=norm)
        self.stage2 = DarkNetV4Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 2, out_channels=channels * 4,
                                                 repeat_num=repeat_nums[1], stride=2, act=act, norm=norm)
        self.stage3 = DarkNetV4Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 4, out_channels=channels * 8,
                                                 repeat_num=repeat_nums[2], stride=2, act=act, norm=norm)
        self.stage4 = DarkNetV4Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 8, out_channels=channels * 16,
                                                 repeat_num=repeat_nums[3], stride=2, act=act, norm=norm)
        self.stage5 = DarkNetV4Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 16, out_channels=channels * 32,
                                                 repeat_num=repeat_nums[4], stride=2, act=act, norm=norm)

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK, norm=NORM.BATCH):
        convs = nn.Sequential(
            Ck3NA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act, norm=norm),
            CSPBlockV4(Module=Module, in_channels=out_channels, out_channels=out_channels,
                       shortcut_channels=out_channels // 2, backbone_inner_channels=out_channels // 2,
                       backbone_channels=out_channels // 2, repeat_num=repeat_num, act=act, norm=norm))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    @staticmethod
    def R53(act=ACT.RELU, norm=NORM.BATCH):
        return DarkNetV4Bkbn(**DarkNetBkbn.PARA_R53, act=act, norm=norm)


class DarkNetV5Bkbn(nn.Module):
    def __init__(self, Module, channels, repeat_nums, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        super(DarkNetV5Bkbn, self).__init__()
        self.pre = CpaNA(in_channels=in_channels, out_channels=channels, kernel_size=7, stride=2, act=act, norm=norm)
        self.stage1 = DarkNetV5Bkbn.ModuleRepeat(Module=Module, in_channels=channels, out_channels=channels * 2,
                                                 repeat_num=repeat_nums[0], stride=2, act=act, norm=norm)
        self.stage2 = DarkNetV5Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 2, out_channels=channels * 4,
                                                 repeat_num=repeat_nums[1], stride=2, act=act, norm=norm)
        self.stage3 = DarkNetV5Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 4, out_channels=channels * 8,
                                                 repeat_num=repeat_nums[2], stride=2, act=act, norm=norm)
        self.stage4 = DarkNetV5Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 8, out_channels=channels * 16,
                                                 repeat_num=repeat_nums[3], stride=2, act=act, norm=norm)

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=2, act=ACT.LK, norm=NORM.BATCH):
        convs = nn.Sequential(
            Ck3NA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act, norm=norm),
            CSPBlockV5(Module=Module, in_channels=out_channels, out_channels=out_channels,
                       shortcut_channels=out_channels // 2, backbone_inner_channels=out_channels // 2,
                       backbone_channels=out_channels // 2, repeat_num=repeat_num, act=act, norm=norm))
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        return feats4

    PARA_NANO = dict(Module=DarkNetResidual, channels=16, repeat_nums=(1, 2, 3, 1))
    PARA_SMALL = dict(Module=DarkNetResidual, channels=32, repeat_nums=(1, 2, 3, 1))
    PARA_MEDIUM = dict(Module=DarkNetResidual, channels=48, repeat_nums=(2, 4, 6, 2))
    PARA_LARGE = dict(Module=DarkNetResidual, channels=64, repeat_nums=(3, 6, 9, 3))
    PARA_XLARGE = dict(Module=DarkNetResidual, channels=80, repeat_nums=(4, 8, 12, 4))

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5Bkbn(**DarkNetV5Bkbn.PARA_NANO, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5Bkbn(**DarkNetV5Bkbn.PARA_SMALL, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5Bkbn(**DarkNetV5Bkbn.PARA_MEDIUM, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Large(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5Bkbn(**DarkNetV5Bkbn.PARA_LARGE, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def XLarge(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5Bkbn(**DarkNetV5Bkbn.PARA_XLARGE, act=act, norm=norm, in_channels=in_channels)


class DarkNetV5ExtBkbn(DarkNetV5Bkbn):
    def __init__(self, Module, channels, repeat_nums, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        DarkNetV5Bkbn.__init__(self, Module, channels, repeat_nums, act=act, norm=norm, in_channels=in_channels)
        self.stage5 = DarkNetV5Bkbn.ModuleRepeat(Module=Module, in_channels=channels * 16, out_channels=channels * 16,
                                                 repeat_num=repeat_nums[4], stride=2, act=act, norm=norm)

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    PARA_NANO = dict(Module=DarkNetResidual, channels=16, repeat_nums=(1, 2, 3, 2, 1))
    PARA_SMALL = dict(Module=DarkNetResidual, channels=32, repeat_nums=(1, 2, 3, 2, 1))
    PARA_MEDIUM = dict(Module=DarkNetResidual, channels=48, repeat_nums=(2, 4, 6, 3, 2))
    PARA_LARGE = dict(Module=DarkNetResidual, channels=64, repeat_nums=(3, 6, 9, 4, 3))
    PARA_XLARGE = dict(Module=DarkNetResidual, channels=80, repeat_nums=(4, 8, 12, 6, 4))

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5ExtBkbn(**DarkNetV5ExtBkbn.PARA_NANO, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5ExtBkbn(**DarkNetV5ExtBkbn.PARA_SMALL, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5ExtBkbn(**DarkNetV5ExtBkbn.PARA_MEDIUM, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Large(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5ExtBkbn(**DarkNetV5ExtBkbn.PARA_LARGE, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def XLarge(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return DarkNetV5ExtBkbn(**DarkNetV5ExtBkbn.PARA_XLARGE, act=act, norm=norm, in_channels=in_channels)


class CSPBlockV4Tiny(nn.Module):
    def __init__(self, channels, act=ACT.LK, norm=NORM.BATCH):
        super(CSPBlockV4Tiny, self).__init__()
        self.inner_channels = channels // 2
        self.conv1 = Ck3s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)
        self.conv2 = Ck3s1NA(in_channels=self.inner_channels, out_channels=self.inner_channels, act=act,
                             norm=norm)
        self.conv3 = Ck3s1NA(in_channels=self.inner_channels, out_channels=self.inner_channels, act=act,
                             norm=norm)
        self.conv4 = Ck1s1NA(in_channels=channels, out_channels=channels, act=act, norm=norm)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_p = x1[:, self.inner_channels:, :, :]
        x2 = self.conv2(x1_p)
        x3 = self.conv3(x2)
        x32 = torch.cat([x3, x2], dim=1)
        x4 = self.conv4(x32)
        x14 = torch.cat([x1, x4], dim=1)
        return x14


class CSPDarkNetV4TinyBkbn(nn.Module):
    def __init__(self, channels, act=ACT.LK, norm=NORM.BATCH, in_channels=3):
        super(CSPDarkNetV4TinyBkbn, self).__init__()

        self.stage1 = nn.Sequential(
            Ck3NA(in_channels=in_channels, out_channels=channels, stride=2, act=act, norm=norm),
            Ck3NA(in_channels=channels, out_channels=channels * 2, stride=2, act=act, norm=norm),
            CSPBlockV4Tiny(channels=channels * 2, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CSPBlockV4Tiny(channels=channels * 4, act=act, norm=norm),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            CSPBlockV4Tiny(channels=channels * 8, act=act, norm=norm),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Ck3s1NA(in_channels=channels * 16, out_channels=channels * 16, act=act, norm=norm),
            Ck1s1NA(in_channels=channels * 16, out_channels=channels * 8, act=act, norm=norm)

        )
        self.c1_2 = nn.Sequential(
            Ck1s1NA(in_channels=channels * 8, out_channels=channels * 4, act=act, norm=norm),
            nn.UpsamplingNearest2d(scale_factor=2),

        )
        self.c2 = Ck3s1NA(in_channels=channels * 8, out_channels=channels * 16, act=act, norm=norm)
        self.c1 = Ck3s1NA(in_channels=channels * 20, out_channels=channels * 8, act=act, norm=norm)

    def forward(self, imgs):
        feat1 = self.stage1(imgs)
        feat2 = self.stage2(feat1)
        c1 = torch.cat([feat1, self.c1_2(feat2)], dim=1)
        c1 = self.c1(c1)
        c2 = self.c2(feat2)
        return c1, c2


if __name__ == '__main__':
    # model = DarkNetV5Bkbn.NANO()
    model = DarkNetV4Bkbn.R53()
    imgs = torch.rand(2, 3, 416, 416)
    y = model(imgs)
    print(y.size())
