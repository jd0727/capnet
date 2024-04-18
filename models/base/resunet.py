from models.base.modules import *
from models.base.resnet import Residual, Bottleneck
from models.template import OneStageSegmentor


class ResUNetAResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilations=(1, 3, 5), act=ACT.RELU, norm=NORM.BATCH):
        super(ResUNetAResidual, self).__init__()

        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(nn.Sequential(
                Ck3NA(in_channels=in_channels, out_channels=out_channels,
                      stride=stride, dilation=dilation, act=act, norm=norm),
                Ck3NA(in_channels=out_channels, out_channels=out_channels,
                      stride=1, dilation=dilation, act=None, norm=norm)
            ))

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1NA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, act=None, norm=norm)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        for seq in self.convs:
            residual = residual + seq(x)
        out = self.act(residual)
        return out


class BottleneckFull(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, act=ACT.RELU, norm=NORM.BATCH):
        super(BottleneckFull, self).__init__()
        self.conv1 = Ck1s1NA(in_channels=in_channels, out_channels=out_channels,
                             act=act, norm=norm)
        self.conv2 = Ck3NA(in_channels=out_channels, out_channels=out_channels,
                           stride=stride, dilation=dilation, act=act, norm=norm)
        self.conv3 = Ck1s1NA(in_channels=out_channels, out_channels=out_channels,
                             act=None, norm=norm)

        self.act = ACT.build(act)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Ck1NA(in_channels=in_channels, out_channels=out_channels,
                                  stride=stride, act=None, norm=norm)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + residual)
        return out


class ResUNetEnc(nn.Module):
    def __init__(self, Module, repeat_nums, in_channels, out_channelss=(32, 64), act=ACT.RELU, norm=NORM.BATCH,
                 strides=(2, 2)):
        super(ResUNetEnc, self).__init__()
        self.stages = nn.ModuleList([])
        for i, repeat_num in enumerate(repeat_nums):
            in_channels = in_channels if i == 0 else out_channelss[i - 1]
            self.stages.append(ResUNetEnc.ModuleRepeat(
                Module, in_channels=in_channels, out_channels=out_channelss[i], stride=strides[i],
                repeat_num=repeat_num, act=act, norm=norm, with_pool=False))

    @staticmethod
    def ModuleRepeat(Module, in_channels, out_channels, repeat_num=1, stride=1, act=ACT.RELU, norm=NORM.BATCH,
                     with_pool=False,
                     **kwargs):
        backbone = [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)] if with_pool else []
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(Module(
                in_channels=last_channels, out_channels=out_channels, stride=stride, act=act,norm=norm, **kwargs))
            last_channels = out_channels
            stride = 1
        backbone = nn.Sequential(*backbone)
        return backbone

    def forward(self, feat):
        feats = []
        for stage in self.stages:
            feat = stage(feat)
            feats.append(feat)
        return feats


class ResUNetAEnc(nn.Module):
    def __init__(self, Module, repeat_nums, in_channels, out_channelss=(32, 64), dilationss=((1, 3, 15), (1, 3, 15)),
                 strides=(2, 2), act=ACT.RELU, norm=NORM.BATCH):
        super(ResUNetAEnc, self).__init__()
        self.stages = nn.ModuleList([])
        for i, (repeat_num, dilations) in enumerate(zip(repeat_nums, dilationss)):
            in_channels = in_channels if i == 0 else out_channelss[i - 1]
            self.stages.append(ResUNetEnc.ModuleRepeat(
                Module, in_channels=in_channels, out_channels=out_channelss[i], stride=strides[i],
                repeat_num=repeat_num, act=act,norm=norm, dilations=dilations, with_pool=False))

    def forward(self, feat):
        feats = []
        for stage in self.stages:
            feat = stage(feat)
            feats.append(feat)
        return feats


class ResUNetDec(nn.Module):
    def __init__(self, Module, repeat_nums, in_channelss=(32, 64), out_channelss=(32, 64), act=ACT.RELU,
                 norm=NORM.BATCH,
                 strides=(2, 2)):
        super(ResUNetDec, self).__init__()
        self.stages = nn.ModuleList([])
        self.cvtors = nn.ModuleList([])
        self.strides = strides
        for i, repeat_num in enumerate(repeat_nums):
            if i < len(repeat_nums) - 1:
                self.cvtors.append(Ck1s1NA(
                    in_channels=in_channelss[i] + out_channelss[i + 1], out_channels=out_channelss[i], act=act,
                    norm=norm))
                in_channels = out_channelss[i]
            else:
                in_channels = in_channelss[i]
            self.stages.append(ResUNetEnc.ModuleRepeat(
                Module, in_channels=in_channels, out_channels=out_channelss[i], stride=1,
                repeat_num=repeat_num, act=act,norm=norm, with_pool=False))

    def forward(self, feats):
        feat = None
        fests_ret = [None] * len(feats)
        for i in range(len(feats) - 1, -1, -1):
            if i < len(feats) - 1:
                feat = torch.cat([F.interpolate(feat, scale_factor=self.strides[i + 1]), feats[i]], dim=1)
                feat = self.cvtors[i](feat)
            else:
                feat = feats[i]
            feat = self.stages[i](feat)
            fests_ret[i] = feat
        return fests_ret


class ResUNetADec(nn.Module):
    def __init__(self, Module, repeat_nums, in_channelss=(32, 64), out_channelss=(32, 64),
                 dilationss=((1, 3, 15), (1, 3, 15)), act=ACT.RELU, norm=NORM.BATCH, strides=(2, 2)):
        super(ResUNetADec, self).__init__()
        self.stages = nn.ModuleList([])
        self.cvtors = nn.ModuleList([])
        self.strides = strides
        for i, (repeat_num, dilations) in enumerate(zip(repeat_nums, dilationss)):
            if i < len(repeat_nums) - 1:
                self.cvtors.append(Ck1s1NA(
                    in_channels=in_channelss[i] + out_channelss[i + 1], out_channels=out_channelss[i], act=act,
                    norm=norm))
                in_channels = out_channelss[i]
            else:
                in_channels = in_channelss[i]
            self.stages.append(ResUNetEnc.ModuleRepeat(
                Module, in_channels=in_channels, out_channels=out_channelss[i], stride=1,
                repeat_num=repeat_num, act=act,norm=norm, dilations=dilations, with_pool=False))

    def forward(self, feats):
        feat = feats[-1]
        fests_ret = [None] * len(feats)
        for i in range(len(feats) - 1, -1, -1):
            if i < len(feats) - 1:
                feat = torch.cat([F.interpolate(feat, scale_factor=self.strides[i + 1]), feats[i]], dim=1)
                feat = self.cvtors[i](feat)
            feat = self.stages[i](feat)
            fests_ret[i] = feat
        return fests_ret


class ResUNetMain(ImageONNXExportable):

    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, pre_channels,
                 channelss, strides, out_channels=20, img_size=(224, 224), act=ACT.RELU, norm=NORM.BATCH,
                 in_channels=3):
        super(ResUNetMain, self).__init__()
        self._img_size = img_size
        self._in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.pre = CpaNA(in_channels=in_channels, out_channels=pre_channels, kernel_size=7, stride=1, act=act,
                         norm=norm)
        self.enc = ResUNetEnc(Module, repeat_nums_enc, in_channels=pre_channels, out_channelss=channelss,
                              strides=strides, act=act, norm=norm)
        self.dec = ResUNetDec(Module, repeat_nums_dec, in_channelss=channelss, out_channelss=channelss,
                              strides=strides, act=act, norm=norm)
        self.out = Ck1s1(in_channels=channelss[0] + pre_channels, out_channels=out_channels)

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def forward(self, imgs):
        feat = self.pre(imgs)
        feats = self.enc(feat)
        feats_rec = self.dec(feats)
        feat_rec = feats_rec[0]
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        imgs_rec = self.out(torch.cat([feat, feat_rec], dim=1))
        return imgs_rec

    PARA_LV6 = dict(Module=Residual, repeat_nums_enc=[1] * 6, repeat_nums_dec=[1] * 6, pre_channels=32,
                    channelss=(32, 64, 128, 256, 512, 1024), strides=[2] * 6)
    PARA_LV5 = dict(Module=Residual, repeat_nums_enc=[1] * 5, repeat_nums_dec=[1] * 5, pre_channels=32,
                    channelss=(32, 64, 128, 256, 512), strides=[2] * 5)

    # PARA_R50Full = dict(Module=BottleneckFull, repeat_nums_enc=(2, 3, 4, 6, 3), repeat_nums_dec=[1, 2, 2, 3, 2],
    #                     channelss=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2))
    PARA_R34 = dict(Module=Residual, repeat_nums_enc=(2, 3, 4, 6, 3), repeat_nums_dec=[1, 2, 2, 3, 2],
                    channelss=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2), pre_channels=32, )
    PARA_R50 = dict(Module=Bottleneck, repeat_nums_enc=(2, 3, 4, 6, 3), repeat_nums_dec=[1, 2, 2, 3, 2],
                    channelss=(128, 256, 512, 1024, 2048), strides=(2, 2, 2, 2, 2), pre_channels=64, )

    @staticmethod
    def LV6(img_size=(224, 224), out_channels=20, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return ResUNetMain(**ResUNetMain.PARA_LV6, img_size=img_size, out_channels=out_channels, act=act,norm=norm,
                           in_channels=in_channels)

    @staticmethod
    def LV5(img_size=(224, 224), out_channels=20, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return ResUNetMain(**ResUNetMain.PARA_LV5, img_size=img_size, out_channels=out_channels, act=act,norm=norm,
                           in_channels=in_channels)

    @staticmethod
    def R50(img_size=(224, 224), out_channels=20, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return ResUNetMain(**ResUNetMain.PARA_R50, img_size=img_size, out_channels=out_channels, act=act,norm=norm,
                           in_channels=in_channels)

    @staticmethod
    def R34(img_size=(224, 224), out_channels=20, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return ResUNetMain(**ResUNetMain.PARA_R34, img_size=img_size, out_channels=out_channels, act=act,norm=norm,
                           in_channels=in_channels)


class ResUNetAMain(ImageONNXExportable):
    def __init__(self, Module, repeat_nums_enc, repeat_nums_dec, strides, pre_channels,
                 channelss, dilationss, out_channels=20, img_size=(224, 224), act=ACT.RELU, norm=NORM.BATCH,
                 in_channels=3):
        super(ResUNetAMain, self).__init__()
        self._img_size = img_size
        self._in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.pre = CpaNA(in_channels=in_channels, out_channels=pre_channels, kernel_size=5, stride=1,  act=act,
                         norm=norm)
        self.enc = ResUNetAEnc(Module, repeat_nums_enc, in_channels=pre_channels, out_channelss=channelss,
                               dilationss=dilationss, act=act,norm=norm, strides=strides)
        self.psp1 = PSP(in_channels=channelss[-1], out_channels=channelss[-1], strides=(1, 2, 4, 8), act=act, norm=norm)
        self.dec = ResUNetADec(Module, repeat_nums_dec, in_channelss=channelss, out_channelss=channelss,
                               dilationss=dilationss, act=act,norm=norm, strides=strides)
        self.psp2 = PSP(in_channels=channelss[0], out_channels=channelss[0], strides=(1, 2, 4, 8), act=act, norm=norm)
        self.out = Ck1s1(in_channels=channelss[0] + pre_channels, out_channels=out_channels)

    @property
    def img_size(self):
        return self._img_size

    @property
    def in_channels(self):
        return self._in_channels

    def forward(self, imgs):
        feat = self.pre(imgs)
        feats = self.enc(feat)
        feats[-1] = self.psp1(feats[-1])
        feats_rec = self.dec(feats)
        feat_rec = feats_rec[0]
        feat_rec = self.psp2(feat_rec)
        if not self.strides[0] == 1:
            feat_rec = F.interpolate(feat_rec, scale_factor=self.strides[0])
        imgs_rec = self.out(torch.cat([feat, feat_rec], dim=1))
        return imgs_rec

    PARA_LV6 = dict(Module=ResUNetAResidual, repeat_nums_enc=[1] * 6, repeat_nums_dec=[1] * 6,
                    channelss=(32, 64, 128, 256, 512, 1024), strides=[2] * 6, pre_channels=32,
                    dilationss=((1, 3, 15, 31), (1, 3, 15, 31), (1, 3, 15), (1, 3, 15), (1,), (1,)))
    PARA_LV5 = dict(Module=ResUNetAResidual, repeat_nums_enc=[1] * 5, repeat_nums_dec=[1] * 5,
                    channelss=(32, 64, 128, 256, 512), strides=[2] * 5, pre_channels=32,
                    dilationss=((1, 3, 15, 31), (1, 3, 15, 31), (1, 3, 15), (1, 3, 15), (1,)))

    @staticmethod
    def LV6(img_size=(224, 224), out_channels=20, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return ResUNetAMain(**ResUNetAMain.PARA_LV6, img_size=img_size, out_channels=out_channels, act=act,norm=norm,
                            in_channels=in_channels)

    @staticmethod
    def LV5(img_size=(224, 224), out_channels=20, act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return ResUNetAMain(**ResUNetAMain.PARA_LV5, img_size=img_size, out_channels=out_channels, act=act,norm=norm,
                            in_channels=in_channels)


class ResUNet(OneStageSegmentor):

    @staticmethod
    def LV6(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = ResUNetMain.LV6(act=ACT.RELU, norm=NORM.BATCH, out_channels=num_cls + 1, img_size=img_size,
                                   in_channels=in_channels)
        return ResUNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def LV5(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = ResUNetMain.LV5(act=ACT.RELU, norm=NORM.BATCH, out_channels=num_cls + 1, img_size=img_size,
                                   in_channels=in_channels)
        return ResUNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def R50(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = ResUNetMain.R50(act=ACT.RELU, norm=NORM.BATCH, out_channels=num_cls + 1, img_size=img_size,
                                   in_channels=in_channels)
        return ResUNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def R34(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = ResUNetMain.R34(act=ACT.RELU, norm=NORM.BATCH, out_channels=num_cls + 1, img_size=img_size,
                                   in_channels=in_channels)
        return ResUNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def ALV6(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = ResUNetAMain.LV6(act=ACT.RELU, norm=NORM.BATCH, out_channels=num_cls + 1, img_size=img_size,
                                    in_channels=in_channels)
        return ResUNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def ALV5(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = ResUNetAMain.LV5(act=ACT.RELU, norm=NORM.BATCH, out_channels=num_cls + 1, img_size=img_size,
                                    in_channels=in_channels)
        return ResUNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)


if __name__ == '__main__':
    imgs = torch.zeros((2, 3, 128, 128))

    model = ResUNetMain.R34(img_size=(128, 128), num_cls=20)
    # y = model(imgs)
    torch.onnx.export(model, imgs, f='./test.onnx', opset_version=11)
