from models.modules import *
from models.template import OneStageSegmentor


class UpSampleCat(nn.Module):
    def __init__(self, in_channels_s, in_channels_l, out_channels, repeat_num=1, num_feat_l=1,
                 act=ACT.RELU,norm=NORM.BATCH, with_convt=True):
        super().__init__()
        if with_convt:
            self.samper = CTk3NA(in_channels=in_channels_s, out_channels=out_channels, act=act,norm=norm, stride=2)
            in_channels = out_channels + in_channels_l * num_feat_l
        else:
            self.samper = nn.UpsamplingBilinear2d(scale_factor=2)
            in_channels = in_channels_s + in_channels_l * num_feat_l
        convs = []
        for i in range(repeat_num):
            convs.append(Ck3s1NA(in_channels=in_channels, out_channels=out_channels, act=act,norm=norm))
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)

    def forward(self, feats_s, *feats_l):
        feats_s = self.samper(feats_s)
        feats = torch.cat([feats_s] + list(feats_l), dim=1)
        feats = self.convs(feats)
        return feats


class UNetV1Main(nn.Module):
    def __init__(self, channelss, num_cls=1, repeat_num=2, with_convt=True, act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224),
                 in_channels=3):
        super(UNetV1Main, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        # downsampling
        self.convs = nn.ModuleList()
        last_channels = in_channels
        for i in range(len(channelss)):
            self.convs.append(UNetV1Main.Ck3Repeat(
                in_channels=last_channels, out_channels=channelss[i], repeat_num=repeat_num, with_pool=i > 0, act=act,norm=norm))
            last_channels = channelss[i]
        # upsampling
        self.uppers = nn.ModuleList()
        for i in range(len(channelss) - 1):
            self.uppers.append(UpSampleCat(
                in_channels_s=channelss[i + 1], in_channels_l=channelss[i], out_channels=channelss[i],
                act=act,norm=norm, with_convt=with_convt, num_feat_l=1, repeat_num=repeat_num))

        self.out = Ck1s1(in_channels=channelss[0], out_channels=num_cls)

    @staticmethod
    def Ck3Repeat(in_channels, out_channels, repeat_num=1, act=ACT.RELU,norm=NORM.BATCH, with_pool=True):
        convs = [nn.MaxPool2d(kernel_size=2)] if with_pool else []
        for i in range(repeat_num):
            convs.append(Ck3s1NA(in_channels=in_channels, out_channels=out_channels, act=act,norm=norm))
            in_channels = out_channels
        return nn.Sequential(*convs)

    def forward(self, imgs):
        feat = imgs
        feats = []
        for conv in self.convs:
            feat = conv(feat)
            feats.append(feat)
        for i in range(len(self.uppers) - 1, -1, -1):
            feat = self.uppers[i](feat, feats[i])
        masks = self.out(feat)
        return masks

    PARA_STD = dict(channelss=(64, 128, 256, 512, 1024), repeat_num=2, with_convt=False)
    PARA_TINY = dict(channelss=(32, 64, 128, 256), repeat_num=2, with_convt=False)

    @staticmethod
    def Std(img_size=(224, 224), num_cls=1, act=ACT.RELU,norm=NORM.BATCH, in_channels=3):
        return UNetV1Main(**UNetV1Main.PARA_STD, img_size=img_size, num_cls=num_cls, act=act,norm=norm, in_channels=in_channels)

    @staticmethod
    def Tiny(img_size=(224, 224), num_cls=1, act=ACT.RELU,norm=NORM.BATCH, in_channels=3):
        return UNetV1Main(**UNetV1Main.PARA_TINY, img_size=img_size, num_cls=num_cls, act=act,norm=norm, in_channels=in_channels)


class UNetConstMain(nn.Module):
    def __init__(self, batch_size, num_cls=1, img_size=(224, 224), in_channels=3):
        super(UNetConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.featmaps = nn.Parameter(torch.zeros(batch_size, num_cls, img_size[1], img_size[0]))

    def forward(self, imgs):
        return self.featmaps


class UNetV2Main(nn.Module):
    def __init__(self, channelss, num_cls=1, repeat_num=2, with_convt=True, act=ACT.RELU,norm=NORM.BATCH, img_size=(224, 224),
                 in_channels=3):
        super(UNetV2Main, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        # downsampling
        self.conv00 = UNetV1Main.Ck3Repeat(in_channels=in_channels, out_channels=channelss[0], repeat_num=repeat_num,
                                           with_pool=False, act=act,norm=norm)
        self.conv10 = UNetV1Main.Ck3Repeat(in_channels=channelss[0], out_channels=channelss[1], repeat_num=repeat_num,
                                           with_pool=True, act=act,norm=norm)
        self.conv20 = UNetV1Main.Ck3Repeat(in_channels=channelss[1], out_channels=channelss[2], repeat_num=repeat_num,
                                           with_pool=True, act=act,norm=norm)
        self.conv30 = UNetV1Main.Ck3Repeat(in_channels=channelss[2], out_channels=channelss[3], repeat_num=repeat_num,
                                           with_pool=True, act=act,norm=norm)
        self.conv40 = UNetV1Main.Ck3Repeat(in_channels=channelss[3], out_channels=channelss[4], repeat_num=repeat_num,
                                           with_pool=True, act=act,norm=norm)

        # upsampling
        self.upper01 = UpSampleCat(in_channels_s=channelss[1], in_channels_l=channelss[0], out_channels=channelss[0],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=1, repeat_num=repeat_num)
        self.upper11 = UpSampleCat(in_channels_s=channelss[2], in_channels_l=channelss[1], out_channels=channelss[1],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=1, repeat_num=repeat_num)
        self.upper21 = UpSampleCat(in_channels_s=channelss[3], in_channels_l=channelss[2], out_channels=channelss[2],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=1, repeat_num=repeat_num)
        self.upper31 = UpSampleCat(in_channels_s=channelss[4], in_channels_l=channelss[3], out_channels=channelss[3],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=1, repeat_num=repeat_num)

        self.upper02 = UpSampleCat(in_channels_s=channelss[1], in_channels_l=channelss[0], out_channels=channelss[0],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=2, repeat_num=repeat_num)
        self.upper12 = UpSampleCat(in_channels_s=channelss[2], in_channels_l=channelss[1], out_channels=channelss[1],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=2, repeat_num=repeat_num)
        self.upper22 = UpSampleCat(in_channels_s=channelss[3], in_channels_l=channelss[2], out_channels=channelss[2],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=2, repeat_num=repeat_num)

        self.upper03 = UpSampleCat(in_channels_s=channelss[1], in_channels_l=channelss[0], out_channels=channelss[0],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=3, repeat_num=repeat_num)
        self.upper13 = UpSampleCat(in_channels_s=channelss[2], in_channels_l=channelss[1], out_channels=channelss[1],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=3, repeat_num=repeat_num)

        self.upper04 = UpSampleCat(in_channels_s=channelss[1], in_channels_l=channelss[0], out_channels=channelss[0],
                                   act=act,norm=norm, with_convt=with_convt, num_feat_l=4, repeat_num=repeat_num)

        self.out1 = Ck1s1(in_channels=channelss[0], out_channels=num_cls)
        self.out2 = Ck1s1(in_channels=channelss[0], out_channels=num_cls)
        self.out3 = Ck1s1(in_channels=channelss[0], out_channels=num_cls)
        self.out4 = Ck1s1(in_channels=channelss[0], out_channels=num_cls)

    def forward(self, imgs):
        feats00 = self.conv00(imgs)
        feats10 = self.conv10(feats00)
        feats20 = self.conv20(feats10)
        feats30 = self.conv30(feats20)
        feats40 = self.conv40(feats30)

        feats01 = self.upper01(feats10, feats00)
        feats11 = self.upper11(feats20, feats10)
        feats21 = self.upper21(feats30, feats20)
        feats31 = self.upper31(feats40, feats30)

        feats02 = self.upper02(feats11, feats00, feats01)
        feats12 = self.upper12(feats21, feats10, feats11)
        feats22 = self.upper22(feats31, feats20, feats21)

        feats03 = self.upper03(feats12, feats00, feats01, feats02)
        feats13 = self.upper13(feats22, feats10, feats11, feats12)

        feats04 = self.upper04(feats13, feats00, feats01, feats02, feats03)

        masks1 = self.out1(feats01)
        masks2 = self.out2(feats02)
        masks3 = self.out3(feats03)
        masks4 = self.out4(feats04)

        masks = (masks1 + masks2 + masks3 + masks4) / 4
        return masks

    PARA_STD = dict(channels=(64, 128, 256, 512, 1024), num_repeat=2, with_convt=True)

    @staticmethod
    def Std(img_size=(224, 224), num_cls=1, act=ACT.RELU,norm=NORM.BATCH, in_channels=3):
        return UNetV2Main(**UNetV2Main.PARA_STD, img_size=img_size, num_cls=num_cls, act=act,norm=norm, in_channels=in_channels)


class UNet(OneStageSegmentor):

    @staticmethod
    def V1Std(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = UNetV1Main.Std(act=ACT.RELU,norm=NORM.BATCH, num_cls=num_cls, img_size=img_size, in_channels=in_channels)
        return UNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V1Tiny(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = UNetV1Main.Tiny(act=ACT.RELU,norm=NORM.BATCH, num_cls=num_cls, img_size=img_size, in_channels=in_channels)
        return UNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def V2Std(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = UNetV2Main.Std(act=ACT.RELU,norm=NORM.BATCH, num_cls=num_cls, img_size=img_size, in_channels=in_channels)
        return UNet(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)


if __name__ == '__main__':
    model = UNetV1Main.Tiny(num_cls=2)
    imgs = torch.zeros(size=(2, 3, 256, 256))
    y = model(imgs)
    print(y.size())
