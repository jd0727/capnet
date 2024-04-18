from models.base.modules import *


class DSAMP_TYPE:
    MAX = 'max'
    MEAN = 'mean'
    CONV = 'conv'


class VGGBkbn(nn.Module):
    def __init__(self, repeat_nums, channelss, strides, act=ACT.RELU, norm=NORM.BATCH, in_channels=3,
                 dsamp_type=DSAMP_TYPE.MAX):
        super(VGGBkbn, self).__init__()
        self.stage1 = VGGBkbn.Ck3Repeat(in_channels=in_channels, out_channels=channelss[0], dsamp_type=dsamp_type,
                                        repeat_num=repeat_nums[0], act=act, norm=norm, stride=strides[0])
        self.stage2 = VGGBkbn.Ck3Repeat(in_channels=channelss[0], out_channels=channelss[1], dsamp_type=dsamp_type,
                                        repeat_num=repeat_nums[1], act=act, norm=norm, stride=strides[1])
        self.stage3 = VGGBkbn.Ck3Repeat(in_channels=channelss[1], out_channels=channelss[2], dsamp_type=dsamp_type,
                                        repeat_num=repeat_nums[2], act=act, norm=norm, stride=strides[2])
        self.stage4 = VGGBkbn.Ck3Repeat(in_channels=channelss[2], out_channels=channelss[3], dsamp_type=dsamp_type,
                                        repeat_num=repeat_nums[3], act=act, norm=norm, stride=strides[3])
        self.stage5 = VGGBkbn.Ck3Repeat(in_channels=channelss[3], out_channels=channelss[4], dsamp_type=dsamp_type,
                                        repeat_num=repeat_nums[4], act=act, norm=norm, stride=strides[4])

    @staticmethod
    def Ck3Repeat(in_channels, out_channels, repeat_num=1, stride=1, act=ACT.RELU, norm=NORM.BATCH,
                  dsamp_type=DSAMP_TYPE.MAX):
        backbone = []
        if stride > 1 and dsamp_type == DSAMP_TYPE.MAX:
            backbone.append(nn.MaxPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=False))
            stride = 1
        elif stride > 1 and dsamp_type == DSAMP_TYPE.MEAN:
            backbone.append(nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=False))
            stride = 1
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(
                Ck3NA(in_channels=last_channels, out_channels=out_channels, stride=stride, act=act, norm=norm))
            stride = 1
            last_channels = out_channels
        return nn.Sequential(*backbone)

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        return feats5

    PARA_A = dict(repeat_nums=(1, 1, 2, 2, 2), channelss=(64, 128, 256, 512, 512),
                  strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.MAX)
    PARA_B = dict(repeat_nums=(2, 2, 2, 2, 2), channelss=(64, 128, 256, 512, 512),
                  strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.MAX)
    PARA_D = dict(repeat_nums=(2, 2, 3, 3, 3), channelss=(64, 128, 256, 512, 512),
                  strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.MAX)
    PARA_E = dict(repeat_nums=(2, 2, 4, 3, 4), channelss=(64, 128, 256, 512, 512),
                  strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.MAX)

    PARA_NANO = dict(repeat_nums=(1, 1, 1, 2, 2), channelss=(32, 32, 64, 64, 64),
                     strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.CONV)
    PARA_SMALL = dict(repeat_nums=(2, 2, 2, 3, 3), channelss=(32, 32, 64, 128, 128),
                      strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.CONV)
    PARA_MEDIUM = dict(repeat_nums=(2, 2, 2, 3, 3), channelss=(32, 64, 128, 256, 256),
                       strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.CONV)
    PARA_LARGE = dict(repeat_nums=(2, 2, 3, 3, 3), channelss=(64, 128, 256, 512, 512),
                      strides=(1, 2, 2, 2, 2), dsamp_type=DSAMP_TYPE.CONV)

    @staticmethod
    def A(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_A, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def B(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_B, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def D(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_D, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def E(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_E, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_NANO, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_SMALL, act=act, norm=norm, in_channels=in_channels)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return VGGBkbn(**VGGBkbn.PARA_MEDIUM, act=act, norm=norm, in_channels=in_channels)


class VGGMain(VGGBkbn, ImageONNXExportable):
    def __init__(self, repeat_nums, channelss, strides, act=ACT.RELU, norm=NORM.BATCH, num_cls=20, head_channel=4096,
                 img_size=(224, 224), in_channels=3, pool_size=None, drop_rate=0.2, dsamp_type=DSAMP_TYPE.MAX):
        super(VGGMain, self).__init__(repeat_nums=repeat_nums, channelss=channelss, strides=strides,
                                      act=act, norm=norm, in_channels=in_channels, dsamp_type=dsamp_type)
        self.num_cls = num_cls
        pool_size = pool_size if pool_size is not None else VGGMain._get_pool_size(img_size, strides)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.head = nn.Sequential(
            nn.Linear(pool_size[0] * pool_size[1] * channelss[-1], head_channel),
            ACT.build(act),
            nn.Dropout(drop_rate),
            nn.Linear(head_channel, head_channel),
            ACT.build(act),
            nn.Dropout(drop_rate),
            nn.Linear(head_channel, num_cls),
        )
        self._img_size = img_size
        self._in_channels = in_channels

    @staticmethod
    def _get_pool_size(img_size, strides):
        scaler = np.prod(strides) * 2
        return (int(img_size[0] / scaler), int(img_size[1] / scaler))

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    def forward(self, imgs):
        feat5 = super(VGGMain, self).forward(imgs)
        feat = self.pool(feat5)
        feat = feat.reshape(feat.size(0), -1)
        feat = self.head(feat)
        return feat

    PARA_A = dict(VGGBkbn.PARA_A, head_channel=4096)
    PARA_B = dict(VGGBkbn.PARA_B, head_channel=4096)
    PARA_D = dict(VGGBkbn.PARA_D, head_channel=4096)
    PARA_E = dict(VGGBkbn.PARA_E, head_channel=4096)

    PARA_AC = dict(VGGBkbn.PARA_A, head_channel=512)
    PARA_BC = dict(VGGBkbn.PARA_B, head_channel=512)
    PARA_DC = dict(VGGBkbn.PARA_D, head_channel=512)
    PARA_EC = dict(VGGBkbn.PARA_E, head_channel=512)

    PARA_NANO = dict(VGGBkbn.PARA_NANO, head_channel=256)
    PARA_SMALL = dict(VGGBkbn.PARA_SMALL, head_channel=512)
    PARA_MEDIUM = dict(VGGBkbn.PARA_MEDIUM, head_channel=1024)
    PARA_LARGE = dict(VGGBkbn.PARA_LARGE, head_channel=2048)

    @staticmethod
    def A(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_A, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def B(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_B, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def D(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_D, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def E(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_E, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def AC(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_AC, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def BC(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_BC, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def DC(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_DC, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def EC(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_EC, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_NANO, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_SMALL, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_MEDIUM, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)

    @staticmethod
    def Large(act=ACT.RELU, norm=NORM.BATCH, num_cls=20, img_size=(224, 224), in_channels=3, pool_size=None):
        return VGGMain(**VGGMain.PARA_LARGE, act=act, num_cls=num_cls, img_size=img_size,
                       in_channels=in_channels, pool_size=pool_size, norm=norm)


class RevVGGBkbn(nn.Module):
    def __init__(self, repeat_nums, channelss, strides, act=ACT.RELU, norm=NORM.BATCH,
                 in_channels=512, out_channels=1):
        super(RevVGGBkbn, self).__init__()
        self.stage1 = RevVGGBkbn.Ck3Repeat(in_channels=in_channels, out_channels=channelss[0],
                                           repeat_num=repeat_nums[0], act=act, norm=norm, stride=strides[0])
        self.stage2 = RevVGGBkbn.Ck3Repeat(in_channels=channelss[0], out_channels=channelss[1],
                                           repeat_num=repeat_nums[1], act=act, norm=norm, stride=strides[1])
        self.stage3 = RevVGGBkbn.Ck3Repeat(in_channels=channelss[1], out_channels=channelss[2],
                                           repeat_num=repeat_nums[2], act=act, norm=norm, stride=strides[2])
        self.stage4 = RevVGGBkbn.Ck3Repeat(in_channels=channelss[2], out_channels=channelss[3],
                                           repeat_num=repeat_nums[3], act=act, norm=norm, stride=strides[3])
        self.stage5 = RevVGGBkbn.Ck3Repeat(in_channels=channelss[3], out_channels=channelss[4],
                                           repeat_num=repeat_nums[4], act=act, norm=norm, stride=strides[4])
        self.cvtor = Ck1s1(in_channels=channelss[4], out_channels=out_channels)

    @staticmethod
    def Ck3Repeat(in_channels, out_channels, repeat_num=1, stride=2, act=ACT.RELU, norm=NORM.BATCH, ):
        backbone = nn.Sequential()
        if stride > 1:
            backbone.add_module('pool', nn.UpsamplingNearest2d(scale_factor=stride))
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.add_module(str(i), Ck3NA(in_channels=last_channels, out_channels=out_channels, stride=1,
                                              act=act, norm=norm))
            last_channels = out_channels
        return backbone

    def forward(self, imgs):
        feats1 = self.stage1(imgs)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats5 = self.stage5(feats4)
        feats5 = self.cvtor(feats5)
        return feats5

    PARA_NANO = dict(repeat_nums=(2, 2, 1, 1, 1), channelss=(64, 64, 64, 32, 32), strides=(2, 2, 2, 2, 1))
    PARA_SMALL = dict(repeat_nums=(3, 3, 2, 2, 2), channelss=(128, 128, 64, 32, 32), strides=(2, 2, 2, 2, 1))
    PARA_MEDIUM = dict(repeat_nums=(3, 3, 2, 2, 2), channelss=(256, 256, 128, 64, 32), strides=(2, 2, 2, 2, 1))
    PARA_LARGE = dict(repeat_nums=(3, 3, 3, 2, 2), channelss=(512, 512, 256, 128, 64), strides=(2, 2, 2, 2, 1))

    PARA_A = dict(repeat_nums=(2, 2, 2, 1, 1), channelss=(512, 512, 256, 128, 64), strides=(2, 2, 2, 2, 1))
    PARA_B = dict(repeat_nums=(2, 2, 2, 2, 2), channelss=(512, 512, 256, 128, 64), strides=(2, 2, 2, 2, 1))
    PARA_D = dict(repeat_nums=(3, 3, 3, 2, 2), channelss=(512, 512, 256, 128, 64), strides=(2, 2, 2, 2, 1))
    PARA_E = dict(repeat_nums=(4, 3, 4, 2, 2), channelss=(512, 512, 256, 128, 64), strides=(2, 2, 2, 2, 1))

    @staticmethod
    def A(act=ACT.RELU, norm=NORM.BATCH, in_channels=512, out_channels=1):
        return RevVGGBkbn(**RevVGGBkbn.PARA_A, act=act, norm=norm,
                          in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def B(act=ACT.RELU, norm=NORM.BATCH, in_channels=512, out_channels=1):
        return RevVGGBkbn(**RevVGGBkbn.PARA_B, act=act, norm=norm,
                          in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def D(act=ACT.RELU, norm=NORM.BATCH, in_channels=512, out_channels=1):
        return RevVGGBkbn(**RevVGGBkbn.PARA_D, act=act, norm=norm,
                          in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def E(act=ACT.RELU, norm=NORM.BATCH, in_channels=512, out_channels=1):
        return RevVGGBkbn(**RevVGGBkbn.PARA_E, act=act, norm=norm,
                          in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, in_channels=512, out_channels=1):
        return RevVGGBkbn(**RevVGGBkbn.PARA_SMALL, act=act, norm=norm,
                          in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH, in_channels=512, out_channels=1):
        return RevVGGBkbn(**RevVGGBkbn.PARA_MEDIUM, act=act, norm=norm,
                          in_channels=in_channels, out_channels=out_channels)


class RevVGGMain(RevVGGBkbn, GeneratorONNXExportable):

    def __init__(self, repeat_nums, channelss, strides, act=ACT.RELU, norm=NORM.BATCH, in_features=20,
                 head_channel=4096, img_size=(224, 224), pool_size=None, pool_channels=512, out_channels=1,
                 ):
        super(RevVGGMain, self).__init__(
            repeat_nums=repeat_nums, channelss=channelss, strides=strides, act=act, norm=norm,
            in_channels=pool_channels, out_channels=out_channels)
        self._in_features = in_features
        self._pool_channels = pool_channels
        pool_size = pool_size if pool_size is not None else VGGMain._get_pool_size(img_size, strides)
        self._pool_size = pool_size
        self._feat_size = RevVGGMain._get_feat_size(img_size, strides)
        self.head = nn.Sequential(
            nn.Linear(in_features, head_channel),
            ACT.build(act),
            nn.Linear(head_channel, head_channel),
            ACT.build(act),
            nn.Linear(head_channel, pool_size[0] * pool_size[1] * pool_channels),
        )
        self._img_size = img_size

    @staticmethod
    def _get_feat_size(img_size, strides):
        scaler = np.prod(strides)
        return (int(img_size[0] / scaler), int(img_size[1] / scaler))

    @property
    def in_features(self):
        return self._in_features

    PARA_A = dict(RevVGGBkbn.PARA_A, head_channel=4096, pool_channels=512)
    PARA_B = dict(RevVGGBkbn.PARA_B, head_channel=4096, pool_channels=512)
    PARA_D = dict(RevVGGBkbn.PARA_D, head_channel=4096, pool_channels=512)
    PARA_E = dict(RevVGGBkbn.PARA_E, head_channel=4096, pool_channels=512)

    PARA_NANO = dict(RevVGGBkbn.PARA_NANO, head_channel=256, pool_channels=64)
    PARA_SMALL = dict(RevVGGBkbn.PARA_SMALL, head_channel=512, pool_channels=128)
    PARA_MEDIUM = dict(RevVGGBkbn.PARA_MEDIUM, head_channel=1024, pool_channels=256)
    PARA_LARGE = dict(RevVGGBkbn.PARA_LARGE, head_channel=2048, pool_channels=512)

    PARA_AC = dict(RevVGGBkbn.PARA_A, head_channel=512, pool_channels=128)
    PARA_BC = dict(RevVGGBkbn.PARA_B, head_channel=512, pool_channels=128)
    PARA_DC = dict(RevVGGBkbn.PARA_D, head_channel=512, pool_channels=128)
    PARA_EC = dict(RevVGGBkbn.PARA_E, head_channel=512, pool_channels=128)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    def forward(self, feat):
        feat = self.head(feat)
        feat = feat.reshape(feat.size(0), self._pool_channels, self._pool_size[1], self._pool_size[0])
        if not (self._pool_size[0] == self._feat_size[0] and self._pool_size[1] == self._feat_size[1]):
            feat = F.interpolate(feat, size=(self._feat_size[1], self._feat_size[0]))
        feat = super(RevVGGMain, self).forward(feat)
        return feat

    @staticmethod
    def A(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_A, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def B(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_B, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def D(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_D, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def E(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_E, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def AC(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_AC, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def BC(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_BC, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def DC(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_DC, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def EC(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_EC, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def Nano(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_NANO, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def Small(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_SMALL, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def Medium(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_MEDIUM, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)

    @staticmethod
    def Large(act=ACT.RELU, norm=NORM.BATCH, img_size=(224, 224), in_features=20, out_channels=3, pool_size=None):
        return RevVGGMain(**RevVGGMain.PARA_LARGE, act=act, norm=norm, in_features=in_features, img_size=img_size,
                          out_channels=out_channels, pool_size=pool_size)


class VGG(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=PACK.AUTO, img_size=(224, 224), num_cls=3):
        super(VGG, self).__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    def freeze_para(self):
        for name, para in self.named_parameters():
            if 'stage' in name:
                print('Freeze ' + name)
                para.requires_grad = False
            else:
                print('Activate ' + name)
                para.requires_grad = True
        return True

    @staticmethod
    def A(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.A(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                             in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def B(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.B(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                             in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def D(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.D(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                             in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def E(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.E(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                             in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def AC(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.AC(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                              in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def BC(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.BC(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                              in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def DC(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.DC(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                              in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def EC(device=None, pack=None, num_cls=20, img_size=(224, 224), in_channels=3):
        backbone = VGGMain.EC(act=ACT.RELU, norm=NORM.BATCH, num_cls=num_cls, img_size=img_size,
                              in_channels=in_channels)
        return VGG(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)


# if __name__ == '__main__':
#     model = VGG.A(img_size=(512, 512), device=1)
#     x = torch.rand(5, 3, 512, 512)
#     y = model(x)

if __name__ == '__main__':
    model = RevVGGMain.A(in_features=10, img_size=(32, 32), out_channels=6, norm=NORM.BATCH)
    model.eval()
    testx = torch.rand(size=(2, 10))
    y0 = model(testx)
    testx[0] += torch.rand(size=(10,)) * 100
    y1 = model(testx)
    z = y0 - y1
    print(z)
