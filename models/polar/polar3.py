from models.base.modules import MLC
from models.base.vit import LocalAttentionMutiHead2D
from models.polar.polar1 import *
from models.polar.polar2 import PolarV2Layer


class PolarV3Layer(PolarLayerProtyp):
    def __init__(self, in_channels, stride, num_cls, img_size=(256, 256), num_div=16, num_dstr=16, act=ACT.RELU):
        super(PolarV3Layer, self).__init__(stride, num_cls, img_size=img_size, num_div=num_div)
        transformer = []
        for k in range(2):
            attn = LocalAttentionMutiHead2D(
                in_channels=in_channels, out_channels=in_channels, num_head=8, qk_channels=in_channels, dropout=0.0,
                kernel_size=7, dilation=1, act=act)
            mlp = MLC(in_channels=in_channels, out_channels=in_channels, inner_channelss=in_channels,
                      act_last=True, act=act, kernel_size=1, dilation=1)
            transformer.append(attn)
            transformer.append(mlp)
        self.transformer = nn.Sequential(*transformer)

        num_reg = num_dstr * num_div
        self.reg_dl = nn.Sequential(
            Ck1s1NA(in_channels=in_channels, out_channels=num_reg, act=act),
            Ck1s1(in_channels=num_reg, out_channels=num_reg),
        )

        self.reg_chot = nn.Sequential(
            Ck1s1NA(in_channels=in_channels, out_channels=num_cls, act=act),
            Ck1s1(in_channels=num_cls, out_channels=num_cls),
        )
        init_sig(bias=self.reg_chot[-1].conv.bias, prior_prob=0.001)

    def forward(self, feat):
        feat = self.transformer(feat)
        reg_dl = self.reg_dl(feat)
        reg_chot = self.reg_chot(feat)
        preds = PolarV2Layer.decode(
            reg_dl=reg_dl, reg_chot=reg_chot, stride=self.stride, xy_offset=self.xy_offset)
        return preds


class PolarV3Main(DarkNetV8Bkbn, ImageONNXExportable):

    def __init__(self, Module, channelss, repeat_nums, num_cls=80, act=ACT.SILU, img_size=(256, 256),
                 in_channels=3, num_div=36, num_dstr=16):
        DarkNetV8Bkbn.__init__(self, Module, channelss, repeat_nums, act=act, in_channels=in_channels)
        self.num_cls = num_cls
        self.num_dstr = num_dstr
        self._img_size = img_size
        self._in_channels = in_channels
        feat_channelss = channelss[1:]
        down_channelss = channelss[1:]
        up_channelss = channelss[1:]
        self.spp = YoloV8Main.C1C3RepeatSPP(in_channels=channelss[-1], out_channels=channelss[-1], act=act)
        self.down = YoloV8DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act)
        self.up = YoloV8UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act)
        self.layers = nn.ModuleList([
            PolarV3Layer(in_channels=channelss[1], stride=8, num_cls=num_cls, img_size=img_size,
                         num_div=num_div, num_dstr=num_dstr),
            PolarV3Layer(in_channels=channelss[2], stride=16, num_cls=num_cls, img_size=img_size,
                         num_div=num_div, num_dstr=num_dstr),
            PolarV3Layer(in_channels=channelss[3], stride=32, num_cls=num_cls, img_size=img_size,
                         num_div=num_div, num_dstr=num_dstr),
        ])

    @property
    def num_div(self):
        return self.layers[0].num_div

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        self.layer.img_size = img_size

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
    def Nano(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18, num_dstr=16):
        return PolarV3Main(**DarkNetV8Bkbn.PARA_NANO, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, num_dstr=num_dstr)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18, num_dstr=16):
        return PolarV3Main(**DarkNetV8Bkbn.PARA_SMALL, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, num_dstr=num_dstr)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18, num_dstr=16):
        return PolarV3Main(**DarkNetV8Bkbn.PARA_MEDIUM, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, num_dstr=num_dstr)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, img_size=(256, 256), in_channels=3, num_div=18, num_dstr=16):
        return PolarV3Main(**DarkNetV8Bkbn.PARA_LARGE, num_cls=num_cls, act=act, img_size=img_size,
                           in_channels=in_channels, num_div=num_div, num_dstr=num_dstr)


if __name__ == '__main__':
    model = PolarV3Main.Medium()
    model.export_onnx('./buff')
