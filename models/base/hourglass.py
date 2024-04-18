from models.base.resnet import Residual
from models.modules import *


class KP(nn.Module):
    def __init__(self, channelss, repeat_nums, act=ACT.RELU,norm=NORM.BATCH):
        super(KP, self).__init__()
        self.top = KP.ConvResidualRepeat(in_channels=channelss[0], out_channels=channelss[0], repeat_num=repeat_nums[0],
                                         stride=1, act=act,norm=norm)
        if len(channelss) == 1:
            self.down = None
        else:
            self.down = nn.Sequential(
                KP.ConvResidualRepeat(in_channels=channelss[0], out_channels=channelss[1], repeat_num=repeat_nums[0],
                                      stride=2, act=act,norm=norm),
                KP(channelss=channelss[1:], repeat_nums=repeat_nums[1:], act=act,norm=norm),
                KP.ConvResidualRepeatEd(in_channels=channelss[1], out_channels=channelss[0], repeat_num=repeat_nums[0],
                                        stride=1, act=act,norm=norm),
                nn.Upsample(scale_factor=2)
            )

    @staticmethod
    def ConvResidualRepeat(in_channels, out_channels, repeat_num=1, stride=1, act=ACT.RELU,norm=NORM.BATCH):
        backbone = []
        backbone.append(Residual(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act,norm=norm))
        for i in range(1, repeat_num):
            backbone.append(Residual(in_channels=out_channels, out_channels=out_channels, stride=1, act=act,norm=norm))
        backbone = nn.Sequential(*backbone)
        return backbone

    @staticmethod
    def ConvResidualRepeatEd(in_channels, out_channels, repeat_num=1, stride=1, act=ACT.RELU,norm=NORM.BATCH):
        backbone = []
        for i in range(1, repeat_num):
            backbone.append(Residual(in_channels=in_channels, out_channels=in_channels, stride=1, act=act,norm=norm))
        backbone.append(Residual(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act,norm=norm))
        backbone = nn.Sequential(*backbone)
        return backbone

    def forward(self, x):
        top = self.top(x)
        if self.down is not None:
            print(top.size(), self.down(x).size())
            top = top + self.down(x)
        return top


class EXKP(nn.Module):
    def __init__(self, nstack, channelss, repeat_nums, cps_channels=256, act=ACT.RELU,norm=NORM.BATCH, in_channels=3):
        super(EXKP, self).__init__()
        self.nstack = nstack
        cur_channels = channelss[0]
        self.pre = nn.Sequential(
            CpaNA(in_channels=in_channels, out_channels=128, kernel_size=7, stride=2, act=act,norm=norm),
            Residual(in_channels=128, out_channels=cur_channels, stride=2, act=act,norm=norm))

        self.backbones = nn.ModuleList([nn.Sequential(
            KP(channelss=channelss, repeat_nums=repeat_nums, act=act,norm=norm),
            Ck3s1NA(in_channels=cur_channels, out_channels=cps_channels, act=act,norm=norm)
        ) for _ in range(nstack)])

        self.sampler_outs = nn.ModuleList([
            Ck1s1NA(in_channels=cps_channels, out_channels=cur_channels, act=act,norm=norm)
            for _ in range(nstack - 1)])
        self.sampler_inters = nn.ModuleList([
            Ck1s1NA(in_channels=cur_channels, out_channels=cur_channels, act=act,norm=norm)
            for _ in range(nstack - 1)])
        self.mixers = nn.ModuleList([
            Residual(in_channels=cur_channels, out_channels=cur_channels, stride=1, act=act,norm=norm)
            for _ in range(nstack - 1)])

        self.act = ACT.build(act)

    def forward(self, imgs):
        inter = self.pre(imgs)
        outs = []
        for ind in range(self.nstack):
            out = self.backbones[ind](inter)
            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.sampler_inters[ind](inter) + self.sampler_outs[ind](out)
                inter = self.act(inter)
                inter = self.mixers[ind](inter)
        return outs[-1]

    SMALL_PARA = dict(nstack=1, channelss=[256, 256, 384, 384, 384, 512],
                      repeat_nums=[2, 2, 2, 2, 2, 4], cps_channels=256)
    LARGE_PARA = dict(nstack=2, channelss=[256, 256, 384, 384, 384, 512],
                      repeat_nums=[2, 2, 2, 2, 2, 4], cps_channels=256)

    @staticmethod
    def Small(act=ACT.RELU,norm=NORM.BATCH):
        return EXKP(**EXKP.SMALL_PARA, act=act,norm=norm)

    @staticmethod
    def Large(act=ACT.RELU,norm=NORM.BATCH):
        return EXKP(**EXKP.LARGE_PARA, act=act,norm=norm)


if __name__ == '__main__':
    model = EXKP.Small()
    # model = EXKP(channels=[10, 11, 12, 13], repeat_nums=[2, 1, 2, 1], nstack=2, cps_channels=40)
    # model=kp_module(n=3,dims=[10, 11, 12, 13],modules=[2, 2, 2, 2])
    # model=exkp(n=3, nstack=2, dims=[10, 11, 12, 13], modules=[2, 1, 2, 1])
    # model2onnx(model, './exkp_cus.onnx', input_size=(1, 3, 128, 128))
    test_input = torch.rand(size=(1, 3, 896, 896))
    y = model(test_input)
    # print(y.size())
