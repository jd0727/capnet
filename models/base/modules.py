from models.modules import *
from models.template import *


# <editor-fold desc='MLP'>
class MLP(nn.Module):

    def __init__(self, in_features: int, out_features: int, inner_featuress: Union[int, Tuple[int]] = (),
                 act_last=False, act=ACT.RELU, dropout=0.0):
        super().__init__()
        inner_featuress = [inner_featuress] if isinstance(inner_featuress, int) else inner_featuress
        in_featuress = [in_features] + list(inner_featuress)
        out_featuress = list(inner_featuress) + [out_features]
        backbone = []
        for i, (inc, outc) in enumerate(zip(in_featuress, out_featuress)):
            backbone.append(nn.Linear(in_features=inc, out_features=outc))
            backbone.append(nn.Dropout(dropout))
            if not act_last and i == len(inner_featuress):
                continue
            backbone.append(ACT.build(act))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        return self.backbone(x)


class MLC(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, inner_channelss: Union[int, Tuple[int]] = (),
                 act_last=False, act=ACT.RELU, norm=NORM.BATCH, kernel_size=1, dilation=1):
        super().__init__()
        inner_channelss = [inner_channelss] if isinstance(inner_channelss, int) else inner_channelss
        in_channelss = [in_channels] + list(inner_channelss)
        out_channelss = list(inner_channelss) + [out_channels]
        backbone = []
        for i, (inc, outc) in enumerate(zip(in_channelss, out_channelss)):
            act_i = None if not act_last and i == len(inner_channelss) else act
            backbone.append(
                CpaNA(in_channels=inc, out_channels=outc, kernel_size=kernel_size, dilation=dilation, act=act_i))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        return self.backbone(x)


# </editor-fold>


class ConvAPSpectral(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, act=None,
                 **kwargs):
        super(ConvAPSpectral, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation), groups=groups,
            bias=bias, **kwargs))
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x) if self.act else x
        return x


class Discriminator(nn.Module):
    def __init__(self, channels=64, num_repeat=4, act=ACT.RELU, norm=NORM.BATCH):
        super(Discriminator, self).__init__()
        bkbn = []
        last_channels = 3
        for n in range(num_repeat):
            out_channels = channels * min(2 ** n, 8)
            bkbn.append(ConvAPSpectral(in_channels=last_channels, out_channels=out_channels,
                                       kernel_size=5, stride=2, act=act, norm=norm))
            last_channels = out_channels
        self.bkbn = nn.Sequential(*bkbn)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=last_channels, out_features=1)

    def forward(self, input):
        feat = self.bkbn(input)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        cls = self.linear(feat)
        return torch.sigmoid(cls)


# <editor-fold desc='注意力'>
class SeModule(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super(SeModule, self).__init__()
        inner_channels = int(ratio * channels)
        self.se_pth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Ck1s1NA(in_channels=channels, out_channels=inner_channels, act=ACT.RELU, norm=NORM.BATCH),
            Ck1s1NA(in_channels=inner_channels, out_channels=channels, act=ACT.SIG)
        )

    def forward(self, x):
        return x * self.se_pth(x)


def attention_mutihead(feat_q, feat_k, feat_v, num_head):
    assert feat_q.size() == feat_k.size(), 'size err'
    head_channels = feat_q.size(1) // num_head
    feat_qu = feat_q.view(feat_q.size(0), num_head, head_channels, -1)
    feat_ku = feat_k.view(feat_k.size(0), num_head, head_channels, -1)
    feat_qut = feat_qu.transpose(dim0=-1, dim1=-2)
    pow = torch.matmul(feat_qut, feat_ku)
    pow = torch.softmax(pow / math.sqrt(head_channels), dim=-2)
    feat_vu = feat_v.view(feat_v.size(0), num_head, feat_v.size(1) // num_head, -1)
    recv = torch.matmul(feat_vu, pow)
    recv = recv.contiguous().view(feat_v.size())
    return recv


def attention(feat_q, feat_k, feat_v):
    assert feat_q.size() == feat_k.size(), 'size err'
    feat_qu = feat_q.view(feat_q.size(0), feat_q.size(1), -1)
    feat_ku = feat_k.view(feat_k.size(0), feat_k.size(1), -1)
    feat_qut = feat_qu.transpose(dim0=-1, dim1=-2)
    pow = torch.matmul(feat_qut, feat_ku)
    pow = torch.softmax(pow / math.sqrt(feat_q.size(1)), dim=-2)
    feat_vu = feat_v.view(feat_v.size(0), feat_v.size(1), -1)
    recv = torch.matmul(feat_vu, pow)
    recv = recv.contiguous().view(feat_v.size())
    return recv


class CpaBASelfAttentionMutiHead(nn.Module):
    def __init__(self, in_channels, qk_channels, out_channels, kernel_size_qk=1, kernel_size=1, num_head=8,
                 act=ACT.RELU, norm=NORM.BATCH):
        super(CpaBASelfAttentionMutiHead, self).__init__()
        self.conv_q = Cpa(in_channels=in_channels, out_channels=qk_channels, kernel_size=kernel_size_qk)
        self.conv_k = Cpa(in_channels=in_channels, out_channels=qk_channels, kernel_size=kernel_size_qk)
        self.conv_v = CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, act=act,
                            norm=norm)
        self.num_head = num_head

    def forward(self, feat):
        feat_q = self.conv_q(feat)
        feat_k = self.conv_k(feat)
        feat_v = self.conv_v(feat)
        recv = attention_mutihead(feat_q, feat_k, feat_v, num_head=self.num_head)
        return recv


class CpaBASelfAttention(nn.Module):
    def __init__(self, in_channels, qk_channels, out_channels, kernel_size_qk=1, kernel_size=1, act=ACT.RELU,
                 norm=NORM.BATCH):
        super(CpaBASelfAttention, self).__init__()
        self.conv_q = Cpa(in_channels=in_channels, out_channels=qk_channels, kernel_size=kernel_size_qk)
        self.conv_k = Cpa(in_channels=in_channels, out_channels=qk_channels, kernel_size=kernel_size_qk)
        self.conv_v = CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, act=act,
                            norm=norm)

    def forward(self, feat):
        feat_q = self.conv_q(feat)
        feat_k = self.conv_k(feat)
        feat_v = self.conv_v(feat)
        recv = attention(feat_q, feat_k, feat_v)
        return recv


class DualAttention(nn.Module):
    def __init__(self, in_channels, out_channels, act=ACT.RELU, norm=NORM.BATCH, ratio=0.25, num_head=8):
        super(DualAttention, self).__init__()
        self.cvtr = Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
        inner_channels = int(ratio * out_channels)
        self.se = SeModule(channels=out_channels, ratio=ratio)
        self.sa = CpaBASelfAttentionMutiHead(
            in_channels=in_channels, qk_channels=inner_channels, out_channels=out_channels, num_head=num_head, act=act,
            norm=norm)
        self.mixr = Ck1s1NA(in_channels=out_channels * 2, out_channels=out_channels, act=act, norm=norm)

    def forward(self, feat):
        feat_se = self.se(self.cvtr(feat))
        feat_sa = self.sa(feat)
        recv = self.mixr(torch.cat([feat_se, feat_sa]))
        return recv


# </editor-fold>

# <editor-fold desc='池化'>

class SPP(nn.Module):
    def __init__(self, kernels=(13, 9, 5), stride=1, shortcut=True):
        super(SPP, self).__init__()
        self.pools = nn.ModuleList()
        for kernel in kernels:
            padding = (kernel - 1) // 2
            self.pools.append(nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding))
        self.shortcut = shortcut

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outs = []
            for pool in self.pools:
                outs.append(pool(x))
            if self.shortcut:
                outs.append(x)
            outs = torch.cat(outs, dim=1)
        return outs


class SPPF(nn.Module):
    '''Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher'''

    def __init__(self, in_channels, out_channels, kernel_size=5, act=ACT.RELU,
                 norm=NORM.BATCH):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        inner_channels = in_channels // 2  # hidden channels
        self.redcr = Ck1s1NA(in_channels, inner_channels, act=act, norm=norm)
        self.concatr = Ck1s1NA(inner_channels * 4, out_channels, act=act, norm=norm)
        self.kernel_size = kernel_size
        self.pooler = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.redcr(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.pooler(x)
            y2 = self.pooler(y1)
            y3 = self.pooler(y2)
            return self.concatr(torch.cat((x, y1, y2, y3), dim=1))


class Focus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)


class PSP(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 2, 4, 8), act=ACT.RELU, norm=NORM.BATCH):
        super(PSP, self).__init__()
        num_stride = len(strides)
        self.strides = strides
        self.cvters = nn.ModuleList([
            Ck1s1NA(in_channels=in_channels, out_channels=out_channels // len(strides), act=act, norm=norm)
            for _ in range(num_stride)])
        self.mixor = Ck1s1A(in_channels=in_channels + out_channels, out_channels=out_channels, act=act)

    def forward(self, x):
        out = [x]
        for stride, cvter in zip(self.strides, self.cvters):
            x_i = F.max_pool2d(x, stride=stride, kernel_size=stride)
            x_i = cvter(x_i)
            x_i = F.interpolate(x_i, scale_factor=stride)
            out.append(x_i)
        out = torch.cat(out, dim=1)
        out = self.mixor(out)
        return out


# </editor-fold>

# <editor-fold desc='多特征处理'>
def _prase_lev_kwargs(lev, kwargs):
    kwargs_lev = {}
    for name, val in kwargs.items():
        if isinstance(val, list) or isinstance(val, tuple):
            kwargs_lev[name] = val[lev]
        else:
            kwargs_lev[name] = val
    return kwargs_lev


class BranchModule(nn.Module):
    def __init__(self, in_channels, out_channels, branch_num, **kwargs):
        super(BranchModule, self).__init__()
        assert branch_num >= 1, 'len err'
        self.modules = nn.ModuleList()
        for i in enumerate(range(branch_num)):
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            self.modules.append(self._build_module(i, in_channels, out_channels, **kwargs_i))

    @abstractmethod
    def _build_module(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass


class BranchModuleAdd(BranchModule):
    def __init__(self, in_channels, out_channels, branch_num, **kwargs):
        super(BranchModuleAdd, self).__init__(in_channels, out_channels, branch_num, **kwargs)

    def forward(self, feat):
        feat0 = self.modules[0](feat)
        for i in range(1, len(self.modules)):
            feat0 = feat0 + self.modules[i](feat)
        return feat0


class BranchModuleConcat(BranchModule):
    def __init__(self, in_channels, out_channels, branch_num, **kwargs):
        super(BranchModuleConcat, self).__init__(in_channels, out_channels, branch_num, **kwargs)

    def forward(self, feat):
        feats = []
        for i in range(len(self.modules)):
            feats.append(self.modules[i](feat))
        feats = torch.cat(feats, dim=1)
        return feats


class ParallelModule(nn.Module):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super(ParallelModule, self).__init__()
        self.modules = nn.ModuleList()
        for i, in_channels, out_channels in enumerate(zip(in_channelss, out_channelss)):
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            self.modules.append(self._build_module(i, in_channelss, out_channelss, **kwargs_i))

    @abstractmethod
    def _build_module(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feats):
        for i in range(len(feats)):
            feats[i] = self.modules[i](feats[i])
        return feats


class ParallelCpaBA(ParallelModule):

    def __init__(self, in_channelss, out_channelss, kernel_size=3, stride=1, dilation=1, groups=1, act=ACT.RELU,
                 norm=NORM.BATCH):
        super(ParallelCpaBA, self).__init__(
            in_channelss, out_channelss, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, act=act, norm=norm)

    def _build_module(self, lev, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                      act=ACT.RELU, norm=NORM.BATCH):
        return CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     dilation=dilation, groups=groups, act=act, norm=norm)


class CascadeModule(nn.Module):
    def __init__(self, in_channels, out_channelss, **kwargs):
        super(CascadeModule, self).__init__()
        self.modules = nn.ModuleList()
        last_channels = in_channels
        for i in range(len(out_channelss)):
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            self.modules.append(self._build_module(i, last_channels, out_channelss[i], **kwargs_i))

    @abstractmethod
    def _build_module(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feat):
        feats_out = []
        for i in range(len(self.modules)):
            feat = self.modules[i](feat)
            feats_out.append(feat)
        return feats_out


class CascadeCpaBA(ParallelModule):

    def __init__(self, in_channels, out_channelss, kernel_size=3, stride=1, dilation=1, groups=1, act=ACT.RELU,
                 norm=NORM.BATCH,
                 cross_c1=True):
        super(CascadeCpaBA, self).__init__(
            in_channels, out_channelss, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, act=act, norm=norm, cross_c1=cross_c1)

    def _build_module(self, lev, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                      act=ACT.RELU, norm=NORM.BATCH, cross_c1=True):
        kernel_size_lev = 1 if cross_c1 and lev % 2 == 0 else kernel_size
        return CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_lev, stride=stride,
                     dilation=dilation, groups=groups, act=act, norm=norm)


class ParallelCpaBARepeat(ParallelModule):
    def __init__(self, in_channelss, out_channelss, kernel_size=3, stride=1, dilation=1, groups=1, act=ACT.RELU,
                 norm=NORM.BATCH,
                 num_repeat=1, cross_c1=True):
        super(ParallelCpaBARepeat, self).__init__(
            in_channelss, out_channelss, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, act=act, norm=norm, num_repeat=num_repeat, cross_c1=cross_c1)

    def _build_module(self, lev, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                      act=ACT.RELU, norm=NORM.BATCH, num_repeat=1, cross_c1=True):
        if num_repeat == 1:
            return CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         dilation=dilation, groups=groups, act=act, norm=norm)
        else:
            convs = []
            for i in range(num_repeat):
                kernel_size_i = 1 if cross_c1 and i % 2 == 0 else kernel_size
                convs.append(CpaNA(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels,
                                   kernel_size=kernel_size_i, act=act, norm=norm))
                return nn.Sequential(*convs)


class MixStreamConcat(nn.Module):
    # feats_in[n]|-mixrs[n]->feats_out[n]-|
    #                                adprs[n-1]
    #                                   |
    # feats_in[n-1]|------------------[+]>-mixrs[n-1]->feats_out[n-1]

    def __init__(self, in_channelss, out_channelss, revsd=True, **kwargs):
        super(MixStreamConcat, self).__init__()
        num_lev = len(in_channelss)
        self.mixrs = nn.ModuleList([nn.Identity()] * num_lev)
        self.adprs = nn.ModuleList([nn.Identity()] * num_lev)
        self.revsd = revsd
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        out_channels_last = -1
        for i in iterator:
            in_channels = in_channelss[i]
            out_channels = out_channelss[i]
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            if out_channels_last == -1:
                adpr_channels = 0
            else:
                adpr_channels = self._adpr_channels(i, in_channels, out_channels, out_channels_last, **kwargs_i)
                self.adprs[i] = self._build_mixr(i, out_channels_last, adpr_channels, **kwargs_i)
            self.mixrs[i] = self._build_mixr(i, in_channels + adpr_channels, out_channels, **kwargs_i)
            out_channels_last = out_channels

    @abstractmethod
    def _adpr_channels(self, lev, in_channels, out_channels, out_channels_last, **kwargs) -> int:
        pass

    @abstractmethod
    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](torch.cat([feats[i], self.adprs[i](feat_buff)], dim=1))
            feats_out[i] = feat_buff
        return feats_out


class MixStreamAdd(nn.Module):
    # feats_in[n]|-mixrs[n]->feats_out[n]-|
    #                                adprs[n-1]
    #                                   |
    # feats_in[n-1]|------------------[+]>-mixrs[n-1]->feats_out[n-1]

    def __init__(self, in_channelss, out_channelss, revsd=True, **kwargs):
        super(MixStreamAdd, self).__init__()
        num_lev = len(in_channelss)
        self.mixrs = nn.ModuleList([nn.Identity()] * num_lev)
        self.adprs = nn.ModuleList([nn.Identity()] * num_lev)
        self.revsd = revsd
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        out_channels_last = -1
        for i in iterator:
            in_channels = in_channelss[i]
            out_channels = out_channelss[i]
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            if not out_channels_last == -1:
                self.adprs[i] = self._build_mixr(i, out_channels_last, in_channels, **kwargs_i)
            self.mixrs[i] = self._build_mixr(i, in_channels, out_channels, **kwargs_i)
            out_channels_last = out_channels

    @abstractmethod
    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](feats[i] + self.adprs[i](feat_buff))
            feats_out[i] = feat_buff
        return feats_out


class DownStreamConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)


class DownStreamConcatSamp(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, mode='nearest', scale_factor=2, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = F.upsample(self.adprs[i](feat_buff), scale_factor=self.scale_factor, mode=self.mode)
                feat_buff = self.mixrs[i](torch.cat([feats[i], feat_buff], dim=1))
            feats_out[i] = feat_buff
        return feats_out


class UpStreamConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)


class DownStreamCk1s1BAConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)

    def _adpr_channels(self, lev, in_channels, out_channels, out_channels_last, **kwargs) -> int:
        return out_channels_last

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


class UpStreamCk1s1BAConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)

    def _adpr_channels(self, lev, in_channels, out_channels, out_channels_last, **kwargs) -> int:
        return out_channels_last

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


class DownStreamAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)


class DownStreamAddSamp(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, mode='nearest', scale_factor=2, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = F.upsample(self.adprs[i](feat_buff), scale_factor=self.scale_factor, mode=self.mode)
                feat_buff = self.mixrs[i](feats[i] + feat_buff)
            feats_out[i] = feat_buff
        return feats_out


class UpStreamAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)


class DownStreamCk1s1BAAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


class UpStreamCk1s1BAAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


# </editor-fold>


if __name__ == '__main__':
    x1 = torch.zeros(size=(1, 5, 6, 6))
    x2 = torch.zeros(size=(1, 4, 6, 6))
    x3 = torch.zeros(size=(1, 3, 6, 6))
    feats = (x1, x2, x3)

    model = DownStreamAdd((5, 4, 3), (2, 4, 6))

    torch.onnx.export(model, (feats,), './test.onnx', opset_version=11)
    # y = model(feats)
