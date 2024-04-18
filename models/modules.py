from collections.abc import Iterable
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import xywhsT2xyxysT, Union, copy
from utils.file import _pair


# <editor-fold desc='激活函数'>
# Swish
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(1.0))

    def forward(self, x):
        x = x * torch.sigmoid(x * self.beta)
        return x


# SiLU
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


# Mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


# Relu6
class ReLU6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, min=0, max=6)
        return x


class HSiLU(nn.Module):
    def forward(self, x):
        out = x * (torch.clamp(x, min=-3, max=3) / 6 + 0.5)
        return out


class HSigmoid(nn.Module):
    def forward(self, x):
        out = torch.clamp(x, min=-3, max=3) / 6 + 0.5
        return out


class ACT:
    LK = 'lk'
    RELU = 'relu'
    SIG = 'sig'
    RELU6 = 'relu6'
    MISH = 'mish'
    SILU = 'silu'
    GELU = 'gelu'
    HSILU = 'hsilu'
    HSIG = 'hsig'
    SWISH = 'swish'
    TANH = 'tanh'
    NONE = None

    @staticmethod
    def build(act_name=None):
        if isinstance(act_name, nn.Module):
            act = act_name
        elif act_name is None or act_name == '':
            act = None
        elif act_name == ACT.LK:
            act = nn.LeakyReLU(0.1, inplace=True)
        elif act_name == ACT.RELU:
            act = nn.ReLU(inplace=True)
        elif act_name == ACT.SIG:
            act = nn.Sigmoid()
        elif act_name == ACT.SWISH:
            act = Swish()
        elif act_name == ACT.RELU6:
            act = ReLU6()
        elif act_name == ACT.MISH:
            act = Mish()
        elif act_name == ACT.SILU:
            act = nn.SiLU()
        elif act_name == ACT.GELU:
            act = nn.GELU()
        elif act_name == ACT.HSILU:
            act = HSiLU()
        elif act_name == ACT.HSIG:
            act = HSigmoid()
        elif act_name == ACT.TANH:
            act = nn.Tanh()
        else:
            raise Exception('err act name' + str(act_name))
        return act


class NORM:
    BATCH = 'norm'
    GROUP = 'group'
    INSTANCE = 'instance'
    NONE = None

    @staticmethod
    def build(channels, norm=BATCH, num_groups=8):
        if norm is None:
            return None
        elif norm == NORM.BATCH:
            return nn.BatchNorm2d(channels)
        elif norm == NORM.GROUP:
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        elif norm == NORM.INSTANCE:
            return nn.InstanceNorm2d(channels)
        else:
            raise Exception('err')


# </editor-fold>

# <editor-fold desc='一般卷积子模块'>
_int2 = Union[int, Tuple[int, int]]


def _auto_pad(kernel_size: _int2, dilation: _int2) -> Tuple[int, int]:
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = ((kernel_size[0] - 1) * dilation[0] // 2, (kernel_size[1] - 1) * dilation[1] // 2)
    return padding


def _auto_pad_output(kernel_size: _int2, padding: _int2, stride: _int2) -> Tuple[int, int]:
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    stride = _pair(stride)
    padding_o = ((2 * padding[0] - kernel_size[0]) % stride[0], (2 * padding[1] - kernel_size[1]) % stride[1])
    return padding_o


# Conv
class C(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1,
                 bias: bool = True, device=None, dtype=None):
        super(C, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
            bias=bias, groups=groups, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return x

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups)

    @staticmethod
    def convert(c):
        if isinstance(c, C):
            return c
        elif isinstance(c, RCpa):
            ct = C(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            return ct
        elif isinstance(c, DC):
            ct = C(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            return ct
        else:
            raise Exception('err module ' + c.__class__.__name__)


class Cpa(C):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(Cpa, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), bias=bias, device=device,
            dtype=dtype)


class Ck1(C):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1, dilation: _int2 = 1,
                 groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(Ck1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
            dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)


class Ck1s1(C):
    def __init__(self, in_channels: int, out_channels: int, groups: _int2 = 1,
                 bias: bool = True, device=None, dtype=None):
        super(Ck1s1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
            groups=groups, bias=bias, device=device, dtype=dtype)


class Ck3(C):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1, groups: _int2 = 1,
                 dilation: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(Ck3, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)


class Ck3s1(C):
    def __init__(self, in_channels: int, out_channels: int, groups: _int2 = 1, dilation: _int2 = 1,
                 bias: bool = True, device=None, dtype=None):
        super(Ck3s1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dilation,
            dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)


# Conv+Act
class CA(C):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None,
                 device=None, dtype=None):
        super(CA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x) if self.act else x
        return x

    @staticmethod
    def convert(c):
        if isinstance(c, CA):
            return c
        elif isinstance(c, RCpaA):
            ct = CA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.act = c.act
            return ct
        elif isinstance(c, DCA):
            ct = CA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.act = c.act
            return ct
        else:
            raise Exception('err module ' + c.__class__.__name__)


# Conv+norm+Act padding=auto
class CpaA(CA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None,
                 device=None, dtype=None):
        super(CpaA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), bias=bias, act=act,
            device=device, dtype=dtype)


# Conv+norm+Act padding=auto
class Ck3A(CA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1, dilation: _int2 = 1,
                 groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(Ck3A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
            dilation=dilation, groups=groups, padding=dilation, bias=bias, act=act, device=device, dtype=dtype)


class Ck3s1A(CA):
    def __init__(self, in_channels: int, out_channels: int, dilation: _int2 = 1,
                 groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(Ck3s1A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
            dilation=dilation, groups=groups, padding=dilation, bias=bias, act=act, device=device, dtype=dtype)


class Ck1A(CA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1,
                 groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(Ck1A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
            dilation=1, groups=groups, padding=0, bias=bias, act=act, device=device, dtype=dtype)


class Ck1s1A(CA):
    def __init__(self, in_channels: int, out_channels: int,
                 groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(Ck1s1A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
            dilation=1, groups=groups, padding=0, bias=bias, act=act, device=device, dtype=dtype)


class CN(C):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1,
                 norm=NORM.BATCH, device=None, dtype=None):
        super(CN, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=norm is not None, groups=groups, device=device, dtype=dtype)
        self.norm = NORM.build(out_channels, norm)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm else x
        return x


class Ck1s1N(CN):
    def __init__(self, in_channels: int, out_channels: int, dilation: _int2 = 1, groups: _int2 = 1,
                 norm=NORM.BATCH, device=None, dtype=None):
        super(Ck1s1N, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
            groups=groups, norm=norm, device=device, dtype=dtype)


# Conv+norm+Act
class CNA(CA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1,
                 norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(CNA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not norm, groups=groups, act=act, device=device, dtype=dtype,
        )
        self.norm = NORM.build(out_channels, norm)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm else x
        x = self.act(x) if self.act else x
        return x

    @staticmethod
    def convert(c):
        if isinstance(c, CNA):
            return c
        elif isinstance(c, RCpaNA):
            ct = CNA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.norm = c.norm
            ct.act = c.act
            return ct
        elif isinstance(c, DCNA):
            ct = CNA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.norm = c.norm
            ct.act = c.act
            return ct
        else:
            raise Exception('err module ' + c.__class__.__name__)


# Conv+norm+Act padding=auto
class CpaNA(CNA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None,
                 ):
        super(CpaNA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), norm=norm, act=act,
            device=device, dtype=dtype)


class CpadwNA(CpaNA):
    def __init__(self, channels: int, kernel_size: _int2 = 1, stride: _int2 = 1, dilation: _int2 = 1,
                 norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(CpadwNA, self).__init__(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=channels, norm=norm, act=act, device=device, dtype=dtype)


class Ck1NA(CNA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1,
                 groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(Ck1NA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
            dilation=1, groups=groups, norm=norm, act=act, device=device, dtype=dtype)


class Ck1s1NA(Ck1NA):
    def __init__(self, in_channels: int, out_channels: int,
                 groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(Ck1s1NA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, stride=1, groups=groups,
            norm=norm, act=act, device=device, dtype=dtype)


class Ck3NA(CNA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1, dilation: _int2 = 1,
                 groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(Ck3NA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, groups=groups, norm=norm, act=act, device=device, dtype=dtype)


class Ck3s1NA(Ck3NA):
    def __init__(self, in_channels: int, out_channels: int, dilation: _int2 = 1,
                 groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(Ck3s1NA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, stride=1, groups=groups, dilation=dilation,
            norm=norm, act=act, device=device, dtype=dtype)


# ConvTrans
class CT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1,
                 output_padding: _int2 = 0, bias: bool = True, device=None, dtype=None):
        super(CT, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), groups=groups, padding=_pair(padding),
            output_padding=_pair(output_padding), dilation=dilation, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        return x

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    output_padding=self.conv.output_padding, kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups)


# ConvTrans+Act
class CTpa(CT):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        padding = _auto_pad(kernel_size, dilation)
        output_padding = _auto_pad_output(kernel_size, padding, stride)
        super(CTpa, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=groups,
                                   output_padding=output_padding, bias=bias, device=device, dtype=dtype)


# ConvTrans+Act
class CTA(CT):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1,
                 output_padding: _int2 = 0, bias: bool = True, act=None, device=None, dtype=None):
        super(CTA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups,
                                  output_padding=output_padding, bias=bias, device=device, dtype=dtype)
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x) if self.act else x
        return x


# ConvTrans+norm+Act
class CTpaA(CTA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        padding = _auto_pad(kernel_size, dilation)
        output_padding = _auto_pad_output(kernel_size, padding, stride)
        super(CTpaA, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, output_padding=output_padding, bias=bias, act=act, device=device, dtype=dtype)


# ConvTrans+norm+Act
class CTNA(CTA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 padding: _int2 = 0, dilation: _int2 = 1, groups: _int2 = 1,
                 output_padding: _int2 = 0, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(CTNA, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, output_padding=output_padding, bias=not norm, act=act, device=device, dtype=dtype)
        self.norm = NORM.build(out_channels, norm)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm else x
        x = self.act(x) if self.act else x
        return x


# ConvTrans+norm+Act padding=auto
class CTpaNA(CTNA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 1, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        padding = _auto_pad(kernel_size, dilation)
        output_padding = _auto_pad_output(kernel_size, padding, stride)
        super(CTpaNA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=padding, output_padding=output_padding, norm=norm, act=act,
            device=device, dtype=dtype)


# ConvTrans+norm+Act
class CTk3NA(CTNA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        output_padding = _auto_pad_output(3, dilation, stride)
        super(CTk3NA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
            dilation=dilation, groups=groups, padding=dilation, output_padding=output_padding, norm=norm, act=act,
            device=device, dtype=dtype)


# </editor-fold>

# <editor-fold desc='Rep卷积子模块'>
def _pad_wei_1(weight_1: torch.Tensor, kernel_size: _int2) -> torch.Tensor:
    kernel_size = _pair(kernel_size)
    pad_pre = (kernel_size[0] // 2, kernel_size[1] // 2)
    pad = (pad_pre[0], kernel_size[0] - pad_pre[0] - 1, pad_pre[1], kernel_size[1] - pad_pre[1] - 1)
    return F.pad(weight_1, pad=pad)


class RCpa(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(RCpa, self).__init__()
        padding = _auto_pad(kernel_size, dilation)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
            bias=bias, groups=groups, device=device, dtype=dtype)

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
            stride=_pair(stride), padding=(0, 0), dilation=(1, 1),
            bias=bias, groups=groups, device=device, dtype=dtype)

        nn.init.constant(self.conv_1.weight, 1)
        if self.conv_1.bias:
            nn.init.constant(self.conv_1.bias, 1)
        self.has_shortcut = in_channels == out_channels and stride == 1
        self._conv_eq = None

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups)

    @property
    def conv_wb(self):
        return self.conv.weight, self.conv.bias

    @property
    def conv_1_wb(self):
        return self.conv_1.weight, self.conv_1.bias

    @property
    def shortcut_wb(self):
        weight = torch.eye(self.conv_1.weight.size(1))[:, :, None, None]
        weight = weight.repeat(self.conv_1.groups, 1, 1, 1)
        return weight, None

    def _get_conv_eq(self):
        weight, bias = self.conv_wb
        weight_1, bias_1 = self.conv_1_wb
        kernel_size = self.conv.kernel_size
        pad_pre = (kernel_size[0] // 2, kernel_size[1] // 2)
        pad = (pad_pre[0], kernel_size[0] - pad_pre[0] - 1, pad_pre[1], kernel_size[1] - pad_pre[1] - 1)
        weight_eq = weight + F.pad(weight_1, pad=pad)
        bias_eq = None if bias is None else bias + bias_1
        if self.has_shortcut:
            weight_sc, _ = self.shortcut_wb
            weight_eq = weight_eq + F.pad(weight_sc, pad=pad)
        conv_eq = nn.Conv2d(**self.config)
        conv_eq.weight.data = weight_eq
        conv_eq.bias = nn.Parameter(bias_eq)
        return conv_eq

    def _init_wb(self, weight_eq, bias_eq):
        weight_1, bias_1 = self.conv_1_wb
        kernel_size = self.conv.kernel_size
        pad_pre = (kernel_size[0] // 2, kernel_size[1] // 2)
        pad = (pad_pre[0], kernel_size[0] - pad_pre[0] - 1, pad_pre[1], kernel_size[1] - pad_pre[1] - 1)
        weight = weight_eq - F.pad(weight_1, pad=pad)
        bias_eq = 0 if bias_eq is None else bias_eq
        bias = bias_eq if bias_1 is None else bias_eq - bias_1
        if self.has_shortcut:
            weight_sc, _ = self.shortcut_wb
            weight = weight - F.pad(weight_sc, pad=pad)
        self.conv.weight.data = weight
        self.conv.bias = nn.Parameter(bias)
        return self

    def train(self, mode: bool = True):
        last_mode = self.training
        super(RCpa, self).train(mode)
        if last_mode and not mode:
            self._conv_eq = self._get_conv_eq()
        elif not last_mode and mode:
            self._conv_eq = None

    @property
    def conv_eq(self):
        if self.training:
            return self._get_conv_eq()
        else:
            return self._conv_eq

    def forward(self, x):
        if self.training:
            out = self.conv(x) + self.conv_1(x)
            out = out + x if self.has_shortcut else out
        else:
            out = self._conv_eq(x)
        return out

    @staticmethod
    def convert(rc):
        if isinstance(rc, RCpa):
            return rc
        elif isinstance(rc, C):  # 不确保相等
            config = rc.config
            del config['padding']
            rct = RCpa(**config).to(rc.conv.weight.device)
            rct._init_wb(rc.conv.weight, rc.conv.bias)
            return rct
        elif isinstance(rc, DC):
            # 没写
            return RCpa.convert(C.convert(rc))
        else:
            raise Exception('err module ' + rc.__class__.__name__)


class RCk3(RCpa):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(RCk3, self).__init__(in_channels, out_channels, kernel_size=3, stride=stride,
                                   dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)


class RCk3s1(RCk3):
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(RCk3s1, self).__init__(
            in_channels, out_channels, stride=1,
            dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)


class RCpaA(RCpa):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(RCpaA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)
        self.act = ACT.build(act) if act else None

    def forward(self, x):
        out = super(RCpaA, self).forward(x)
        return self.act(out) if self.act else out

    @staticmethod
    def convert(rc):
        if isinstance(rc, RCpaA):
            return rc
        elif isinstance(rc, CA):
            config = rc.config
            del config['padding']
            rct = RCpaA(**config).to(rc.conv.weight.device)
            rct._init_wb(rc.conv.weight, rc.conv.bias)
            rct.act = rc.act
            return rct
        else:
            raise Exception('err module ' + rc.__class__.__name__)


class RCk3A(RCpaA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(RCk3A, self).__init__(in_channels, out_channels, kernel_size=3, stride=stride,
                                    dilation=dilation, groups=groups, bias=bias, act=act, device=device, dtype=dtype)


class RCk3s1A(RCk3A):
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None, device=None, dtype=None):
        super(RCk3s1A, self).__init__(in_channels, out_channels, stride=1,
                                      dilation=dilation, groups=groups, bias=bias, act=act, device=device, dtype=dtype)


class RCpaNA(RCpaA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(RCpaNA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     dilation=dilation, groups=groups, bias=not norm, act=act, device=device,
                                     dtype=dtype)
        self.norm = NORM.build(out_channels, norm)
        self.norm_1 = nn.BatchNorm2d(out_channels) if norm else None
        self.norm_sc = nn.BatchNorm2d(out_channels) if norm else None

    def _fuse_norm(self, weight, bias, norm):
        running_mean = norm.running_mean
        running_var = norm.running_var
        gamma = norm.weight
        beta = norm.bias
        eps = norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        bias_fusd = beta - running_mean * gamma / std
        bias_fusd = bias_fusd if bias is None else bias_fusd + bias * gamma / std
        weight_fusd = weight * t
        return weight_fusd, bias_fusd

    @property
    def conv_wb(self):
        return self._fuse_norm(self.conv.weight, self.conv.bias, self.norm)

    @property
    def conv_1_wb(self):
        return self._fuse_norm(self.conv_1.weight, self.conv_1.bias, self.norm_1)

    @property
    def shortcut_wb(self):
        weight = torch.eye(self.conv_1.weight.size(1))[:, :, None, None]
        weight = weight.repeat(self.conv_1.groups, 1, 1, 1)
        return self._fuse_norm(weight, None, self.norm_sc)

    def forward(self, x):
        if self.training:
            out = self.norm(self.conv(x)) + self.norm_1(self.conv_1(x))
            out = out + self.norm_sc(x) if self.has_shortcut else out
        else:
            out = self.conv_eq(x)
        return self.act(out) if self.act else out

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups)

    @staticmethod
    def convert(rc):
        if isinstance(rc, RCpaA):
            return rc
        elif isinstance(rc, CNA):
            config = rc.config
            del config['padding']
            rct = RCpaNA(**config).to(rc.conv.weight.device)
            rct._init_wb(rc.conv.weight, rc.conv.bias)
            rct.act = rc.act
            rct.norm = rc.norm
            return rct
        else:
            raise Exception('err module ' + rc.__class__.__name__)


class RCk3NA(RCpaNA):
    def __init__(self, in_channels: int, out_channels: int, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(RCk3NA, self).__init__(
            in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, groups=groups, norm=norm,
            act=act,
            device=device, dtype=dtype)


class RCk3s1NA(RCk3NA):
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None, dtype=None):
        super(RCk3s1NA, self).__init__(
            in_channels, out_channels, stride=1, dilation=dilation, groups=groups, norm=norm, act=act, device=device,
            dtype=dtype)


# if __name__ == '__main__':
#     layer = RCpaB(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation=1, groups=3, )
#     print(layer.training)
#     layer.eval()
#     x = torch.rand(size=(1, 6, 5, 5), dtype=torch.float32)
#     y1 = layer(x)
#     y2 = layer.conv_eq(x)
#     print(y1 - y2)


# </editor-fold>

# <editor-fold desc='Dw卷积子模块'>
class DC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 padding: _int2 = 1, dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None,
                 dtype=None):
        super(DC, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=False, groups=groups, device=device, dtype=dtype)
        self.conv = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
            bias=bias, groups=1, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        return x

    @property
    def conv_eq(self):
        bias = self.conv.bias is not None
        _conv_eq = nn.Conv2d(in_channels=self.conv.in_channels,
                             out_channels=self.conv.out_channels,
                             kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                             padding=self.conv.padding, dilation=self.conv.dilation,
                             groups=self.conv.groups, bias=bias)
        _conv_eq.weight.data = self.conv.weight.data * self.conv_1.weight.data
        if bias:
            _conv_eq.bias.data = self.conv.bias.data
        return _conv_eq

    @property
    def config(self):
        return dict(in_channels=self.conv_1.in_channels, out_channels=self.conv_1.out_channels,
                    kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv_1.groups)


class DCA(DC):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 padding: _int2 = 1, dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None, device=None,
                 dtype=None):
        super(DCA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        x = self.act(x) if self.act else x
        return x


class DCB(DC):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 padding: _int2 = 1, dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, device=None,
                 dtype=None):
        super(DCB, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not norm, groups=groups, device=device, dtype=dtype)
        self.norm = NORM.build(out_channels, norm)

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        x = self.norm(x) if self.norm else x
        return x


class DCNA(DCA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 padding: _int2 = 1, dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None,
                 dtype=None):
        super(DCNA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not norm, groups=groups, act=act, device=device, dtype=dtype)
        self.norm = NORM.build(out_channels, norm)

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        x = self.norm(x) if self.norm else x
        x = self.act(x) if self.act else x
        return x


class DCpa(DC):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, device=None, dtype=None):
        super(DCpa, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), bias=bias, device=device,
            dtype=dtype)


class DCpaNA(DCNA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, norm=NORM.BATCH, act=None, device=None,
                 dtype=None):
        super(DCpaNA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), norm=norm, act=act,
            device=device,
            dtype=dtype)


class DCpaA(DCA):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _int2 = 3, stride: _int2 = 1,
                 dilation: _int2 = 1, groups: _int2 = 1, bias: bool = True, act=None, device=None,
                 dtype=None):
        super(DCpaA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=(kernel_size - 1) * dilation // 2, act=act, bias=bias,
            device=device, dtype=dtype)


# </editor-fold>

# <editor-fold desc='FFT卷积子模块'>
class FC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: _int2 = 1, bias: bool = True, device=None,
                 dtype=None):
        super(FC, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2 * in_channels, out_channels=2 * out_channels, kernel_size=(1, 1),
            stride=(1, 1), padding=(0, 0), dilation=(1, 1),
            bias=bias, groups=groups, device=device, dtype=dtype)

    @staticmethod
    def fft_apply(x, func):
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((x.size(0), -1,) + ffted.size()[3:])

        ffted = func(ffted)

        ffted = ffted.view((x.size(0), -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm='ortho')
        return output

    def forward(self, x):
        return FC.fft_apply(x, self.conv)


class FCA(FC):
    def __init__(self, in_channels: int, out_channels: int, groups: _int2 = 1, bias: bool = True, device=None,
                 dtype=None, act=None, ):
        super(FCA, self).__init__(in_channels=in_channels, out_channels=out_channels, groups=groups,
                                  bias=bias, device=device, dtype=dtype, )
        self.act = ACT.build(act)

    def _forward_spec(self, x):
        x = self.conv(x)
        x = self.act(x) if self.act else x
        return x

    def forward(self, x):
        return FC.fft_apply(x, self._forward_spec)


class FCNA(FCA):
    def __init__(self, in_channels: int, out_channels: int, groups: _int2 = 1, device=None,
                 dtype=None, act=None, norm=NORM.BATCH):
        super(FCNA, self).__init__(in_channels=in_channels, out_channels=out_channels, groups=groups,
                                   bias=not norm, device=device, dtype=dtype, act=act)
        self.norm = nn.BatchNorm2d(out_channels * 2) if norm else None

    def _forward_spec(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm else x
        x = self.act(x) if self.act else x
        return x

    def forward(self, x):
        return FC.fft_apply(x, self._forward_spec)


# </editor-fold>

# <editor-fold desc='模型替换'>

RC_MAPPER = {CNA: RCpaNA.convert, Ck1s1NA: RCpaNA.convert, Ck3s1NA: RCpaNA.convert, CpaNA: RCpaNA.convert,
             Ck3NA: RCpaNA.convert, Ck1NA: RCpaNA.convert,
             CA: RCpaA.convert, Ck1s1A: RCpaA.convert, Ck3s1A: RCpaA.convert, CpaA: RCpaA.convert,
             C: RCpa.convert, Ck1s1: RCpa.convert, Cpa: RCpa.convert, }


def model_apply(model: nn.Module, module_mapper: dict, show_detial: bool = False):
    for name, sub_module in model.named_children():
        if sub_module.__class__ in module_mapper.keys():
            func = module_mapper[sub_module.__class__]
            if isinstance(func, nn.Module):
                sub_module_replalce = func
            else:
                sub_module_replalce = func(sub_module)
            if show_detial:
                print(sub_module.__class__.__name__ + ' [ '
                      + func.__class__.__name__ + ' ]-> ' + sub_module_replalce.__class__.__name__)
            setattr(model, name, sub_module_replalce)
        else:
            model_apply(sub_module, module_mapper, show_detial)
    return model


def model_react(model, act_old, act_new):
    func_dct = {ACT.build(act_old).__class__: lambda x: ACT.build(act_new)}
    model_apply(model, func_dct)
    return model


def model_rc2c(model):
    func_dct = {RCpa: C.convert, RCpaA: CA.convert, RCpaNA: CNA.convert, }
    model_apply(model, func_dct)
    return model


def model_c2rc(model):
    model_apply(model, RC_MAPPER)
    return model


def model_dc2c(model):
    func_dct = {DC: C.convert, DCA: CA.convert, DCNA: CNA.convert, }
    model_apply(model, func_dct)
    return model


def model_spectral_norm(model, show_detial: bool = False):
    func_dct = {nn.Conv2d: torch.nn.utils.spectral_norm}
    model_apply(model, func_dct, show_detial)
    return model


# </editor-fold>

# <editor-fold desc='检测框层'>
# 基于基础尺寸和长宽比生成anchor_sizes
def generate_anchor_sizes(base, scales=(8, 16, 32), wh_ratios=(0.5, 1, 2)):
    wh_ratios = torch.sqrt(torch.Tensor(wh_ratios))[None, :]
    scales = torch.Tensor(scales)[:, None]
    ws = scales * wh_ratios * base
    hs = scales / wh_ratios * base
    anchor_sizes = torch.stack([ws, hs], dim=-1)
    anchor_sizes = anchor_sizes.reshape(-1, 2)
    return anchor_sizes


def _generate_grid(Wf, Hf):
    x = torch.arange(Wf)[None, :].expand(Hf, Wf)
    y = torch.arange(Hf)[:, None].expand(Hf, Wf)
    xy_offset = torch.stack([x, y], dim=2)  # (Hf,Wf,2)
    return xy_offset


def _calc_feat_size(img_size, stride):
    (W, H) = img_size if isinstance(img_size, Iterable) else (img_size, img_size)
    return (int(math.ceil(W / stride)), int(math.ceil(H / stride)))


class PointAnchorLayer(nn.Module):
    def __init__(self, stride, feat_size=(0, 0)):
        super(PointAnchorLayer, self).__init__()
        self.stride = stride
        self.feat_size = feat_size

    @property
    def feat_size(self):
        return (self.Wf, self.Hf)

    @property
    def num_anchor(self):
        return self.Wf * self.Hf

    @property
    def anchors(self):
        anchors = torch.cat([self.xy_offset, self.xy_offset + 1], dim=-1) * self.stride
        anchors = torch.reshape(anchors, shape=(self.Wf * self.Hf, 4))
        return anchors

    @property
    def cens(self):
        return torch.reshape((self.xy_offset + 0.5) * self.stride, shape=(self.Wf * self.Hf, 2))

    @feat_size.setter
    def feat_size(self, feat_size):
        self.Wf, self.Hf = _pair(feat_size)
        self.xy_offset = _generate_grid(self.Wf, self.Hf)


# 有先验框
class AnchorLayer(nn.Module):
    def __init__(self, anchor_sizes, stride, feat_size=(0, 0)):
        super(AnchorLayer, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.stride = stride
        self.feat_size = feat_size

    @property
    def anchor_sizes(self):
        return self._anchor_sizes

    @anchor_sizes.setter
    def anchor_sizes(self, anchor_sizes):
        anchor_sizes = torch.Tensor(anchor_sizes)
        self._anchor_sizes = anchor_sizes
        self.Na = anchor_sizes.size(0)

    @property
    def feat_size(self):
        return (self.Wf, self.Hf)

    @property
    def num_anchor(self):
        return self.Na * self.Wf * self.Hf

    @property
    def anchors(self):
        anchors = torch.cat([(self.xy_offset + 0.5) * self.stride, self.wh_offset], dim=-1)
        anchors = xywhsT2xyxysT(anchors)
        return anchors

    @feat_size.setter
    def feat_size(self, feat_size):
        self.Wf, self.Hf = _pair(feat_size)
        self.xy_offset = _generate_grid(self.Wf, self.Hf)
        self.xy_offset = self.xy_offset[:, :, None, :]. \
            expand(self.Hf, self.Wf, self.Na, 2).contiguous()
        self.wh_offset = self.anchor_sizes[None, None, :, :]. \
            expand(self.Hf, self.Wf, self.Na, 2).contiguous()


class RotationalAnchorLayer(AnchorLayer):
    def __init__(self, anchor_sizes, alphas, stride, feat_size=(0, 0)):
        AnchorLayer.__init__(self, anchor_sizes=anchor_sizes, stride=stride, feat_size=feat_size)
        self.alphas = alphas

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, alphas):
        alphas = torch.Tensor(alphas)
        self._alphas = alphas
        self.a_offset = torch.Tensor(self.alphas)[..., None].expand(self.Hf, self.Wf, self.Na, 1)

    @property
    def anchors(self):
        anchors = torch.cat([(self.xy_offset + 0.5) * self.stride, self.wh_offset, self.a_offset], dim=-1)
        return anchors


class RotationalAnchorImgLayer(RotationalAnchorLayer):
    def __init__(self, anchor_sizes, alphas, stride, img_size=(0, 0)):
        super().__init__(anchor_sizes, alphas, stride, _calc_feat_size(img_size, stride))

    @property
    def img_size(self):
        return (self.Wf * self.stride, self.Hf * self.stride)

    @img_size.setter
    def img_size(self, img_size):
        self.feat_size = _calc_feat_size(img_size, self.stride)


class AnchorImgLayer(AnchorLayer):
    def __init__(self, anchor_sizes, stride, img_size=(0, 0)):
        super().__init__(anchor_sizes, stride, _calc_feat_size(img_size, stride))

    @property
    def img_size(self):
        return (self.Wf * self.stride, self.Hf * self.stride)

    @img_size.setter
    def img_size(self, img_size):
        self.feat_size = _calc_feat_size(img_size, self.stride)


class PointAnchorImgLayer(PointAnchorLayer):
    def __init__(self, stride, img_size=(0, 0)):
        super().__init__(stride, _calc_feat_size(img_size, stride))

    @property
    def img_size(self):
        return (self.Wf * self.stride, self.Hf * self.stride)

    @img_size.setter
    def img_size(self, img_size):
        self.feat_size = _calc_feat_size(img_size, self.stride)


# </editor-fold>

# <editor-fold desc='初始化'>

def init_sig(bias, prior_prob=0.1):
    nn.init.constant_(bias.data, -math.log((1 - prior_prob) / prior_prob))
    return bias


def init_xavier(weight):
    """Caffe2 XavierFill Implementation"""
    fan_in = weight.numel() / weight.size(0)
    scale = math.sqrt(3 / fan_in)
    nn.init.uniform_(weight, -scale, scale)
    return weight


def init_msra(weight):
    """Caffe2 MSRAFill Implementation"""
    fan_in = weight.numel() / weight.size(0)
    scale = math.sqrt(2 / fan_in)
    nn.init.uniform_(weight, -scale, scale)
    return weight


# </editor-fold>

# <editor-fold desc='自定义loss'>

# def binaryfocal_loss(pred, target, alpha=0.5, gamma=2, reduction='sum'):
#     loss_ce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
#     prop = target * pred + (1 - target) * (1 - pred)
#     alpha_full = target * alpha + (1 - target) * (1 - alpha)
#     loss = loss_ce * alpha_full * (1 - prop) ** gamma
#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     elif reduction == 'none':
#         return loss
#     else:
#         raise Exception('err reduction')

def _reduct_loss(loss, reduction='sum'):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise Exception('err reduction')


def binary_focal_loss_with_logits(pred, target, alpha=0.5, gamma=2, reduction='sum'):
    pred_prob = torch.sigmoid(pred)  # prob from logits
    pt = target * pred_prob + (1 - target) * (1 - pred_prob)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = (1 - pt) ** gamma
    focal_weight = alpha_factor * modulating_factor
    loss = F.binary_cross_entropy_with_logits(input=pred, target=target, weight=focal_weight, reduction=reduction)
    return loss


def binary_focal_loss(pred, target, alpha=0.5, gamma=2, reduction='sum', weight=None):
    pt = target * pred + (1 - target) * (1 - pred)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = (1 - pt) ** gamma
    focal_weight = alpha_factor * modulating_factor
    focal_weight = focal_weight if weight is None else focal_weight * weight.detach()
    loss = F.binary_cross_entropy(pred, target, reduction='none')
    loss = _reduct_loss(loss * focal_weight, reduction)
    return loss


def binary_qfocal_loss(pred, target, alpha=0.5, gamma=2, reduction='sum'):
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.abs(pred - target) ** gamma
    focal_weight = alpha_factor * modulating_factor
    loss = F.binary_cross_entropy(pred, target, weight=focal_weight, reduction=reduction)
    return loss


# 默认最后一个维度是分类，pred和target同大小
def focal_loss(pred, target, alphas=(1, 2, 3), gamma=2, reduction='sum'):
    pred_sft = torch.softmax(pred, dim=-1)
    focal_weight = (1 - pred_sft).pow(gamma)
    if alphas is not None:
        focal_weight = focal_weight * torch.Tensor(alphas).to(pred.device)
    loss = -torch.sum(torch.log(pred_sft + 1e-7) * target * focal_weight, dim=-1)
    return _reduct_loss(loss, reduction)


def distribute_loss(dls_reg: torch.Tensor, dls_tg: torch.Tensor, reduction='mean'):
    max_reg = dls_reg.size(-1) - 1
    dls_tg_low = dls_tg.long()
    dls_tg_high = dls_tg_low + 1
    dls_dt_low = dls_tg - dls_tg_low
    dls_dt_high = 1 - dls_dt_low
    loss_low = F.cross_entropy(dls_reg, dls_tg_low.clamp(min=0, max=max_reg), reduction='none') * dls_dt_high
    loss_high = F.cross_entropy(dls_reg, dls_tg_high.clamp(min=0, max=max_reg), reduction='none') * dls_dt_low

    dist_ori = torch.stack([dls_dt_low, dls_dt_high], dim=-1)
    norm = torch.sum(-torch.log(dist_ori.clamp(min=1e-7)) * dist_ori, dim=-1)
    return _reduct_loss(loss_low + loss_high - norm, reduction)


def dlslog_loss(dls_pd: torch.Tensor, dls_tg: torch.Tensor, reduction='mean', weight=None) -> torch.Tensor:
    dls_min = torch.where(dls_pd >= dls_tg, dls_tg, dls_pd)
    dls_max = torch.where(dls_pd <= dls_tg, dls_tg, dls_pd)
    loss = torch.log((dls_max / dls_min.clamp(min=1e-7)).clamp(min=1e-2, max=1e2))
    if weight is not None:
        loss = loss * weight
    return _reduct_loss(loss, reduction)


# </editor-fold>


class TM(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = CpaNA(4, 5, kernel_size=3, act=ACT.RELU)
        self.c2 = CpaNA(5, 6, kernel_size=3, act=ACT.LK)

    def forward(self, x):
        return self.c2(self.c1(x))

# if __name__ == '__main__':
#     model = TM()
#     model.eval()
#     x = torch.rand(1, 4, 10, 10) * 10
#     # print(model)
#     y1 = model(x)
#     w1 = copy.deepcopy(model.c1.conv.weight)
#     # model_dc2c(model)
#     model_c2rc(model)
#     # model_react(model, ACT.RELU, ACT.SWISH)
#     # print(model)
#     model.eval()
#     y2 = model(x)
#     model.eval()
#     w2 = copy.deepcopy(model.c1.conv_eq.weight)
#     # y2 = model(x)
#     # y3 = model(x)
#     print(y1 - y2)
#     # print(y3 - y2)
#     # print(w2 - w1)
