import warnings
from typing import Iterable, Union, List

import PIL
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from utils.file import DEVICE

warnings.filterwarnings("ignore")


# <editor-fold desc='numpy和torch转化'>
def arrsN2arrsT(arrsN: Union[np.ndarray, torch.Tensor, Iterable], device: torch.device = DEVICE) \
        -> Union[torch.Tensor, Iterable]:
    if isinstance(arrsN, torch.Tensor):
        return arrsN.to(device)
    elif isinstance(arrsN, np.ndarray):
        arrsN = torch.as_tensor(arrsN).to(device)
        if arrsN.dtype == torch.float64:
            arrsN = arrsN.float()
        if arrsN.dtype == torch.int32:
            arrsN = arrsN.long()
        return arrsN
    elif isinstance(arrsN, dict):
        for key in arrsN.keys():
            arrsN[key] = arrsN2arrsT(arrsN[key], device=device)
        return arrsN
    elif isinstance(arrsN, list) or isinstance(arrsN, tuple):
        arrsN = list(arrsN)
        for i in range(len(arrsN)):
            arrsN[i] = arrsN2arrsT(arrsN[i], device=device)
        return arrsN
    else:
        raise Exception('err')


def arrsT2arrsN(arrsT: Union[np.ndarray, torch.Tensor, Iterable]) -> Union[np.ndarray, Iterable]:
    if isinstance(arrsT, np.ndarray):
        return arrsT
    elif isinstance(arrsT, torch.Tensor):
        return arrsT.detach().cpu().numpy()
    elif isinstance(arrsT, dict):
        for key in arrsT.keys():
            arrsT[key] = arrsT2arrsN(arrsT[key])
        return arrsT
    elif isinstance(arrsT, Iterable):
        arrsT = list(arrsT)
        for i in range(len(arrsT)):
            arrsT[i] = arrsT2arrsN(arrsT[i])
        return arrsT
    else:
        raise Exception('err')


# </editor-fold>

# <editor-fold desc='numpy水平边界'>
CORNERSN = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])


def xywhN2xyxyN(xywhN: np.ndarray) -> np.ndarray:
    xcyc, wh_2 = xywhN[:2], xywhN[2:4] / 2
    return np.concatenate([xcyc - wh_2, xcyc + wh_2], axis=0)


def xywhsN2xyxysN(xywhsN: np.ndarray) -> np.ndarray:
    xcyc, wh_2 = xywhsN[..., :2], xywhsN[..., 2:4] / 2
    return np.concatenate([xcyc - wh_2, xcyc + wh_2], axis=-1)


def xyxyN2xywhN(xyxyN: np.ndarray) -> np.ndarray:
    x1y1, x2y2 = xyxyN[:2], xyxyN[2:4]
    return np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=0)


def xyxyN2areaN(xyxyN: np.ndarray) -> np.ndarray:
    return np.prod(xyxyN[2:4] - xyxyN[:2])


def xyxysN2areasN(xyxysN: np.ndarray) -> np.ndarray:
    return np.prod(xyxysN[..., 2:4] - xyxysN[..., :2], axis=-1)


def xyxysN2xywhsN(xyxysN: np.ndarray) -> np.ndarray:
    x1y1, x2y2 = xyxysN[..., :2], xyxysN[..., 2:4]
    return np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=-1)


def xyxyN2xlylN(xyxyN: np.ndarray) -> np.ndarray:
    xlyl = np.stack([xyxyN[[0, 0, 2, 2]], xyxyN[[1, 3, 3, 1]]], axis=1)
    return xlyl


def xyxysN2xlylsN(xyxysN: np.ndarray) -> np.ndarray:
    xlyls = np.stack([xyxysN[..., [0, 0, 2, 2]], xyxysN[..., [1, 3, 3, 1]]], axis=-1)
    return xlyls


def xywhN2xlylN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[:2] + CORNERSN * xywhN[2:4] / 2


def xywhsN2xlylsN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[..., None, :2] + CORNERSN * xywhN[..., None, 2:4] / 2


# </editor-fold>

# <editor-fold desc='numpy旋转边界'>

def aN2matN(aN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(aN), np.sin(aN)
    mat = np.stack([np.stack([cos, sin], axis=0), np.stack([-sin, cos], axis=0)], axis=0)
    return mat


def asN2matsN(asN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(asN), np.sin(asN)
    mat = np.stack([np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)], axis=-2)
    return mat


def abN2matN(abN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(abN), np.sin(abN)
    mat = np.stack([cos, sin], axis=1)
    return mat


def absN2matsN(absN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(absN), np.sin(absN)
    mat = np.stack([cos, sin], axis=-1)
    return mat


def uvN2matN(uvN: np.ndarray) -> np.ndarray:
    mat = np.array([uvN, [-uvN[1], uvN[0]]])
    return mat


def uvsN2matsN(uvsN: np.ndarray) -> np.ndarray:
    mats = np.stack([uvsN, np.stack([-uvsN[..., 1], uvsN[..., 0]], axis=-1)], axis=-2)
    return mats


def uvN2aN(uvN: np.ndarray) -> np.ndarray:
    return np.arctan2(uvN[1], uvN[0])


def uvsN2asN(uvsN: np.ndarray) -> np.ndarray:
    return np.arctan2(uvsN[..., 1], uvsN[..., 0])


def asN2uvsN(asN: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(asN), np.sin(asN)], axis=-1)


def aN2uvN(aN: np.ndarray) -> np.ndarray:
    return np.array([np.cos(aN), np.sin(aN)])


def xywhaN2x1y1whaN(xywhaN: np.ndarray) -> np.ndarray:
    mat = aN2matN(xywhaN[4])
    return np.concatenate([xywhaN[:2] - xywhaN[2:4] @ mat / 2, xywhaN[2:5]], axis=0)


def xywhasN2x1y1whasN(xywhasN: np.ndarray) -> np.ndarray:
    mats = asN2matsN(xywhasN[..., 4])
    wh_2 = xywhasN[..., 2:4, None] / 2
    return np.concatenate([xywhasN[..., :2] - wh_2 @ mats, xywhasN[..., 2:5]], axis=-1)


def xywhaN2xlylN(xywhaN: np.ndarray) -> np.ndarray:
    mat = aN2matN(xywhaN[4])
    xlyl = xywhaN[:2] + (CORNERSN * xywhaN[2:4] / 2) @ mat
    return xlyl


def xywhasN2xlylsN(xywhasN: np.ndarray) -> np.ndarray:
    mats = asN2matsN(xywhasN[..., 4])
    xlyls = xywhasN[..., None, :2] + (CORNERSN * xywhasN[..., None, 2:4] / 2) @ mats
    return xlyls


def xywhuvN2xlylN(xywhuvN: np.ndarray) -> np.ndarray:
    mat = uvN2matN(xywhuvN[4:6])
    xlyl = xywhuvN[:2] + (CORNERSN * xywhuvN[2:4] / 2) @ mat
    return xlyl


def xywhuvsN2xlylsN(xywhuvsN: np.ndarray) -> np.ndarray:
    mats = uvsN2matsN(xywhuvsN[..., 4:6])
    xlyls = xywhuvsN[..., None, :2] + (CORNERSN * xywhuvsN[..., None, 2:4] / 2) @ mats
    return xlyls


def xlylN2xyxyN(xlylN: np.ndarray) -> np.ndarray:
    if xlylN.shape[0] == 0:
        return np.zeros(shape=4)
    else:
        x1y1 = np.min(xlylN, axis=0)
        x2y2 = np.max(xlylN, axis=0)
        return np.concatenate([x1y1, x2y2], axis=0)


def xlylsN2xyxysN(xlylsN: np.ndarray) -> np.ndarray:
    if xlylsN.shape[-2] == 0:
        return np.zeros(shape=list(xlylsN.shape[:-2]) + [4])
    else:
        x1y1 = np.min(xlylsN, axis=-2)
        x2y2 = np.max(xlylsN, axis=-2)
        return np.concatenate([x1y1, x2y2], axis=-1)


def xywhaN2xyxyN(xywhaN: np.ndarray) -> np.ndarray:
    return xlylN2xyxyN(xywhaN2xlylN(xywhaN))


def xywhasN2xyxysN(xywhasN: np.ndarray) -> np.ndarray:
    return xlylsN2xyxysN(xywhasN2xlylsN(xywhasN))


def xywhabN2xlylN(xywhabN: np.ndarray) -> np.ndarray:
    mat = abN2matN(xywhabN[4:6])
    xlyl = xywhabN[:2] + (CORNERSN * xywhabN[2:4] / 2) @ mat
    return xlyl


def xywhabsN2xlylsN(xywhabsN: np.ndarray) -> np.ndarray:
    mat = absN2matsN(xywhabsN[..., 4:6])
    xlyls = xywhabsN[..., None, :2] + (CORNERSN * xywhabsN[..., None, 2:4] / 2) @ mat
    return xlyls


def xywhabN2xyxyN(xywhabN: np.ndarray) -> np.ndarray:
    return xlylN2xyxyN(xywhabN2xlylN(xywhabN))


def xywhabsN2xyxysN(xywhabsN: np.ndarray) -> np.ndarray:
    return xlylsN2xyxysN(xywhabsN2xlylsN(xywhabsN))


def xlylN2xywhN(xlylN: np.ndarray) -> np.ndarray:
    return xyxyN2xywhN(xlylN2xyxyN(xlylN))


def xlylsN2xywhsN(xlylsN: np.ndarray) -> np.ndarray:
    return xyxysN2xywhsN(xlylsN2xyxysN(xlylsN))


def xywhN2xywhaN(xywhN: np.ndarray, longer_width=True) -> np.ndarray:
    if longer_width and xywhN[3] > xywhN[2]:
        return np.concatenate([xywhN[:2], xywhN[2:4][::-1], [np.pi / 2]], axis=0)
    else:
        return np.concatenate([xywhN, [0]], axis=0)


def xywhsN2xywhasN(xywhsN: np.ndarray, longer_width=True) -> np.ndarray:
    alphas = np.zeros(shape=xywhsN.shape[:-1])
    if longer_width:
        fltr = xywhsN[..., 3] > xywhsN[..., 2]
        alphas = np.where(fltr, alphas + np.pi / 2, alphas)
        ws = np.where(fltr, xywhsN[..., 3], xywhsN[..., 2])
        hs = np.where(fltr, xywhsN[..., 2], xywhsN[..., 3])

        return np.concatenate([xywhsN[..., :2], ws[..., None], hs[..., None], alphas[..., None]], axis=0)
    else:
        return np.concatenate([xywhsN, alphas[..., None]], axis=-1)


def xyxyN2xywhaN(xyxyN: np.ndarray, longer_width=True) -> np.ndarray:
    return xywhN2xywhaN(xyxyN2xywhN(xyxyN), longer_width=longer_width)


def xyxysN2xywhasN(xyxysN: np.ndarray, longer_width=True) -> np.ndarray:
    return xywhsN2xywhasN(xyxysN2xywhsN(xyxysN), longer_width=longer_width)


# </editor-fold>

# <editor-fold desc='numpy旋转边界补充'>
def xlylN_samp(xlylN: np.ndarray, num_samp: int = 1) -> np.ndarray:
    if num_samp == 1:
        return xlylN
    xlylN_rnd = np.concatenate([xlylN[1:], xlylN[:1]], axis=0)
    pows = np.linspace(start=0, stop=1, num=num_samp, endpoint=False)[..., None]
    xlylN_mix = xlylN_rnd[:, None, :] * pows + xlylN[:, None, :] * (1 - pows)
    xlylN_mix = np.reshape(xlylN_mix, newshape=(xlylN.shape[0] * num_samp, 2))
    return xlylN_mix


def xysN2matN(xysN: np.ndarray, powersN: np.ndarray = None) -> np.ndarray:
    if xysN.shape[0] < 2:
        return np.array([1, 0])
    if powersN is not None:
        xysN = xysN * powersN[..., None]
    xysN = xysN - np.mean(xysN, axis=0)
    s, v, d = np.linalg.svd(xysN)
    return d


def xysN2uvN(xysN: np.ndarray, powersN: np.ndarray = None) -> np.ndarray:
    return xysN2matN(xysN, powersN=powersN)[0]


def xysN2aN(xysN: np.ndarray, powersN: np.ndarray = None) -> np.ndarray:
    return uvN2aN(xysN2uvN(xysN, powersN=powersN))


def xlylN2xywhaN(xlylN: np.ndarray, aN: np.ndarray = None) -> np.ndarray:
    if xlylN.shape[0] <= 1:
        return np.zeros(shape=5)
    if aN is None:
        aN = uvN2aN(xysN2uvN(xlylN))
    mat = aN2matN(aN)
    xlyl_proj = xlylN @ mat.T
    xywh_proj = xlylN2xywhN(xlyl_proj)
    xy_cen = xywh_proj[:2] @ mat
    return np.concatenate([xy_cen, xywh_proj[2:4], [aN]], axis=0)


def xlylN2xywhaN_simple(xlylN: np.ndarray) -> np.ndarray:
    vh = xlylN[3] - xlylN[0]
    vw = xlylN[2] - xlylN[3]
    xy = np.mean(xlylN, axis=0)
    a = np.arctan2(vw[1], vw[0])
    w = np.sqrt(np.sum(vw ** 2))
    h = np.sqrt(np.sum(vh ** 2))
    return np.concatenate([xy, [w, h, a]], axis=0)


def xywhaN2xywhuvN(xywhaN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhaN[:4], aN2uvN(xywhaN[4])], axis=0)


def xywhasN2xywhuvsN(xywhasN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhasN[..., :4], asN2uvsN(xywhasN[..., 4])], axis=-1)


def xywhuvN2xywhaN(xywhuvN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhuvN[:4], [uvN2aN(xywhuvN[4:6])]], axis=0)


def xywhuvsN2xywhasN(xywhuvsN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhuvsN[..., :4], uvsN2asN(xywhuvsN[..., 4:6])[..., None]], axis=-1)


def xywhaN2xywhN(xywhaN: np.ndarray) -> np.ndarray:
    mat = aN2matN(xywhaN[4])
    wh = np.sum(np.abs(xywhaN[2:4, None] * mat), axis=0)
    return np.concatenate([xywhaN[:2], wh], axis=0)


def xywhabN2xywhN(xywhabN: np.ndarray) -> np.ndarray:
    mat = abN2matN(xywhabN[4:6])
    wh = np.sum(np.abs(xywhabN[2:4, None] * mat), axis=0)
    return np.concatenate([xywhabN[:2], wh], axis=0)


# </editor-fold>

# <editor-fold desc='numpy转换mask'>
def arange2dN(H: int, W: int) -> (np.ndarray, np.ndarray):
    ys = np.broadcast_to(np.arange(H)[:, None], (H, W))
    xs = np.broadcast_to(np.arange(W)[None, :], (H, W))
    return ys, xs


def create_meshN(H: int, W: int) -> np.ndarray:
    ys, xs = arange2dN(H, W)
    return np.stack([xs, ys], axis=2).reshape(H * W, 2) + 0.5


def create_meshesN(N: int, H: int, W: int) -> np.ndarray:
    ys, xs = arange2dN(H, W)
    meshes = np.broadcast_to(np.stack([xs, ys], axis=2), (N, H, W, 2)).reshape(N, H * W, 2)
    return meshes


def xyxyN2ixysN(xyxyN: np.ndarray, size: tuple) -> np.ndarray:
    xyxyN = xyxyN.astype(np.int32)
    xyxyN = xyxyN_clip(xyxyN, xyxyN_rgn=np.array([0, 0, size[0], size[1]]))
    iys, ixs = arange2dN(xyxyN[3] - xyxyN[1], xyxyN[2] - xyxyN[0])
    ixys = np.stack([ixs, iys], axis=2).reshape(-1, 2) + xyxyN[:2]
    return ixys


def xywhN2ixysN(xywhN: np.ndarray, size: tuple) -> np.ndarray:
    return xyxyN2ixysN(xywhN2xyxyN(xywhN), size=size)


def xlylN2ixysN(xlylN: np.ndarray, size: tuple) -> np.ndarray:
    xyxy = xlylN2xyxyN(xlylN).astype(np.int32)
    xyxy = xyxyN_clip(xyxy, xyxyN_rgn=np.array([0, 0, size[0], size[1]]))
    patch_size = xyxy[2:4] - xyxy[:2]
    maskNb = xlylN2maskNb(xlylN - xyxy[:2], patch_size)
    iys, ixs = np.nonzero(maskNb)
    ixys = np.stack([ixs, iys], axis=1) + xyxy[:2]
    return ixys


def ixysN2xyxyN(ixysN: np.ndarray) -> np.ndarray:
    if ixysN.shape[0] == 0:
        return np.zeros(shape=4, dtype=np.int32)
    else:
        ixysN = ixysN.astype(np.int32)
        xyxy = np.concatenate([np.min(ixysN, axis=0), np.max(ixysN, axis=0) + 1], axis=0)
        return xyxy


def ixysN2xywhN(ixysN: np.ndarray) -> np.ndarray:
    return xyxyN2xywhN(ixysN2xyxyN(ixysN))


def xlylN2abclN(xlylN: np.ndarray) -> np.ndarray:
    xlylN_rnd = np.roll(xlylN, shift=-1, axis=0)
    As = xlylN[:, 1] - xlylN_rnd[:, 1]
    Bs = xlylN_rnd[:, 0] - xlylN[:, 0]
    Cs = xlylN_rnd[:, 1] * xlylN[:, 0] - xlylN[:, 1] * xlylN_rnd[:, 0]
    return np.stack([As, Bs, Cs], axis=1)


def xlylsN2abclsN(xlylsN: np.ndarray) -> np.ndarray:
    xlylsN_rnd = np.roll(xlylsN, shift=-1, axis=-2)
    As = xlylsN[..., 1] - xlylsN_rnd[..., 1]
    Bs = xlylsN_rnd[..., 0] - xlylsN[..., 0]
    Cs = xlylsN_rnd[..., 1] * xlylsN[..., 0] - xlylsN[..., 1] * xlylsN_rnd[..., 0]
    return np.stack([As, Bs, Cs], axis=-1)


def xlylN_clkwise(xlylN: np.ndarray) -> np.ndarray:
    xlylN_rnd = np.roll(xlylN, shift=-1, axis=0)
    area = xlylN_rnd[:, 0] * xlylN[:, 1] - xlylN[:, 0] * xlylN_rnd[:, 1]
    if np.sum(area) < 0:
        xlylN = xlylN[::-1, :]
    return xlylN


def xlylsN_clkwise(xlylsN: np.ndarray) -> np.ndarray:
    xlylsN_rnd = np.roll(xlylsN, shift=-1, axis=-2)
    areas = xlylsN_rnd[..., 0] * xlylsN[..., 1] - xlylsN[..., 0] * xlylsN_rnd[..., 1]
    fltr = np.sum(areas, axis=-1) < 0
    xlylsN = np.where(fltr[..., None, None], xlylsN[..., ::-1, :], xlylsN)
    return xlylsN


def xlylN2areaN(xlylN: np.ndarray) -> np.ndarray:
    xlylN_rnd = np.roll(xlylN, shift=-1, axis=0)
    area = xlylN_rnd[..., 0] * xlylN[..., 1] - xlylN[..., 0] * xlylN_rnd[..., 1]
    area = np.abs(np.sum(area)) / 2
    return area


def xlylsN2areasN(xlylsN: np.ndarray) -> np.ndarray:
    xlylsN_rnd = np.roll(xlylsN, shift=-1, axis=-2)
    areas = xlylsN_rnd[..., 0] * xlylsN[..., 1] - xlylsN[..., 0] * xlylsN_rnd[..., 1]
    areas = np.abs(np.sum(areas, axis=-1) / 2)
    return areas


def xyxyN2maskNb(xyxyN: np.ndarray, size: tuple) -> np.ndarray:
    ys, xs = arange2dN(size[1], size[0])
    maskN = (xs > xyxyN[0]) * (xs < xyxyN[2]) * (ys > xyxyN[1]) * (ys < xyxyN[3])
    return maskN.astype(bool)


def xywhN2maskNb(xywhN: np.ndarray, size: tuple) -> np.ndarray:
    xyxyN = xywhN2xyxyN(xywhN)
    return xyxyN2maskNb(xyxyN, size)


def xywhsN2masksNb(xywhsN: np.ndarray, size: tuple) -> np.ndarray:
    return xyxysN2masksNb(xywhsN2xyxysN(xywhsN), size)


def xyxysN2masksNb(xyxysN: np.ndarray, size: tuple) -> np.ndarray:
    W, H = size
    N = xyxysN.shape[0]
    meshes = create_meshesN(N, H, W)
    maskNb = np.all((meshes < xyxysN[:, None, 2:4]) * (meshes > xyxysN[:, None, :2]), axis=2)
    maskNb = maskNb.reshape(N, H, W)
    return maskNb


def xywhuvN2maskNb(xywhuvN: np.ndarray, size: tuple) -> np.ndarray:
    xlylN = xywhuvN2xlylN(xywhuvN)
    return xlylN2maskNb(xlylN, size)


def xywhaN2maskNb(xywhaN: np.ndarray, size: tuple) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    return xlylN2maskNb(xlylN, size)


def xywhasN2masksN_guss(xywhasN: np.ndarray, size: tuple) -> np.ndarray:
    xywhuvsN = xywhasN2xywhuvsN(xywhasN)
    return xywhuvsN2masksN_guss(xywhuvsN, size)


def xywhuvsN2masksN_guss(xywhuvsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhuvsN.shape[0]
    Wf, Hf = size
    xys, whs, uvs = xywhuvsN[:, None, 0:2], xywhuvsN[:, None, 2:4], xywhuvsN[:, 4:6]
    mats = uvsN2matsN(uvs)

    meshes = create_meshesN(Nb, Hf, Wf) - xys
    meshes_proj = meshes @ mats
    pows_guss = np.exp(-np.sum((meshes_proj * 2 / whs) ** 2, axis=2) / 2)
    pows_guss = pows_guss.reshape(Nb, Hf, Wf)
    return pows_guss


def xyxysN2masksN_guss(xyxysN: np.ndarray, size: tuple) -> np.ndarray:
    return xywhsN2masksN_guss(xyxysN2xywhsN(xyxysN), size)


def xywhsN2masksN_guss(xywhsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhsN.shape[0]
    Wf, Hf = size
    xys, whs = xywhsN[:, None, 0:2], xywhsN[:, None, 2:4]
    meshes = create_meshesN(Nb, Hf, Wf) - xys
    pows_guss = np.exp(-np.sum((meshes * 2 / whs) ** 2, axis=2) / 2)
    pows_guss = pows_guss.reshape(Nb, Hf, Wf)
    return pows_guss


def xywhasN2masksN_cness(xywhasN: np.ndarray, size: tuple) -> np.ndarray:
    xywhuvsN = xywhasN2xywhuvsN(xywhasN)
    return xywhuvsN2masksN_cness(xywhuvsN, size)


def xywhuvsN2masksN_cness(xywhuvsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhuvsN.shape[0]
    Wf, Hf = size
    xys, whs, uvs = xywhuvsN[:, None, 0:2], xywhuvsN[:, None, 2:4], xywhuvsN[:, 4:6]
    mats = uvsN2matsN(uvs)

    meshes = create_meshesN(Nb, Hf, Wf) - xys
    meshes_proj = meshes @ mats
    meshes_proj = np.abs(meshes_proj)

    pows_cness = np.sqrt(np.prod(np.clip(whs / 2 - meshes_proj, a_min=0, a_max=None), axis=2)
                         / np.prod(whs / 2 + meshes_proj, axis=2))
    pows_cness = pows_cness.reshape(Nb, Hf, Wf)
    return pows_cness


def xywhasN2masksNb(xywhasN: np.ndarray, size: tuple) -> np.ndarray:
    xywhuvsN = xywhasN2xywhuvsN(xywhasN)
    return xywhuvsN2masksNb(xywhuvsN, size)


def xywhuvsN2masksNb(xywhuvsN: np.ndarray, size: tuple) -> np.ndarray:
    Nb = xywhuvsN.shape[0]
    Wf, Hf = size
    xys, whs, uvs = xywhuvsN[:, None, 0:2], xywhuvsN[:, None, 2:4], xywhuvsN[:, 4:6]
    mats = uvsN2matsN(uvs)
    meshes = create_meshesN(Nb, Hf, Wf) - xys
    meshes_proj = meshes @ mats
    meshes_proj = np.abs(meshes_proj)
    masksNb = np.all(meshes_proj < whs / 2, axis=2)
    masksNb = masksNb.reshape(Nb, Hf, Wf)
    return masksNb


def xlylN2maskNb_convex(xlylN: np.ndarray, size: tuple) -> np.ndarray:
    maskN = np.zeros(shape=(size[1], size[0]), dtype=bool)
    if xlylN.shape[0] >= 3:
        abcl = xlylN2abclN(xlylN)
        xyxy = np.round(xlylN2xyxyN(xlylN)).astype(np.int32)
        xyxy = xyxyN_clip(xyxy, np.array([0, 0, size[0], size[1]]))
        xs = np.arange(xyxy[0], xyxy[2])[None, :, None] + 0.5
        ys = np.arange(xyxy[1], xyxy[3])[:, None, None] + 0.5
        maskN_ref = (xs * abcl[..., 0] + ys * abcl[..., 1] + abcl[..., 2]) >= 0
        maskN_ref = np.all(maskN_ref, axis=2) + np.all(~maskN_ref, axis=2)
        maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] += maskN_ref
    return maskN


# 带距离权重的内点
def xlylN2maskN_dpow(xlylN: np.ndarray, size: tuple, normlize: bool = True) -> np.ndarray:
    maskN = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
    if xlylN.shape[0] >= 3:
        abcl = xlylN2abclN(xlylN)
        xyxy = np.round(xlylN2xyxyN(xlylN)).astype(np.int32)
        xyxy = xyxyN_clip(xyxy, np.array([0, 0, size[0], size[1]]))
        xs = np.arange(xyxy[0], xyxy[2])[None, :, None] + 0.5
        ys = np.arange(xyxy[1], xyxy[3])[:, None, None] + 0.5

        norm = np.linalg.norm(abcl[..., 0:2], axis=-1)
        dist = (xs * abcl[..., 0] + ys * abcl[..., 1] + abcl[..., 2]) / norm
        dist = np.where(np.isnan(dist), 0, dist)
        dist_pos = np.all(dist > 0, axis=-1) * np.min(dist, axis=-1)
        dist_neg = np.all(dist < 0, axis=-1) * np.min(-dist, axis=-1)
        pow = dist_neg + dist_pos
        if normlize and pow.size > 0:
            pow = pow / np.clip(np.max(pow), a_min=1e-7, a_max=None)
        maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] += pow
    return maskN.astype(np.float32)


def xlylN2maskNb(xlylN: np.ndarray, size: tuple) -> np.ndarray:
    maskNb = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
    if size[1] == 0 or size[0] == 0:
        return maskNb.astype(bool)
    if xlylN.shape[0] >= 3:
        cv2.fillPoly(maskNb, [xlylN.astype(np.int32)], color=1.0)
    return maskNb.astype(bool)


def xlylNs2maskNb(xlylNs: List[np.ndarray], size: tuple) -> np.ndarray:
    maskNb = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
    xlylNs = [xlylN.astype(np.int32) for xlylN in xlylNs]
    cv2.fillPoly(maskNb, xlylNs, color=1.0)
    return maskNb.astype(bool)


# </editor-fold>

# <editor-fold desc='numpy多边形'>

def abclN_intersect(abcl1N: np.ndarray, abcl2N: np.ndarray) -> np.ndarray:
    norm = abcl1N[:, 0] * abcl2N[:, 1] - abcl1N[:, 1] * abcl2N[:, 0]
    x = (abcl1N[:, 1] * abcl2N[:, 2] - abcl1N[:, 2] * abcl2N[:, 1]) / norm
    y = -(abcl1N[:, 0] * abcl2N[:, 2] - abcl1N[:, 2] * abcl2N[:, 0]) / norm
    return np.stack([x, y], axis=1)


def abclsN_intersect(abcls1N: np.ndarray, abcls2N: np.ndarray) -> np.ndarray:
    norm = abcls1N[..., 0] * abcls2N[..., 1] - abcls1N[..., 1] * abcls2N[..., 0]
    x = (abcls1N[..., 1] * abcls2N[..., 2] - abcls1N[..., 2] * abcls2N[..., 1]) / norm
    y = -(abcls1N[..., 0] * abcls2N[..., 2] - abcls1N[..., 2] * abcls2N[..., 0]) / norm
    return np.stack([x, y], axis=-1)


def isin_arr_abclsN(xysN: np.ndarray, abclsN: np.ndarray) -> np.ndarray:
    cross = xysN[..., 0] * abclsN[..., 0] + xysN[..., 1] * abclsN[..., 1] + abclsN[..., 2]
    fltr = np.all(cross >= 0, axis=-1) + np.all(cross <= 0, axis=-1)
    return fltr


def isin_arr_xlylsN(xysN: np.ndarray, xlylsN: np.ndarray, eps=1e-7) -> np.ndarray:
    xlylsN_rnd = np.concatenate([xlylsN[..., 1:, :], xlylsN[..., :1, :]], axis=-2)
    vs = xlylsN_rnd - xlylsN
    rs = xysN[..., None, :] - xlylsN
    cross = vs[..., 0] * rs[..., 1] - vs[..., 1] * rs[..., 0]
    fltr = np.all(cross >= -eps, axis=-1) + np.all(cross <= eps, axis=-1)
    return fltr


def isin_arr_xywhasN(xysN: np.ndarray, xywhasN: np.ndarray, eps=1e-7) -> np.ndarray:
    mat = asN2matsN(-xywhasN[..., 4])
    xysN_porj = mat @ (xysN - xywhasN[..., :2])[..., None]
    fltr = np.all(np.abs(xysN_porj[..., 0]) <= xywhasN[..., 2:4] / 2 + eps, axis=-1)
    return fltr


def isin_arr_xyxysN(xysN: np.ndarray, xyxysN: np.ndarray, eps=1e-7) -> np.ndarray:
    fltr = np.all((xysN >= xyxysN[..., :2] - eps) * (xysN <= xyxysN[..., 2:4] + eps), axis=-1)
    return fltr


def isin_arr_xywhsN(xysN: np.ndarray, xywhsN: np.ndarray, eps=1e-7) -> np.ndarray:
    fltr = np.all(np.abs(xysN - xywhsN[..., :2]) <= xywhsN[..., 2:4] / 2 + eps, axis=-1)
    return fltr


def xlylN_intersect_coreN(xlyl1N: np.ndarray, xlyl2N: np.ndarray, eps: float = 1e-7) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    abcl1 = xlylN2abclN(xlyl1N)[:, None, :]
    abcl2 = xlylN2abclN(xlyl2N)[None, :, :]
    r1 = np.sum(xlyl1N[:, None, :] * abcl2[..., :2], axis=-1) + abcl2[..., 2]
    r2 = np.sum(xlyl2N * abcl1[..., :2], axis=-1) + abcl1[..., 2]
    msk_1pin2 = np.all(r1 >= -eps, axis=-1) + np.all(r1 <= eps, axis=-1)
    msk_2pin1 = np.all(r2 >= -eps, axis=-2) + np.all(r2 <= eps, axis=-2)
    if np.all(msk_1pin2) or np.all(msk_2pin1):
        return np.zeros_like(r1), msk_1pin2, msk_2pin1
    norm = (abcl1[..., 0] * abcl2[..., 1] - abcl1[..., 1] * abcl2[..., 0]).astype(np.float64)
    r1 = r1.astype(np.float64) / norm
    r2 = -r2.astype(np.float64) / norm
    fltr1 = (r1 > -eps) * (r1 < 1 + eps)
    fltr2 = (r2 > -eps) * (r2 < 1 + eps)
    fltr_int = fltr1 * fltr2
    ratio1 = np.where(fltr_int, r1, 0)
    return ratio1, msk_1pin2, msk_2pin1


def intersect_coreN2xlylN(xlyl1N: np.ndarray, xlyl2N: np.ndarray, ratio1: np.ndarray,
                          msk_1pin2: np.ndarray, msk_2pin1: np.ndarray) -> np.ndarray:
    if np.all(msk_1pin2):
        return xlyl1N
    elif np.all(msk_2pin1):
        return xlyl2N
    elif not np.any(ratio1 > 0):
        return np.zeros(shape=(0, 2), dtype=np.float32)
    # 节点排序
    idls1, idls2 = np.nonzero(ratio1 > 0)
    num1 = xlyl1N.shape[0]
    num2 = xlyl2N.shape[0]
    num_int = len(idls1)

    xlyl1N_rnd = np.concatenate([xlyl1N[..., 1:, :], xlyl1N[..., :1, :]], axis=-2)
    ratios = ratio1[idls1, idls2]
    xlyl_int = xlyl1N_rnd[idls1] * ratios[:, None] + xlyl1N[idls1] * (1 - ratios[:, None])
    dists = ratios + idls1
    order = np.argsort(dists)
    idls2 = idls2[order] + 1
    idls1 = idls1 + 1
    xlyl_int = xlyl_int[order]
    # 按序遍历
    idls1_nxt = np.concatenate([idls1[1:], idls1[0:1]])
    idls1_nxt = np.where(idls1_nxt < idls1, idls1_nxt + num1, idls1_nxt)
    idls2_nxt = np.concatenate([idls2[1:], idls2[0:1]])
    idls2_nxt = np.where(idls2_nxt < idls2, idls2_nxt + num2, idls2_nxt)
    ids = [np.zeros(shape=0, dtype=np.int32)]
    for i in range(num_int):
        ids.append([i + num1 + num2])
        ids.append(np.arange(idls1[i], idls1_nxt[i]) % num1)
        ids.append(np.arange(idls2[i], idls2_nxt[i]) % num2 + num1)
    ids = np.concatenate(ids, axis=0)
    pnts = np.concatenate([xlyl1N, xlyl2N, xlyl_int], axis=0)
    msks = np.concatenate([msk_1pin2, msk_2pin1, np.full(shape=num_int, fill_value=True)])
    xlyl_final = pnts[ids][msks[ids]]
    return xlyl_final


def xlylN_intersect(xlyl1: np.ndarray, xlyl2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    xlyl1 = xlylN_clkwise(xlyl1)
    xlyl2 = xlylN_clkwise(xlyl2)
    ratio1, msk_1pin2, msk_2pin1 = xlylN_intersect_coreN(xlyl1, xlyl2, eps=eps)
    xlyl_final = intersect_coreN2xlylN(xlyl1, xlyl2, ratio1, msk_1pin2, msk_2pin1)
    return xlyl_final


def xlylsN_intersect_coresN(xlyls1N: np.ndarray, xlyls2N: np.ndarray, eps: float = 1e-7) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    abcls1 = xlylsN2abclsN(xlyls1N)[..., None, :]
    abcls2 = xlylsN2abclsN(xlyls2N)[..., None, :, :]
    r1 = np.sum(xlyls1N[..., None, :] * abcls2[..., :2], axis=-1) + abcls2[..., 2]
    r2 = np.sum(xlyls2N * abcls1[..., :2], axis=-1) + abcls1[..., 2]
    msk_1pin2 = np.all(r1 >= -eps, axis=-1) + np.all(r1 <= eps, axis=-1)
    msk_2pin1 = np.all(r2 >= -eps, axis=-2) + np.all(r2 <= eps, axis=-2)
    norm = (abcls1[..., 0] * abcls2[..., 1] - abcls1[..., 1] * abcls2[..., 0]).astype(np.float64)
    r1 = r1.astype(np.float64) / norm
    r2 = -r2.astype(np.float64) / norm
    fltr1 = (r1 > -eps) * (r1 < 1 + eps)
    fltr2 = (r2 > -eps) * (r2 < 1 + eps)
    fltr_int = fltr1 * fltr2
    ratio1 = np.where(fltr_int, r1, 0)
    return ratio1, msk_1pin2, msk_2pin1


def xyxyN_clip(xyxyN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xy_max = np.tile(xyxyN_rgn[0:2], 2)
        return np.clip(np.minimum(xyxyN, xy_max), a_min=0, a_max=None)
    else:
        xy_min = np.tile(xyxyN_rgn[0:2], 2)
        xy_max = np.tile(xyxyN_rgn[2:4], 2)
        return np.maximum(np.minimum(xyxyN, xy_max), xy_min)


def xyxysN_clip(xyxysN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xy_max = np.tile(xyxyN_rgn[0:2], 2)
        return np.clip(np.minimum(xyxysN, xy_max), a_min=0, a_max=None)
    else:
        xy_min = np.tile(xyxyN_rgn[0:2], 2)
        xy_max = np.tile(xyxyN_rgn[2:4], 2)
        return np.maximum(np.minimum(xyxysN, xy_max), xy_min)


def xywhN_clip(xywhN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    xyxy = xywhN2xyxyN(xywhN)
    xyxy = xyxyN_clip(xyxy, xyxyN_rgn=xyxyN_rgn)
    xywhN = xyxyN2xywhN(xyxy)
    return xywhN


def xlylN_clip(xlylN: np.ndarray, xyxyN_rgn: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.array([0, 0, xyxyN_rgn[0], xyxyN_rgn[1]])
    xlyl_rgn = xyxyN2xlylN(xyxyN_rgn)
    xlylN_clpd = xlylN_intersect(xlylN, xlyl_rgn, eps=eps)
    return xlylN_clpd


def xysN_clip(xysN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        return np.clip(np.minimum(xysN, xyxyN_rgn[0:2]), a_min=0, a_max=None)
    else:
        return np.maximum(np.minimum(xysN, xyxyN_rgn[2:4]), xyxyN_rgn[0:2])


def xywhaN_clip(xywhaN: np.ndarray, xyxyN_rgn: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if np.any(xywhaN[2:4] == 0):
        return xywhaN
    xlyl = xywhaN2xlylN(xywhaN)
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.array([0, 0, xyxyN_rgn[0], xyxyN_rgn[1]])
    xlyl_rgn = xyxyN2xlylN(xyxyN_rgn)

    ratio1, msk_1pin2, msk_2pin1 = xlylN_intersect_coreN(xlyl, xlyl_rgn, eps=eps)
    idls1, idls2 = np.nonzero(ratio1 > 0)
    xlyl_rnd = np.concatenate([xlyl[..., 1:, :], xlyl[..., :1, :]], axis=-2)
    ratios = ratio1[idls1, idls2]
    xlyl_int = xlyl_rnd[idls1] * ratios[:, None] + xlyl[idls1] * (1 - ratios[:, None])
    pnts = np.concatenate([xlyl_int, xlyl[msk_1pin2], xlyl_rgn[msk_2pin1]], axis=0)
    mat = aN2matN(xywhaN[4])
    pnts_cast = (pnts - xywhaN[:2]) @ mat.T
    if len(pnts_cast) == 0:
        return np.array([xywhaN[0], xywhaN[1], 0, 0, xywhaN[4]])
    w_min = np.min(pnts_cast[:, 0])
    w_max = np.max(pnts_cast[:, 0])
    h_min = np.min(pnts_cast[:, 1])
    h_max = np.max(pnts_cast[:, 1])
    xy = np.array([(w_min + w_max) / 2, (h_min + h_max) / 2]) @ mat + xywhaN[:2]
    xywhaN_clp = np.concatenate([xy, [w_max - w_min, h_max - h_min, xywhaN[4]]])

    return xywhaN_clp


def xlylN2homography(xlylN_src: np.ndarray, xlylN_dst: np.ndarray) -> np.ndarray:
    assert xlylN_src.shape[0] == xlylN_dst.shape[0], 'len err'
    num_vert = xlylN_src.shape[0]
    xxp = xlylN_src * xlylN_dst[:, 0:1]
    yyp = xlylN_src * xlylN_dst[:, 1:2]
    Ax = np.concatenate([xlylN_src, np.ones((num_vert, 1)), np.zeros((num_vert, 3)), -xxp], axis=1)
    Ay = np.concatenate([np.zeros((num_vert, 3)), xlylN_src, np.ones((num_vert, 1)), -yyp], axis=1)
    A, b = np.concatenate([Ax, Ay], axis=0), np.concatenate([xlylN_dst[:, 0:1], xlylN_dst[:, 1:2]], axis=0)
    h = np.linalg.inv(A.T @ A) @ A.T @ b
    H = np.concatenate([h.reshape(-1), [1]], axis=0).reshape((3, 3))
    return H


def xlylN_perspective(xlylN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlyl_ext = np.concatenate([xlylN, np.ones(shape=(xlylN.shape[0], 1))], axis=1)
    xlyl_td = xlyl_ext @ H.T
    return xlyl_td[:, :2] / xlyl_td[:, 2:]


def xyN_perspective(xyN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xyN_ext = np.concatenate([xyN, [1]])
    xyN_trd = H @ xyN_ext
    return xyN_trd[:2] / xyN_trd[2]


def xyxyN_perspective(xyxyN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlylN = xyxyN2xlylN(xyxyN)
    xlylN = xlylN_perspective(xlylN, H=H)
    return xlylN2xyxyN(xlylN)


def xywhN_perspective(xywhN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlylN = xywhN2xlylN(xywhN)
    xlylN = xlylN_perspective(xlylN, H=H)
    return xlylN2xywhN(xlylN)


def xywhaN_perspective(xywhaN: np.ndarray, H: np.ndarray) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    xlylN = xlylN_perspective(xlylN, H=H)
    return xlylN2xywhaN_simple(xlylN)


# </editor-fold>


def xlylT2xyxyT(xlylT: torch.Tensor) -> torch.Tensor:
    if xlylT.size(0) == 0:
        return torch.zeros(size=(4,)).to(xlylT.device)
    else:
        xy1 = torch.min(xlylT, dim=0)[0]
        xy2 = torch.max(xlylT, dim=0)[0]
        return torch.cat([xy1, xy2], dim=0)


def xlylT2xywhT(xlylT: torch.Tensor) -> torch.Tensor:
    return xyxyT2xywhT(xlylT2xyxyT(xlylT))


def xlylT2xywhaT(xlylT: torch.Tensor, aT: torch.Tensor = None) -> torch.Tensor:
    if xlylT.size(0) < 2:
        return torch.zeros(size=(5,)).to(xlylT.device)
    if aT is None:
        aT = uvT2aT(xysT2uvT(xlylT))
    mat = aT2matT(aT)
    xlyl_proj = xlylT @ mat.T
    xywh_proj = xlylT2xywhT(xlyl_proj)
    xy_cen = xywh_proj[:2] @ mat
    return torch.cat([xy_cen, xywh_proj[2:4], aT[None]], dim=0)


def xysT2matT(xysT: torch.Tensor, powersT: torch.Tensor = None) -> torch.Tensor:
    if xysT.size(0) < 2:
        return torch.Tensor([1, 0]).to(xysT.device)
    if powersT is not None:
        xysT = xysT * powersT[..., None]
    xysT = xysT - torch.mean(xysT, dim=0)
    s, v, d = torch.linalg.svd(xysT)
    return d


def xysT2uvT(xysT: torch.Tensor, powersT: torch.Tensor = None) -> torch.Tensor:
    return xysT2matT(xysT, powersT=powersT)[0]


def isin_arr_xlylsT(xysT: torch.Tensor, xlylsT: torch.Tensor, eps=1e-7) -> torch.Tensor:
    xlylsT_rnd = torch.cat([xlylsT[..., 1:, :], xlylsT[..., :1, :]], dim=-2)
    vs = xlylsT_rnd - xlylsT
    rs = xysT[..., None, :] - xlylsT
    cross = vs[..., 0] * rs[..., 1] - vs[..., 1] * rs[..., 0]
    fltr = torch.all(cross >= -eps, dim=-1) + torch.all(cross <= eps, dim=-1)
    return fltr


def isin_arr_xywhasT(xysT: torch.Tensor, xywhasT: torch.Tensor) -> torch.Tensor:
    mat = asN2matsN(-xywhasT[..., 4])
    xysN_porj = mat @ (xysT - xywhasT[..., :2])[..., None]
    fltr = torch.all(torch.abs(xysN_porj[..., 0]) <= xywhasT[..., 2:4] / 2, dim=-1)
    return fltr


def isin_arr_xyxysT(xysT: torch.Tensor, xyxysT: torch.Tensor) -> torch.Tensor:
    fltr = torch.all((xysT >= xyxysT[..., :2]) * (xysT >= xyxysT[..., 2:4]), dim=-1)
    return fltr


def isin_arr_xywhsT(xysT: torch.Tensor, xywhsT: torch.Tensor) -> torch.Tensor:
    fltr = torch.all(torch.abs(xysT - xywhsT[..., :2]) <= xywhsT[..., 2:4] / 2, dim=-1)
    return fltr


# <editor-fold desc='torch分类格式转换'>
def cindT2chotT(cindT: torch.Tensor, num_cls: int) -> torch.Tensor:
    clso = torch.zeros(num_cls)
    clso[cindT] = 1
    return clso


def cindsT2chotsT(cindsT: torch.Tensor, num_cls: int) -> torch.Tensor:
    num = len(cindsT)
    clsos = torch.zeros(num, num_cls)
    clsos[range(num), cindsT] = 1
    return clsos


# </editor-fold>


# <editor-fold desc='torch边界格式转换'>
CORNERST = torch.Tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])


def xyxyT2xywhT(xyxyT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxyT[:2], xyxyT[2:4]
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=0)


def xywhT2xywhaT(xywhT: torch.Tensor, longer_width: bool = True) -> torch.Tensor:
    if longer_width and xywhT[3] > xywhT[2]:
        return torch.cat([xywhT[:2], xywhT[2:4][::-1], torch.Tensor([np.pi / 2]).to(xywhT.device)], dim=0)
    else:
        return torch.cat([xywhT, torch.Tensor([0]).to(xywhT.device)], dim=0)


def xyxyT2xywhaT(xyxyT: torch.Tensor, longer_width: bool = True) -> torch.Tensor:
    return xywhT2xywhaT(xyxyT2xywhT(xyxyT), longer_width=longer_width)


def xyxysT2xywhsT(xyxysT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxysT[..., :2], xyxysT[..., 2:4]
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=-1)


def xyxysT2xywhasT(xyxysT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxysT[..., :2], xyxysT[..., 2:4]
    alphas = torch.zeros_like(xyxysT[..., :1]).to(xyxysT.device)
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1, alphas], dim=-1)


def xywhT2xyxyT(xywhT: torch.Tensor) -> torch.Tensor:
    xcyc, wh_2 = xywhT[:2], xywhT[2:4] / 2
    return torch.cat([xcyc - wh_2, xcyc + wh_2], dim=0)


def xywhsT2xyxysT(xywhsT: torch.Tensor) -> torch.Tensor:
    xcyc, wh_2 = xywhsT[..., :2], xywhsT[..., 2:4] / 2
    return torch.cat([xcyc - wh_2, xcyc + wh_2], dim=-1)


def xyxyT2xlylT(xyxyT: torch.Tensor) -> torch.Tensor:
    xlyl = torch.stack([xyxyT[[0, 0, 2, 2]], xyxyT[[1, 3, 3, 1]]], dim=1)
    return xlyl


def xyxysT2xlylsT(xyxysT: torch.Tensor) -> torch.Tensor:
    xlyls = torch.stack([xyxysT[..., [0, 0, 2, 2]], xyxysT[..., [1, 3, 3, 1]]], dim=-1)
    return xlyls


def xywhaT2xlylT(xywhaT: torch.Tensor) -> torch.Tensor:
    mat = aT2matT(xywhaT[4])
    xlyl = xywhaT[:2] + (CORNERST * xywhaT[2:4] / 2) @ mat
    return xlyl


def xywhasT2xlylsT(xywhasT: torch.Tensor) -> torch.Tensor:
    mat = asT2matsT(xywhasT[..., 4])
    xlyls = xywhasT[..., None, :2] + (CORNERST.to(xywhasT.device) * xywhasT[..., None, 2:4] / 2) @ mat
    return xlyls


def xlylsT2xyxysT(xlylsT: torch.Tensor) -> torch.Tensor:
    x1y1 = torch.min(xlylsT, dim=-2)[0]
    x2y2 = torch.max(xlylsT, dim=-2)[0]
    return torch.cat([x1y1, x2y2], dim=-1)


def xlylsT2xywhsT(xlylsT: torch.Tensor) -> torch.Tensor:
    return xyxysT2xywhsT(xlylsT2xyxysT(xlylsT))


def xywhasT2xyxysT(xywhasT: torch.Tensor) -> torch.Tensor:
    return xlylsT2xyxysT(xywhasT2xlylsT(xywhasT))


def xyxysT_clip(xyxysT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    if xyxyN_rgn.shape[0] == 2:
        xy_max = arrsN2arrsT(np.tile(xyxyN_rgn[0:2], 2), device=xyxysT.device)
        return torch.clamp(torch.minimum(xyxysT, xy_max), min=0)
    else:
        xy_min = arrsN2arrsT(np.tile(xyxyN_rgn[0:2], 2), device=xyxysT.device)
        xy_max = arrsN2arrsT(np.tile(xyxyN_rgn[2:4], 2), device=xyxysT.device)
        return torch.maximum(torch.minimum(xyxysT, xy_max), xy_min)


def xysT_clip(xysT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    if xyxyN_rgn.shape[0] == 2:
        xy_max = arrsN2arrsT(xyxyN_rgn[0:2], device=xysT.device)
        return torch.clamp(torch.minimum(xysT, xy_max), min=0)
    else:
        xy_min = arrsN2arrsT(xyxyN_rgn[0:2], device=xysT.device)
        xy_max = arrsN2arrsT(xyxyN_rgn[2:4], device=xysT.device)
        return torch.maximum(torch.minimum(xysT, xy_max), xy_min)


def xywhsT_clip(xywhsT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    xyxysT = xywhsT2xyxysT(xywhsT)
    xyxysT = xyxysT_clip(xyxysT, xyxyN_rgn=xyxyN_rgn)
    xywhsT = xyxysT2xywhsT(xyxysT)
    return xywhsT


def arange2dT(sz0, sz1, device=DEVICE) -> (torch.Tensor, torch.Tensor):
    ys = torch.arange(sz0, device=device)[:, None].expand(sz0, sz1)
    xs = torch.arange(sz1, device=device)[None, :].expand(sz0, sz1)
    return ys, xs


def create_meshT(H: int, W: int, device=DEVICE) -> torch.Tensor:
    ys, xs = arange2dT(H, W, device)
    mesh = torch.stack([xs, ys], dim=2).view(H * W, 2) + 0.5
    return mesh


def create_meshesT(Nb: int, H: int, W: int, device=DEVICE) -> torch.Tensor:
    ys, xs = arange2dT(H, W, device)
    meshes = torch.stack([xs, ys], dim=2).view(H * W, 2) + 0.5
    meshes = meshes.expand(Nb, W * H, 2)
    return meshes


def xywhasT2masksT_guss(xywhasT: torch.Tensor, size: tuple) -> torch.Tensor:
    xywhuvsT = xywhasT2xywhuvsT(xywhasT)
    return xywhuvsT2masksT_guss(xywhuvsT, size)


def uvT2matT(uvT: torch.Tensor) -> torch.Tensor:
    mat = torch.stack([uvT, torch.stack([-uvT[1], uvT[0]], dim=0)], dim=0)
    return mat


def uvsT2matsT(uvsT: torch.Tensor) -> torch.Tensor:
    mats = torch.stack([uvsT, torch.stack([-uvsT[..., 1], uvsT[..., 0]], dim=-1)], dim=-2)
    return mats


def aT2matT(aT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(aT), torch.sin(aT)
    mat = torch.stack([torch.stack([cos, sin], dim=0), torch.stack([-sin, cos], dim=0)], dim=0)
    return mat


def asT2matsT(asT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(asT), torch.sin(asT)
    mat = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=-2)
    return mat


def aT2uvT(aT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(aT), torch.sin(aT)
    return torch.stack([cos, sin], dim=0)


def uvT2aT(uvT: torch.Tensor) -> torch.Tensor:
    return torch.atan2(uvT[1], uvT[0])


def uvsT2asT(uvsT: torch.Tensor) -> torch.Tensor:
    return torch.atan2(uvsT[..., 1], uvsT[..., 0])


def asT2uvsT(asT: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(asT), torch.sin(asT)], dim=-1)


def xywhuvsT2masksT_guss(xywhuvsT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, uvs = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4:6]
    mats = uvsT2matsT(uvs)

    meshes = create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)

    pows_guss = torch.exp(-torch.sum((meshes_proj * 2 / whs) ** 2, dim=2) / 2)
    pows_guss = pows_guss.view(Nb, Hf, Wf)
    return pows_guss


def xywhasT2masksT_cness(xywhasT: torch.Tensor, size: tuple) -> torch.Tensor:
    xywhuvsT = xywhasT2xywhuvsT(xywhasT)
    return xywhuvsT2masksT_cness(xywhuvsT, size)


def xywhuvsT2masksT_cness(xywhuvsT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, uvs = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4:6]
    mats = uvsT2matsT(uvs)

    meshes = create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)
    meshes_proj = torch.abs(meshes_proj)

    pows_cness = torch.sqrt(torch.prod(torch.clamp_(whs / 2 - meshes_proj, min=0), dim=2)
                            / torch.prod(whs / 2 + meshes_proj, dim=2))

    pows_cness = pows_cness.view(Nb, Hf, Wf)
    return pows_cness


def xywhasT2masksTb(xywhasT: torch.Tensor, size: tuple) -> torch.Tensor:
    xywhuvsT = xywhasT2xywhuvsT(xywhasT)
    return xywhuvsT2masksTb(xywhuvsT, size)


def xywhuvsT2masksTb(xywhuvsT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, uvs = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4:6]
    mats = uvsT2matsT(uvs)

    meshes = create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)
    meshes_proj = torch.abs(meshes_proj)
    masks = torch.all(meshes_proj < whs / 2, dim=2)
    masks = masks.view(Nb, Hf, Wf)
    return masks


def xywhuvsT2masksTb_border(xywhuvsT: torch.Tensor, size: tuple, expand_ratio=1.2) -> torch.Tensor:
    Nb = xywhuvsT.size(0)
    Wf, Hf = size
    xys, whs, uvs = xywhuvsT[:, None, 0:2], xywhuvsT[:, None, 2:4], xywhuvsT[:, 4:6]
    mats = uvsT2matsT(uvs)

    meshes = create_meshesT(Nb, Hf, Wf, xywhuvsT.device) - xys
    meshes_proj = torch.bmm(meshes, mats)
    meshes_proj = torch.abs(meshes_proj)
    masks = torch.all((meshes_proj >= whs / 2) * (meshes_proj < whs / 2 * expand_ratio), dim=2)
    masks = masks.view(Nb, Hf, Wf)
    return masks


def xyxysT2masksTb(xyxysT: torch.Tensor, size: tuple) -> torch.Tensor:
    Nb, _ = xyxysT.size()
    Wf, Hf = size
    x1y1s, x2y2s = xyxysT[:, None, 0:2], xyxysT[:, None, 2:4]
    meshes = create_meshesT(Nb, Hf, Wf, xyxysT.device)
    masks = torch.all((meshes < x2y2s) * (meshes > x1y1s), dim=2)
    masks = masks.view(Nb, Hf, Wf)
    return masks


def _masksT_scatter_btch(masksT: torch.Tensor, ids_b: torch.Tensor, Nb: int) -> torch.Tensor:
    masks = torch.zeros((Nb, masksT.size(1), masksT.size(2)), device=masksT.device, dtype=masksT.dtype)
    masks.scatter_add_(dim=0, index=ids_b[:, None, None].expand(masksT.size()), src=masksT)
    return masks


def _masksTb_scatter_btch_cind(masksTb: torch.Tensor, ids_b: torch.Tensor, Nb: int, cinds: torch.Tensor,
                               num_cls: int) -> torch.Tensor:
    masks = torch.full((Nb, masksTb.size(1), masksTb.size(2)), device=masksTb.device, dtype=torch.long,
                       fill_value=num_cls)
    ib, ih, iw = torch.nonzero(masksTb, as_tuple=True)
    masks[ids_b[ib], ih, iw] = cinds[ib]
    return masks


def bxyxysT2masksTb(xyxys: torch.Tensor, ids_b: torch.Tensor, Nb: int, size: tuple) -> torch.Tensor:
    masks_bool = xyxysT2masksTb(xyxys, size)
    return _masksT_scatter_btch(masks_bool, ids_b, Nb)


def bxywhasT2masksTb(xywhas: torch.Tensor, ids_b: torch.Tensor, Nb: int, size: tuple) -> torch.Tensor:
    masks_bool = xywhasT2masksTb(xywhas, size)
    return _masksT_scatter_btch(masks_bool, ids_b, Nb)


def bcxyxysT2masksT_enc(xyxys: torch.Tensor, ids_b: torch.Tensor, Nb: int, cinds: torch.Tensor, num_cls: int,
                        size: tuple) -> torch.Tensor:
    masks_bool = xyxysT2masksTb(xyxys, size)
    return _masksTb_scatter_btch_cind(masks_bool, ids_b, Nb, cinds, num_cls)


def xywhuvT2xywhaT(xywhuv: torch.Tensor) -> torch.Tensor:
    alpha = torch.acos(xywhuv[4:5]) if xywhuv[5:6] > 0 else -torch.acos(xywhuv[4:5])
    return torch.cat([xywhuv[:4], alpha], dim=0)


def xywhuvsT2xywhasT(xywhuvs: torch.Tensor) -> torch.Tensor:
    alphas = torch.acos(xywhuvs[..., 4:5])
    alphas = torch.where(xywhuvs[..., 5:6] > 0, alphas, -alphas)
    return torch.cat([xywhuvs[..., :4], alphas], dim=-1)


def xywhaT2xywhuvT(xywha: torch.Tensor) -> torch.Tensor:
    return torch.cat([xywha[:4], torch.cos(xywha[4:5]), torch.sin(xywha[4:5])], dim=0)


def xywhasT2xywhuvsT(xywhas: torch.Tensor) -> torch.Tensor:
    alphas = xywhas[..., 4:5]
    return torch.cat([xywhas[..., :4], torch.cos(alphas), torch.sin(alphas)], dim=-1)


def abclT_intersect(abcl1T: torch.Tensor, abcl2T: torch.Tensor) -> torch.Tensor:
    norm = abcl1T[:, 0] * abcl2T[:, 1] - abcl1T[:, 1] * abcl2T[:, 0]
    x = (abcl1T[:, 1] * abcl2T[:, 2] - abcl1T[:, 2] * abcl2T[:, 1]) / norm
    y = -(abcl1T[:, 0] * abcl2T[:, 2] - abcl1T[:, 2] * abcl2T[:, 0]) / norm
    return torch.stack([x, y], dim=1)


def abclsT_intersect(abcls1T: torch.Tensor, abcls2T: torch.Tensor) -> torch.Tensor:
    norm = abcls1T[..., 0] * abcls2T[..., 1] - abcls1T[..., 1] * abcls2T[..., 0]
    x = (abcls1T[..., 1] * abcls2T[..., 2] - abcls1T[..., 2] * abcls2T[..., 1]) / norm
    y = -(abcls1T[..., 0] * abcls2T[..., 2] - abcls1T[..., 2] * abcls2T[..., 0]) / norm
    return torch.stack([x, y], dim=-1)


def xlylT2abclT(xlylT: torch.Tensor) -> torch.Tensor:
    xlylT_rnd = torch.roll(xlylT, shifts=-1, dims=0)
    As = xlylT[:, 1] - xlylT_rnd[:, 1]
    Bs = xlylT_rnd[:, 0] - xlylT[:, 0]
    Cs = xlylT_rnd[:, 1] * xlylT[:, 0] - xlylT[:, 1] * xlylT_rnd[:, 0]
    return torch.stack([As, Bs, Cs], dim=1)


def xlylsT2abclsT(xlylsT: torch.Tensor) -> torch.Tensor:
    xlylsT_rnd = torch.roll(xlylsT, shifts=-1, dims=-2)
    As = xlylsT[..., 1] - xlylsT_rnd[..., 1]
    Bs = xlylsT_rnd[..., 0] - xlylsT[..., 0]
    Cs = xlylsT_rnd[..., 1] * xlylsT[..., 0] - xlylsT[..., 1] * xlylsT_rnd[..., 0]
    return torch.stack([As, Bs, Cs], dim=-1)


def xlylT2areaT(xlylT: torch.Tensor) -> torch.Tensor:
    xlylT_rnd = torch.roll(xlylT, shifts=-1, dims=0)
    area = xlylT_rnd[..., 0] * xlylT[..., 1] - xlylT[..., 0] * xlylT_rnd[..., 1]
    area = torch.abs(torch.sum(area)) / 2
    return area


def xlylsT2areasT(xlylsT: torch.Tensor) -> torch.Tensor:
    xlylsT_rnd = torch.roll(xlylsT, shifts=-1, dims=-2)
    areas = xlylsT_rnd[..., 0] * xlylsT[..., 1] - xlylsT[..., 0] * xlylsT_rnd[..., 1]
    areas = torch.abs(torch.sum(areas, dim=-1)) / 2
    return areas


def xlylT_clkwise(xlylT: torch.Tensor) -> torch.Tensor:
    xlylT_rnd = torch.roll(xlylT, shifts=-1, dims=0)
    area = xlylT_rnd[..., 0] * xlylT[..., 1] - xlylT[..., 0] * xlylT_rnd[..., 1]
    if torch.sum(area) < 0:
        xlylT = xlylT[::-1, :]
    return xlylT


def xlylsT_clkwise(xlylsT: torch.Tensor) -> torch.Tensor:
    xlylsT_rnd = torch.roll(xlylsT, shifts=-1, dims=-2)
    areas = xlylsT_rnd[..., 0] * xlylsT[..., 1] - xlylsT[..., 0] * xlylsT_rnd[..., 1]
    fltr = torch.sum(areas, dim=-1) < 0
    xlylsT = torch.where(fltr[..., None, None], xlylsT[..., ::-1, :], xlylsT)
    return xlylsT


def xlylT_intersect_coreT(xlyl1T: torch.Tensor, xlyl2T: torch.Tensor, eps: float = 1e-7) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor):
    abcl1 = xlylT2abclT(xlyl1T)[:, None, :]
    abcl2 = xlylT2abclT(xlyl2T)[None, :, :]
    r1 = torch.sum(xlyl1T[:, None, :] * abcl2[..., :2], dim=-1) + abcl2[..., 2]
    r2 = torch.sum(xlyl2T * abcl1[..., :2], dim=-1) + abcl1[..., 2]
    msk_1pin2 = torch.all(r1 >= -eps, dim=-1) + torch.all(r1 <= eps, dim=-1)
    msk_2pin1 = torch.all(r2 >= -eps, dim=-2) + torch.all(r2 <= eps, dim=-2)
    if torch.all(msk_1pin2) or torch.all(msk_2pin1):
        return torch.zeros_like(r1), msk_1pin2, msk_2pin1
    norm = (abcl1[..., 0] * abcl2[..., 1] - abcl1[..., 1] * abcl2[..., 0]).float()
    r1 = r1.float() / norm
    r2 = -r2.float() / norm
    fltr1 = (r1 > -eps) * (r1 < 1 + eps)
    fltr2 = (r2 > -eps) * (r2 < 1 + eps)
    fltr_int = fltr1 * fltr2
    ratio1 = torch.where(fltr_int, r1, 0)
    return ratio1, msk_1pin2, msk_2pin1


def xlylsT_intersect_coresT(xlyls1T: torch.Tensor, xlyls2T: torch.Tensor, eps: float = 1e-7) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor):
    abcls1 = xlylsT2abclsT(xlyls1T)[..., None, :]
    abcls2 = xlylsT2abclsT(xlyls2T)[..., None, :, :]
    r1 = torch.sum(xlyls1T[..., None, :] * abcls2[..., :2], dim=-1) + abcls2[..., 2]
    r2 = torch.sum(xlyls2T * abcls1[..., :2], dim=-1) + abcls1[..., 2]
    msk_1pin2 = torch.all(r1 >= -eps, dim=-1) + torch.all(r1 <= eps, dim=-1)
    msk_2pin1 = torch.all(r2 >= -eps, dim=-2) + torch.all(r2 <= eps, dim=-2)
    norm = (abcls1[..., 0] * abcls2[..., 1] - abcls1[..., 1] * abcls2[..., 0]).float()
    r1 = r1.float() / norm
    r2 = -r2.float() / norm
    fltr1 = (r1 > -eps) * (r1 < 1 + eps)
    fltr2 = (r2 > -eps) * (r2 < 1 + eps)
    fltr_int = fltr1 * fltr2
    ratio1 = torch.where(fltr_int, r1, 0)
    return ratio1, msk_1pin2, msk_2pin1


def intersect_coreT2xlylT(xlyl1T: torch.Tensor, xlyl2T: torch.Tensor, ratio1: torch.Tensor,
                          msk_1pin2: torch.Tensor, msk_2pin1: torch.Tensor) -> torch.Tensor:
    if torch.all(msk_1pin2):
        return xlyl1T
    elif torch.all(msk_2pin1):
        return xlyl2T
    elif not np.any(ratio1 > 0):
        return torch.zeros(size=(0, 2))

    idls1, idls2 = torch.nonzero(ratio1 > 0, as_tuple=True)
    xlyl1T_rnd = torch.cat([xlyl1T[..., 1:, :], xlyl1T[..., :1, :]], dim=-2)
    ratios = ratio1[idls1, idls2]
    xlyl_int = xlyl1T_rnd[idls1] * ratios[:, None] + xlyl1T[idls1] * (1 - ratios[:, None])
    dists = ratios + idls1
    order = torch.argsort(dists)
    idls2 = idls2[order] + 1
    idls1 = idls1 + 1
    xlyl_int = xlyl_int[order]
    # 按序遍历
    num1 = xlyl1T.size(0)
    num2 = xlyl2T.size(0)
    num_int = len(idls1)
    idls1_nxt = torch.cat([idls1[1:], idls1[0:1]])
    idls1_nxt = torch.where(idls1_nxt < idls1, idls1_nxt + num1, idls1_nxt)
    idls2_nxt = torch.cat([idls2[1:], idls2[0:1]])
    idls2_nxt = torch.where(idls2_nxt < idls2, idls2_nxt + num2, idls2_nxt)
    ids = [torch.zeros(size=(0,), dtype=torch.long)]
    for i in range(num_int):
        ids.append(torch.Tensor([i + num1 + num2]).long())
        ids.append(torch.arange(int(idls1[i]), int(idls1_nxt[i]), dtype=torch.long) % num1)
        ids.append(torch.arange(int(idls2[i]), int(idls2_nxt[i]), dtype=torch.long) % num2 + num1)
    ids = torch.cat(ids, dim=0)
    pnts = torch.cat([xlyl1T, xlyl2T, xlyl_int])
    msks = torch.cat(
        [msk_1pin2, msk_2pin1, torch.full(size=(xlyl_int.size(0),), fill_value=True, device=xlyl_int.device)])
    xlyl_final = pnts[ids][msks[ids]]
    return xlyl_final


def xlylT_intersect(xlyl1: torch.Tensor, xlyl2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    xlyl1 = xlylT_clkwise(xlyl1)
    xlyl2 = xlylT_clkwise(xlyl2)
    ratio1, msk_1pin2, msk_2pin1 = xlylT_intersect_coreT(xlyl1, xlyl2, eps=eps)
    xlyl_final = intersect_coreT2xlylT(xlyl1, xlyl2, ratio1, msk_1pin2, msk_2pin1)
    return xlyl_final


# </editor-fold>Z

# <editor-fold desc='list边界格式转换'>

def xyxyL2xywhL(xyxy: list) -> list:
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywhL2xyxyL(xywh: list) -> list:
    xc, yc, w_2, h_2 = xywh[0], xywh[1], xywh[2], xywh[3]
    return [xc - w_2, yc - h_2, xc + w_2, yc + h_2]


def xyxyL2xywhN(xyxy: list) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def xywhL2xyxyN(xywh: list) -> np.ndarray:
    xc, yc, w_2, h_2 = xywh[0], xywh[1], xywh[2], xywh[3]
    return np.array([xc - w_2, yc - h_2, xc + w_2, yc + h_2])


# </editor-fold>


# <editor-fold desc='图像格式转换'>


def imgP2imgN(imgP: PIL.Image.Image) -> np.ndarray:
    if imgP.size[0] == 0 or imgP.size[1] == 0:
        if imgP.mode == 'L':
            return np.zeros(shape=(imgP.size[1], imgP.size[0]))
        elif imgP.mode == 'RGB':
            return np.zeros(shape=(imgP.size[1], imgP.size[0], 3))
        elif imgP.mode == 'RGBA':
            return np.zeros(shape=(imgP.size[1], imgP.size[0], 4))
        else:
            raise Exception('err num ' + str(imgP.mode))
    imgN = np.array(imgP)
    return imgN


def imgN2imgP(imgN: np.ndarray) -> PIL.Image.Image:
    if len(imgN.shape) == 2:
        imgP_tp = 'L'
    elif len(imgN.shape) == 3 and imgN.shape[2] == 1:
        imgP_tp = 'L'
        imgN = imgN.squeeze(axis=2)
    elif imgN.shape[2] == 3:
        imgP_tp = 'RGB'
    elif imgN.shape[2] == 4:
        imgP_tp = 'RGBA'
    else:
        raise Exception('err num ' + str(imgN.shape))
    imgN = Image.fromarray(imgN.astype(np.uint8), mode=imgP_tp)
    return imgN


def imgN2imgT(imgN: np.ndarray, device=DEVICE) -> torch.Tensor:
    imgT = torch.from_numpy(imgN).float()
    imgT = imgT.permute((2, 0, 1)) / 255
    imgT = imgT[None, :]
    return imgT.to(device)


def imgT2imgN(imgT: torch.Tensor) -> np.ndarray:
    imgT = imgT * 255
    imgN = imgT.detach().cpu().numpy().astype(np.uint8)
    if len(imgN.shape) == 4 and imgN.shape[0] == 1:
        imgN = imgN.squeeze(axis=0)
    imgN = np.transpose(imgN, (1, 2, 0))  # CHW转为HWC
    return imgN


def imgP2imgT(imgP: PIL.Image.Image, device=DEVICE) -> torch.Tensor:
    imgT = torch.from_numpy(np.array(imgP)).float()
    imgT = imgT.permute((2, 0, 1)) / 255
    imgT = imgT[None, :]
    return imgT.to(device)


def imgT2imgP(imgT: torch.Tensor) -> PIL.Image.Image:
    imgN = imgT2imgN(imgT)
    imgP = imgN2imgP(imgN)
    return imgP


def imgsN2imgsP(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = Image.fromarray(imgs[i].astype('uint8')).convert('RGB')
    return imgs


def imgsP2imgsN(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = np.array(imgs[i])
    return imgs


def imgs2imgsN(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = img2imgN(imgs[i])
    return imgs


def img2imgT(img, device=DEVICE) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        return imgN2imgT(img, device)
    elif isinstance(img, PIL.Image.Image):
        return imgP2imgT(img, device)
    elif isinstance(img, torch.Tensor):
        return img.to(device)
    else:
        raise Exception('err type ' + img.__class__.__name__)


def img2imgP(img) -> PIL.Image.Image:
    if isinstance(img, np.ndarray):
        return imgN2imgP(img)
    elif isinstance(img, PIL.Image.Image):
        return img
    elif isinstance(img, torch.Tensor):
        return imgT2imgP(img)
    else:
        raise Exception('err type ' + img.__class__.__name__)


def img2imgN(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, PIL.Image.Image):
        return imgP2imgN(img)
    elif isinstance(img, torch.Tensor):
        return imgT2imgN(img)
    else:
        raise Exception('err type ' + img.__class__.__name__)


# </editor-fold>

# <editor-fold desc='图像绘制处理'>
def xlylN2maskP(xlylN: np.ndarray, size: tuple) -> PIL.Image.Image:
    maskP = Image.new('1', tuple(size), 0)
    PIL.ImageDraw.Draw(maskP).polygon(list(xlylN.reshape(-1)), fill=1)
    return maskP


def xywhaN2maskP(xywhaN: np.ndarray, size: tuple) -> PIL.Image.Image:
    xlylN = xywhaN2xlylN(xywhaN)
    maskP = xlylN2maskP(xlylN=xlylN, size=size)
    return maskP


def xywhuvN2maskP(xywhuvN: np.ndarray, size: tuple) -> PIL.Image.Image:
    xlylN = xywhuvN2xlylN(xywhuvN)
    maskP = xlylN2maskP(xlylN=xlylN, size=size)
    return maskP


def imgN_crop(imgN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    xy_min = np.maximum(xyxyN_rgn[:2].astype(np.int32), np.zeros(shape=2))
    xy_max = np.minimum(xyxyN_rgn[2:4].astype(np.int32), np.array((imgN.shape[1], imgN.shape[0])))
    return imgN[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]


# </editor-fold>

# <editor-fold desc='图像批次处理'>

def _size_limt_ratio(size: tuple, max_size: tuple, only_smaller: bool = False, only_larger: bool = False):
    ratio = min(np.array(max_size) / np.array(size))
    if (ratio > 1.0 and only_smaller) or (ratio < 1.0 and only_larger):
        return 1.0
    else:
        return ratio


def imgP_lmtsize(imgP: PIL.Image.Image, max_size: tuple, resample=Image.BILINEAR,
                 only_smaller: bool = False, only_larger: bool = False) -> (PIL.Image.Image, float):
    ratio = _size_limt_ratio(imgP.size, max_size, only_smaller=only_smaller, only_larger=only_larger)
    if ratio == 1.0:
        return imgP, ratio
    imgP = imgP.resize(size=tuple((np.array(imgP.size) * ratio).astype(np.int32)), resample=resample)
    return imgP, ratio


def imgN_lmtsize(imgN: np.ndarray, max_size: tuple, resample=cv2.INTER_LANCZOS4,
                 only_smaller: bool = False, only_larger: bool = False) -> (np.ndarray, float):
    size = (imgN.shape[1], imgN.shape[0])
    ratio = _size_limt_ratio(size, max_size, only_smaller=only_smaller, only_larger=only_larger)
    if ratio == 1.0:
        return imgN, ratio
    imgN = cv2.resize(imgN, tuple(np.round(np.array(size) * ratio).astype(np.int32)), interpolation=resample)
    return imgN, ratio


def imgP_lmtsize_pad(imgP: PIL.Image.Image, max_size: tuple, pad_val: int = 127, resample=Image.BILINEAR) \
        -> (PIL.Image.Image, float):
    if imgP.size == max_size:
        return imgP, 1
    imgN = imgP2imgN(imgP)
    aspect = max_size[0] / max_size[1]
    imgN = imgN_pad_aspect(imgN, aspect=aspect, pad_val=pad_val)
    imgP = imgN2imgP(imgN)
    ratio = min(max_size[0] / imgP.size[0], max_size[1] / imgP.size[1])
    imgP = imgP.resize(max_size, resample=resample)
    return imgP, ratio


def imgN_lmtsize_pad(imgN: np.ndarray, max_size: tuple, pad_val: int = 127, resample=cv2.INTER_LANCZOS4) \
        -> (np.ndarray, float):
    if img2size(imgN) == max_size:
        return imgN, 1
    aspect = max_size[0] / max_size[1]
    imgN = imgN_pad_aspect(imgN, aspect=aspect, pad_val=pad_val)
    ratio = min(max_size[0] / imgN.shape[1], max_size[1] / imgN.shape[0])
    imgN = cv2.resize(imgN, max_size, interpolation=resample)
    return imgN, ratio


def imgT_lmtsize_pad(imgT: torch.Tensor, max_size: tuple, pad_val: int = 127) \
        -> (torch.Tensor, float):
    aspect = max_size[0] / max_size[1]
    imgT = imgT_pad_aspect(imgT, aspect=aspect, pad_val=pad_val)
    ratio = min(max_size[0] / imgT.size(3), max_size[1] / imgT.size(2))
    imgT = F.interpolate(imgT, size=(max_size[1], max_size[0]))
    return imgT, ratio


def imgN_pad_aspect(imgN: np.ndarray, aspect: float, pad_val: int = 127) -> np.ndarray:
    h, w, _ = imgN.shape
    if w / h > aspect:
        imgN = np.pad(imgN, pad_width=((0, int(w / aspect - h)), (0, 0), (0, 0)), constant_values=pad_val)
    elif w / h < aspect:
        imgN = np.pad(imgN, pad_width=((0, 0), (0, int(h * aspect - w)), (0, 0)), constant_values=pad_val)
    return imgN


def imgT_pad_aspect(imgT: torch.Tensor, aspect: float, pad_val: int = 127) -> torch.Tensor:
    _, _, h, w = imgT.size()
    if w / h > aspect:
        imgT = F.pad(imgT, pad=(0, 0, 0, int(w / aspect - h)), value=pad_val / 255)
    elif w / h < aspect:
        imgT = F.pad(imgT, pad=(0, int(h * aspect - w), 0, 0), value=pad_val / 255)
    return imgT


def imgs2imgsT(imgs: list, img_size: tuple, pad_val: int = 127, device=DEVICE) -> (torch.Tensor, np.ndarray):
    if isinstance(imgs, torch.Tensor):
        if imgs.size(3) == img_size[0] and imgs.size(2) == img_size[1]:
            return imgs.to(device), np.ones(shape=len(imgs))
        else:
            imgsT, ratio = imgT_lmtsize_pad(imgT=imgs, max_size=img_size, pad_val=pad_val)
            return imgsT.to(device), np.full(shape=len(imgs), fill_value=ratio)
    imgsT = []
    ratios = []
    for img in imgs:
        imgT = img2imgT(img, device=device)
        imgT, ratio = imgT_lmtsize_pad(imgT=imgT, max_size=img_size, pad_val=pad_val)
        imgsT.append(imgT)
        ratios.append(ratio)
    imgsT = torch.cat(imgsT, dim=0)
    ratios = np.array(ratios)
    return imgsT, ratios


def img2size(img) -> tuple:
    if isinstance(img, PIL.Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return (img.shape[1], img.shape[0])
    elif isinstance(img, torch.Tensor):
        return (img.size(-1), img.size(-2))
    else:
        raise Exception('err type ' + img.__class__.__name__)


# </editor-fold>


# <editor-fold desc='DenseCRF'>

try:
    import pydensecrf.densecrf as dcrf
except Exception as e:
    pass


def masksT_crf(imgT: torch.Tensor, masksT: torch.Tensor, sxy: int = 40, srgb: int = 10,
               num_infer: int = 2) -> torch.Tensor:
    C, H, W = masksT.size()

    d = dcrf.DenseCRF2D(W, H, C)
    masksT = -torch.log(masksT.clamp_(min=1e-8))
    maskN = masksT.detach().cpu().numpy()
    maskN = maskN.reshape(maskN.shape[0], -1)

    d.setUnaryEnergy(maskN)
    imgN = imgT2imgN(imgT)
    imgN = np.ascontiguousarray(imgN.astype(np.uint8))
    d.addPairwiseBilateral(sxy=(sxy, sxy), srgb=(srgb, srgb, srgb), rgbim=imgN, compat=10, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(num_infer)
    qn = np.array(Q).reshape((C, H, W))
    masksT_crf = torch.from_numpy(qn).to(imgT.device)
    return masksT_crf


def masksN_crf(imgN: np.ndarray, masksN: np.ndarray, sxy: int = 40, srgb: int = 10,
               num_infer: int = 2) -> np.ndarray:
    H, W, C = masksN.shape
    d = dcrf.DenseCRF2D(W, H, C)
    masksN = -np.log(np.clip(masksN, a_min=1e-5, a_max=None))
    maskN = masksN.transpose((2, 0, 1))  # (C, homography, W)
    maskN = np.ascontiguousarray(maskN.reshape(maskN.shape[0], -1))

    d.setUnaryEnergy(maskN.astype(np.float32))
    imgN = np.ascontiguousarray(imgN.astype(np.uint8))
    d.addPairwiseBilateral(sxy=(sxy, sxy), srgb=(srgb, srgb, srgb), rgbim=imgN, compat=10, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(num_infer)
    qn = np.array(Q).reshape((C, H, W))
    qn = qn.transpose((1, 2, 0))  # (homography,W,C)
    return qn


def masksT2masksNb_with_conf(masksT: torch.Tensor, conf_thres: float = None) -> np.ndarray:
    conf_thres_i = conf_thres if conf_thres is not None else torch.max(masksT) / 2
    masksTb = (masksT > conf_thres_i).bool()
    masksNb = masksTb.detach().cpu().numpy().astype(bool)
    if len(masksNb.shape) == 4 and masksNb.shape[0] == 1:
        masksNb = masksNb.squeeze(axis=0)
    masksNb = np.transpose(masksNb, (1, 2, 0))  # CHW转为HWC
    return masksNb

# </editor-fold>
# if __name__ == '__main__':
#     xywha = np.array([20, 20, 50, 40, 1])
#     xlyl = xywhaN2xlylN(xywha)
#     xywha2 = xlylN2xywhaN(xlyl)
