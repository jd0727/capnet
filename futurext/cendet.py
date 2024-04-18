import skimage.filters as skif
import skimage.filters.rank as sfr
from skimage.morphology import disk

from utils import *

INSULATOR_COLOR = np.array([142.06330492, 160.62559905, 153.84767562])
INSULATOR_STD = np.array([57.41854421, 53.317534, 56.40438266])


def imgN_cluster_vis(imgN: np.ndarray, indexs: np.ndarray, n_clusters: int):
    img_vis = np.zeros_like(imgN)
    for index in range(n_clusters):
        msk_src = (indexs == index)[..., None]
        # color = np.sum(msk_src * imgN, axis=(0, 1)) / np.sum(msk_src, axis=(0, 1))
        # color = color.astype(np.uint8)
        color = random_color(index)
        img_vis = np.where(msk_src, color, img_vis)
    return img_vis


def local_nms(feat, sigma):
    feat = skif.gaussian(feat, sigma=sigma, )
    feat = (feat / np.max(feat) * 255).astype(np.uint8)
    key_val = sfr.maximum(feat, disk(sigma))
    # key_val = sfr.minimum(feat, disk(sigma))
    mskr = (feat == key_val)
    return mskr


def imgN_maskNb_cendet(imgN, maskNb, num_div=36):
    colors = imgN[maskNb]
    color_mean = np.mean(colors, axis=0)
    color_std = np.std(colors, axis=0)
    # mask_valid = np.all(np.abs(color_mean - imgN) < color_std, axis=2) * maskNb
    mask_valid = maskNb
    heat = np.exp(-np.sum(((color_mean - imgN) / color_std) ** 2, axis=2)) * mask_valid

    sigma = np.sqrt(np.sum(maskNb)) / 20
    mask_peak = local_nms(heat, sigma) * mask_valid
    ys, xs = np.nonzero(mask_peak)
    cens = np.stack([xs, ys], axis=1)

    nms_inds = nms_xysN(xysN=cens, confsN=heat[ys, xs], radius=sigma, num_presv=10000)
    cens_nmsd = cens[nms_inds]
    indexs = imgN_cluster(imgN=imgN, cens=cens_nmsd, sxy=sigma, srgb=color_std,
                          radius=sigma * 9, num_infer=10, maskNb=mask_valid)
    xlyls = []
    for i in range(len(cens_nmsd)):
        mask_i = indexs == i
        dls = maskNb_xyN2dlN(mask_i, cens_nmsd[i], num_div=num_div)
        if np.all(dls > 0):
            xlyl = cenN_dlN2xlylN(cens_nmsd[i], dls)
            xlyls.append(xlyl)
    return xlyls


def imgN_xlylN_cendet(imgN, xlylN, num_div=36):
    xyxy = xlylN2xyxyN(xlylN).astype(np.int32)
    imgN_piece = imgN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    xlylN = xlylN - xyxy[:2]
    maskNb = xlylN2maskNb(xlylN, size=xyxy[2:4] - xyxy[:2])
    xlyls = imgN_maskNb_cendet(imgN_piece, maskNb, num_div=num_div)
    for i in range(len(xlyls)):
        xlyls[i] = xlyls[i] + xyxy[:2]
    return xlyls


def imgN_xyxyN_cendet(imgN, xyxyN, stride, num_div=36, num_infer=10, flexible_cen=True):
    xyxyN = xyxyN.astype(np.int32)
    imgN_piece = imgN[xyxyN[1]:xyxyN[3], xyxyN[0]:xyxyN[2]]

    cens = uniform_cens(imgN_piece.shape[::-1], stride)
    color_std = np.std(imgN_piece, axis=(0, 1))
    indexs, _, _ = imgN_cluster(imgN=imgN_piece, cens=cens, sxy=stride / 3, srgb=color_std,
                                radius=stride * 3, num_infer=num_infer, flexible_cen=flexible_cen)
    xlyls = []
    for i in range(len(cens)):
        dls = maskNb_xyN2dlN(indexs == i, cens[i], num_div=num_div, erode=1)
        xlyl = cenN_dlN2xlylN(cens[i], dls)
        xlyl = xlyl + xyxyN[:2]
        xlyls.append(xlyl)

    return xlyls


def cond_cens(img_size: tuple, thres=3, base_ratio=2.5) -> np.ndarray:
    w, h = img_size

    aspect = w / h
    if aspect > thres:
        nx, ny = np.ceil(aspect * base_ratio), 1
    elif aspect > 1:
        nx, ny = np.ceil(aspect * base_ratio * 2), 2
    elif aspect > 1 / thres:
        nx, ny = 2, np.ceil(1 / aspect * base_ratio * 2)
    else:
        nx, ny = 1, np.ceil(1 / aspect * base_ratio)
    stride = np.array(img_size) / np.array([nx, ny])
    sw, sh = stride
    x, y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
    cens = np.stack([(x + 0.5) * sw, (y + 0.5) * sh], axis=2).reshape(-1, 2)
    cens = cens[(cens[..., 0] < w) * (cens[..., 1] < h)]
    return cens, stride


def imgN_xyxyN_cendet2(imgN, xyxyN, num_div=36, num_infer=10, flexible_cen=True):
    xyxyN = xyxyN.astype(np.int32)
    imgN_piece = imgN[xyxyN[1]:xyxyN[3], xyxyN[0]:xyxyN[2]]

    cens, stride = cond_cens(imgN_piece.shape[:2][::-1])
    color_std = np.std(imgN_piece, axis=(0, 1))

    indexs, colors, cens = imgN_cluster(
        imgN=imgN_piece, cens=cens, sxy=stride / 3, srgb=color_std,
        radius=stride * 3, num_infer=num_infer, flexible_cen=flexible_cen)
    xlyls = []
    for i in range(len(cens)):
        if np.any(np.abs(colors[i] - INSULATOR_COLOR) > INSULATOR_STD * 2):
            continue
        dls = maskNb_xyN2dlN(indexs == i, cens[i], num_div=num_div, erode=0)
        xlyl = cenN_dlN2xlylN(cens[i], dls)
        xlyl = xlyl + xyxyN[:2]
        xlyls.append(xlyl)

    return xlyls
