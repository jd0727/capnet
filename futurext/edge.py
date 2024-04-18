import os
import sys

PROJECT_NAME = 'Geo'
PROJECT_PTH = os.path.abspath(__file__).split(PROJECT_NAME)[0] + PROJECT_NAME
sys.path.append(PROJECT_PTH)
from utils import *


def dlsT_scan_with_edge(imgsT: torch.Tensor, ids_b: torch.Tensor, cens: torch.Tensor, dls: torch.Tensor,
                        radius=3, kernel_size=5, dilation=1) -> torch.Tensor:
    num_pnt, num_div = dls.size()
    N, C, H, W = imgsT.size()
    num_dt = 2 * radius + 1
    dirs = create_dirsT(num_div, device=imgsT.device, bias=0)
    dtdl = torch.arange(-radius, radius + 1, device=imgsT.device)
    dls_ext = torch.clamp_(dls[..., None].detach() + dtdl, min=0)
    verts = cens[:, None, None, :] + dls_ext[..., None] * dirs[:, None, :]
    verts = verts.long()
    ids_x, ids_y = verts[..., 0], verts[..., 1]
    fltr_out = (ids_x < 0) + (ids_x >= W) + (ids_y < 0) + (ids_y >= H)
    ids_x[fltr_out], ids_y[fltr_out] = 0, 0
    ids_b = ids_b.long()[:, None, None].expand(num_pnt, num_div, num_dt)
    dls_rpj = torch.sum((verts + 0.5 - cens[:, None, None, :]) * dirs[:, None, :], dim=-1)
    dls_rpj = torch.clamp_(dls_rpj, min=0)

    padding = (kernel_size - 1) * dilation // 2
    imgs_u = F.unfold(imgsT, kernel_size=kernel_size, padding=padding, dilation=dilation)
    imgs_u = imgs_u.view(N, C, kernel_size * kernel_size, H, W)

    pieces = imgs_u[ids_b, :, :, ids_y, ids_x].contiguous()
    pieces = pieces.view(num_pnt, num_div, num_dt, C, kernel_size * kernel_size)
    power = create_dir_kernel(kernel_size=kernel_size, num_div=num_div, device=imgsT.device)

    heats = torch.sum(pieces * power[:, None, None, :], dim=-1)
    heats = torch.norm(torch.abs(heats), dim=-1)
    heats[fltr_out] = 0
    heats[..., radius] += 1e-7

    max_inds = torch.argmax(heats, dim=-1, keepdim=True)
    dls_tg = torch.gather(dls_rpj, index=max_inds, dim=-1)[..., 0]
    return dls_tg


def featsT_pnt_pool(featsT: torch.Tensor, ids_b: torch.Tensor, censT: torch.Tensor,
                    dlsT_grid: torch.Tensor) -> torch.Tensor:
    num_pnt, num_div, num_samp = dlsT_grid.size()
    N, C, H, W = featsT.size()
    dirs = create_dirsT(num_div, device=featsT.device, bias=0)
    verts = censT[:, None, None, :] + dlsT_grid[..., None] * dirs[:, None, :]
    verts = verts.long()
    ids_x, ids_y = verts[..., 0], verts[..., 1]
    fltr_out = (ids_x < 0) + (ids_x >= W) + (ids_y < 0) + (ids_y >= H)
    ids_x[fltr_out], ids_y[fltr_out] = 0, 0
    ids_b = ids_b.long()[:, None, None].expand(num_pnt, num_div, num_samp)
    samps = featsT[ids_b, :, ids_y, ids_x]
    samps[fltr_out] = 0
    return samps


INSULATOR_COLOR = np.array([142.06330492, 160.62559905, 153.84767562])
INSULATOR_STD = np.array([57.41854421, 53.317534, 56.40438266])


def dlsT_scan_with_pool2(imgsT: torch.Tensor, ids_b: torch.Tensor, censT: torch.Tensor, dlsT: torch.Tensor,
                         low=0.5, high=1.5, num_samp=30, method='loc', select='max') -> (torch.Tensor, torch.Tensor):
    num_inner = round((1 - low) / (high - low) * num_samp)
    dtdl = torch.linspace(low, high, steps=num_samp, device=imgsT.device)
    dls_grid = dlsT[..., None].detach() * dtdl
    colors = featsT_pnt_pool(featsT=imgsT, ids_b=ids_b, censT=censT, dlsT_grid=dls_grid)

    colors_inner = colors[..., :num_inner, :]
    if method == 'loc':
        colors_aver = torch.mean(colors_inner, dim=2, keepdim=True)
    elif method == 'gol':
        colors_aver = torch.mean(colors_inner, dim=(1, 2), keepdim=True)
    elif method == 'aver':
        colors_loc = torch.mean(colors_inner, dim=2, keepdim=True)
        colors_gol = torch.mean(colors_inner, dim=(1, 2), keepdim=True)
        colors_aver = (colors_loc + colors_gol) / 2
    elif method == 'proj':
        colors_loc = torch.mean(colors_inner, dim=2, keepdim=True)
        colors_gol = torch.mean(colors_inner, dim=(1, 2), keepdim=True)
        colors_aver = colors_gol * (torch.sum(colors_gol * colors_loc, dim=-1, keepdim=True) /
                                    torch.sum(colors_gol * colors_gol, dim=-1, keepdim=True))
    else:
        raise Exception('err')

    daver = torch.linalg.norm(colors - colors_aver, dim=-1)
    if select == 'max':
        filler = torch.full(size=(colors.size(0), colors.size(1), 1), device=imgsT.device, fill_value=0)
        heats_diff = torch.cat([filler, daver[..., 1:] - daver[..., :-1]], dim=-1)

        max_val, max_inds = torch.max(heats_diff, dim=-1, keepdim=True)
    elif select == 'sum':
        thres = -torch.std(daver, dim=-1, keepdim=True) + torch.mean(daver, dim=-1, keepdim=True)
        max_inds = torch.sum(daver < thres, dim=-1, keepdim=True).long().clamp_(min=0, max=daver.size(-1) - 1)
        max_val = torch.ones_like(max_inds, device=max_inds.device)
    else:
        raise Exception('err')
    dls_tg = torch.gather(dls_grid, index=max_inds, dim=-1)[..., 0]

    return dls_tg, max_val[..., 0]
    # thres = torch.mean(heats_diff, dim=(1, 2)) + torch.std(heats_diff, dim=(1, 2))
    # return dls_tg, max_val[..., 0] > thres[..., None]


def dlsT_scan_with_pool3(imgsT: torch.Tensor, ids_b: torch.Tensor, censT: torch.Tensor, dlsT: torch.Tensor,
                         low=0.5, high=1.5, num_samp=30, num_super=4) -> (torch.Tensor, torch.Tensor):
    dlsT_smpd = dlsT_samp(dlsT.detach(), num_samp=num_super)
    num_inner = round((1 - low) / (high - low) * num_samp)
    dtdl = torch.linspace(low, high, steps=num_samp, device=imgsT.device)
    dls_grid = dlsT_smpd[..., None] * dtdl
    colors = featsT_pnt_pool(featsT=imgsT, ids_b=ids_b, censT=censT, dlsT_grid=dls_grid)

    colors_inner = colors[..., :num_inner, :]
    colors_loc = torch.mean(colors_inner, dim=2, keepdim=True)
    colors_gol = torch.mean(colors_inner, dim=(1, 2), keepdim=True)
    colors_aver = colors_gol * (torch.sum(colors_gol * colors_loc, dim=-1, keepdim=True) /
                                torch.sum(colors_gol * colors_gol, dim=-1, keepdim=True))

    daver = torch.linalg.norm(colors - colors_aver, dim=-1)

    heats_diff = (daver[..., 1:] * 2 - daver[..., :-1])
    # heats_diff = (daver[..., 1:] - daver[..., :-1]) * daver[..., 1:]
    # heats_diff = daver[..., 1:] ** 2 / daver[..., :-1].clamp(min=1e-5)
    # heats_diff[..., -1] = torch.maximum(heats_diff[..., -1], torch.mean(heats_diff, dim=(-1, -2))[0])

    # show_arrs(heats_diff)
    # plt.pause(1e5)

    max_val, max_inds = torch.max(heats_diff, dim=-1, keepdim=True)
    dls_tg = torch.gather(dls_grid, index=max_inds, dim=-1)
    if num_super > 1:
        shifts = num_super // 2
        size = (dlsT.size(0), dlsT.size(1), num_super)
        dls_tg = torch.mean(torch.roll(dls_tg, dims=1, shifts=shifts).view(size), dim=-1)
        max_val = torch.mean(torch.roll(max_val, dims=1, shifts=shifts).view(size), dim=-1)
    else:
        max_val = max_val.view(dlsT.size())
        dls_tg = dls_tg.view(dlsT.size())
    return dls_tg, max_val


def _build_pooled_feats(feats, scale_base=4, num_pool=2):
    if scale_base == 1:
        feats_base = feats
    else:
        # feats_base = F.avg_pool2d(feats, kernel_size=scale_base, stride=scale_base)
        # feats_base = F.avg_pool2d(feats, kernel_size=5, stride=1)
        feats_base = featsT_bilateral(feats, kernel_size=5, srgb=0.2)
    _, _, H, W = feats_base.size()
    feats_pold = [feats_base]
    buffer = feats_base
    for i in range(1, num_pool):
        buffer = F.avg_pool2d(buffer, kernel_size=3, stride=1, padding=1)
        feats_pold.append(buffer)
    feats_pold = torch.stack(feats_pold, dim=1)
    return feats_pold


def dlsT_scan_with_pool_ly(imgsT: torch.Tensor, ids_b: torch.Tensor, ids_ly: torch.Tensor, censT: torch.Tensor,
                           dlsT: torch.Tensor, low=0.5, high=1.5, num_samp=30, scale_base=4) -> torch.Tensor:
    max_ly = int(torch.max(ids_ly).item() + 1)
    imgsT_pold = _build_pooled_feats(imgsT, scale_base=scale_base, num_pool=max_ly)
    N, _, C, H, W = imgsT_pold.size()
    imgsT_pold = imgsT_pold.view(N * max_ly, C, H, W)
    ids_b_rec = ids_b * max_ly + ids_ly
    dls_tg = dlsT_scan_with_pool(imgsT=imgsT_pold, ids_b=ids_b_rec, censT=censT, dlsT=dlsT,
                                 low=low, high=high, num_samp=num_samp)
    return dls_tg


def xlylsT_hit_refine(xlylsT, num_div=18, num_iter=1, only_squeeze=True):
    num_pnt = xlylsT.size(0)
    dirs = create_dirsT(num_div, device=xlylsT.device, bias=0)
    for i in range(num_iter):
        censT, dlsT = xlylsT2censT_dlsT(xlylsT)
        dts = xlylsT[..., None, :] - censT
        dists = torch.norm(dts, dim=-1)
        ainds = xysT2iasT(dts, num_div=num_div)
        dists_limt = dlsT[torch.arange(num_pnt, device=dlsT.device).expand_as(ainds), ainds]
        xlylsT_repj = dists_limt[..., None] * dirs[ainds] + censT
        fltr_dt = (dists_limt > dists)[..., None]
        if only_squeeze:
            dts_cen = censT[..., None, :] - censT
            fltr_dt *= (torch.sum(dts_cen[:, None] * dts, dim=-1, keepdim=True) > 0)
            # dists_cen = torch.norm(censT[..., None, :] - censT, dim=-1)
            # fltr_dt *= (dists_limt < dists_cen[:, None, :])[..., None]
        xlylsT_dt = torch.where(
            fltr_dt, xlylsT_repj - xlylsT[..., None, :], torch.as_tensor(0).to(xlylsT.device).float())
        xlylsT = torch.sum(xlylsT_dt, dim=-2) / (torch.sum(fltr_dt, dim=-2) + 1) + xlylsT
    return xlylsT


def combine_refine(img, xlylsT, num_div=18, low=0.5, high=1.5, num_samp=30):
    censT, dlsT = xlylsT2censT_dlsT(xlylsT, num_div=num_div)
    dls_col = dlsT_scan_with_pool(
        img[None], ids_b=torch.zeros(xlylsT.size(0), device=xlylsT.device, dtype=torch.long), censT=censT,
        dlsT=dlsT, low=low, high=high, num_samp=num_samp)
    xlyls_col = censT_dlsT2xlylsT(censT, dls_col)
    xlyls_hit = xlylsT_hit_refine(xlylsT, num_div=num_div)
    xlyls_final = (xlyls_col + xlyls_hit) / 2
    return xlyls_final


def xlylsT_filt_by_area(xlylsT, thres=2):
    areas = xlylsT2areasT(xlylsT) + 1e-7
    areas_log = torch.log(areas)
    area_log_aver = torch.mean(areas_log)
    presv_mask = torch.abs(areas_log - area_log_aver) < thres
    return torch.nonzero(presv_mask, as_tuple=True)[0]


def xlylsT_filt_by_area2(xlylsT, min_area=0, max_area=1e5):
    areas = xlylsT2areasT(xlylsT) + 1e-7
    presv_mask = (areas > min_area) * (areas < max_area)
    return torch.nonzero(presv_mask, as_tuple=True)[0]


# if __name__ == '__main__':
#     device = torch.device('cuda:0')
#     img = Image.open(os.path.join(PROJECT_PTH, 'res/insu.png')).convert('RGB')
#     img = np.array(img)
#     imgT = imgN2imgT(img).to(device)
#     cens = torch.Tensor([[250, 200], [400, 400]]).to(device)
#     ids_b = torch.zeros((cens.size(0),)).long().to(device)
#     dls = torch.full(size=(cens.size(0), 8), fill_value=10).to(device)
#     colors = featsT_pnt_pool(imgT, ids_b, cens, dls, num_samp=10)
#
#     censN = cens.detach().cpu().numpy()
#     colorsN = colors.detach().cpu().numpy()
#     axis = plt.subplot()
#     axis.imshow(img)
#     for j in range(cens.size(0)):
#         print(colorsN[j])
#         axis.plot(censN[j, 0], censN[j, 1], 'o', color=colorsN[j], markersize=30)
#     plt.pause(1e5)
#
if __name__ == '__main__':
    device = torch.device('cuda:0')
    img = Image.open(os.path.join(PROJECT_PTH, 'res/insu.png')).convert('RGB')
    img = np.array(img)
    imgT = imgN2imgT(img).to(device)
    heats_edge = imgsT_edge(imgT, kernel_size=7, dilation=1)
    # heats_edge = heatsT_gauss(heats_edge, kernel_size=15, dilation=1)

    heats = imgsT_edge_dir(imgT, kernel_size=7, dilation=1, num_div=8)
    heatsN = heats.detach().cpu().numpy()
    plt.imshow(heatsN[0, 3])
    plt.pause(1e5)
