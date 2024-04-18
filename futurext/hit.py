import os
import sys

PROJECT_NAME = 'Geo'
PROJECT_PTH = os.path.abspath(__file__).split(PROJECT_NAME)[0] + PROJECT_NAME
sys.path.append(PROJECT_PTH)

from utils import *


def hit_loss2(xlylsT, censT, dlsT):
    inner_flag = isin_arr_xlylsT(xysT=censT[None, :, :], xlylsT=xlylsT[:, None, :, :])
    dts = xlylsT[None, :, :] - censT[:, None, None, :]
    alphas = torch.atan2(dts[..., 1], dts[..., 0]) % (2 * np.pi)
    num_div = dlsT.size(1)
    inds = alphas / (2 * np.pi) * num_div
    inds = (inds + 0.5).long() % num_div
    dists = torch.norm(dts, dim=-1)

    dlsT_exp = dlsT[:, None, :].expand_as(inds)
    dls_ref = torch.gather(dlsT_exp, dim=2, index=inds)
    dls_ref, dists = dls_ref[~inner_flag], dists[~inner_flag]
    if dls_ref.size(0) > 0:
        err = (dls_ref - torch.minimum(dls_ref, dists)) / dls_ref
        return torch.mean(err)
    else:
        return torch.as_tensor(0).to(xlylsT.device)


def hit_loss3(censT, dlsT, scales):
    scales_max = torch.maximum(scales[:, None], scales[None, :])

    dts = censT[:, None, :] - censT[None, :, :]
    alphas = torch.atan2(dts[..., 1], dts[..., 0]) % (2 * np.pi)
    num_cen, num_div = dlsT.size()
    inds = alphas / (2 * np.pi) * num_div
    inds_pos = (inds + 0.5).long() % num_div
    inds_neg = (inds + 0.5 + num_div / 2).long() % num_div
    dists = torch.norm(dts, dim=-1)

    dlsT_pos = dlsT[None, :, :].expand(num_cen, num_cen, num_div)
    dlsT_neg = dlsT[:, None, :].expand(num_cen, num_cen, num_div)
    dlsT_pos_ref = torch.gather(dlsT_pos, dim=2, index=inds_pos[..., None])[..., 0]
    dlsT_neg_ref = torch.gather(dlsT_neg, dim=2, index=inds_neg[..., None])[..., 0]

    dists_sum = dlsT_pos_ref + dlsT_neg_ref
    thres = torch.maximum(dlsT_pos_ref, dlsT_neg_ref).detach()
    thres = torch.maximum(thres, scales_max)
    flag_hit = (dists_sum > dists) * (dists > thres)

    dists_sum, dists = dists_sum[flag_hit], dists[flag_hit]

    # print(dists_sum)
    if dists.size(0) > 0:
        err = (dists_sum - dists) / dists
        return torch.mean(err)
    else:
        return torch.as_tensor(0.0).requires_grad_().to(censT.device)


def hit_loss4(censT, dlsT, confsT, ilbsT):
    dts = censT[:, None, :] - censT[None, :, :]
    alphas = torch.atan2(dts[..., 1], dts[..., 0]) % (2 * np.pi)
    num_cen, num_div = dlsT.size()
    inds = alphas / (2 * np.pi) * num_div
    inds_pos = (inds + 0.5).long() % num_div
    inds_neg = (inds + 0.5 + num_div / 2).long() % num_div
    dists = torch.norm(dts, dim=-1)

    dlsT_pos = dlsT[None, :, :].expand(num_cen, num_cen, num_div)
    dlsT_neg = dlsT[:, None, :].expand(num_cen, num_cen, num_div)
    dlsT_pos_ref = torch.gather(dlsT_pos, dim=2, index=inds_pos[..., None])[..., 0]
    dlsT_neg_ref = torch.gather(dlsT_neg, dim=2, index=inds_neg[..., None])[..., 0]

    dists_sum = dlsT_pos_ref.detach() + dlsT_neg_ref
    thres = torch.maximum(dlsT_pos_ref, dlsT_neg_ref).detach()
    fltr_conf = confsT[:, None] > confsT[None, :]
    fltr_clus = ilbsT[:, None] != ilbsT[None, :]
    flag_hit = (dists_sum > dists) * (dists > thres) * fltr_conf * fltr_clus

    err = (dists_sum - dists) * flag_hit / (dists + 1e-7)
    return torch.sum(err, dim=-1)


def hit_limt(censT: torch.Tensor, dlsT: torch.Tensor, ilbsT: torch.Tensor) -> torch.Tensor:
    num_cen, num_div = dlsT.size()
    dts = censT[:, None, :] - censT[None, :, :]
    ias_pos = xysT2iasT(-dts, num_div=num_div)
    ias_neg = xysT2iasT(dts, num_div=num_div)
    dists = torch.norm(dts, dim=-1)
    dlsT_neg = dlsT[:, None, :].expand(num_cen, num_cen, num_div)
    dlsT_neg_ref = torch.gather(dlsT_neg, dim=-1, index=ias_neg[..., None])[..., 0]

    dlsT_limt_part = dists - dlsT_neg_ref
    dlsT_limt_part[ilbsT[:, None] == ilbsT] = np.inf
    dlsT_limt = torch.full(size=(num_cen, num_cen, num_div), fill_value=np.inf, device=censT.device)
    dlsT_limt.scatter_(dim=-1, index=ias_pos[..., None], src=dlsT_limt_part[..., None])

    # 直线划分
    radius = (num_div // 4) - 1
    thetas = (torch.arange(radius * 2 + 1, device=censT.device) - radius) / num_div * np.pi * 2
    scaler = 1 / torch.cos(thetas)
    dlsT_limt_pad = torch.cat([dlsT_limt[..., -radius:], dlsT_limt, dlsT_limt[..., :radius]], dim=-1)
    dlsT_limt_u = F.unfold(dlsT_limt_pad[..., None], kernel_size=(radius * 2 + 1, 1), padding=0)
    dlsT_limt_u = dlsT_limt_u.view(num_cen, num_cen, radius * 2 + 1, num_div)
    dlsT_limt = torch.min(dlsT_limt_u * scaler[..., None], dim=-2)[0]

    dlsT_limt = torch.min(dlsT_limt, dim=-2)[0]
    return dlsT_limt


def hit_limt_cluster(censT: torch.Tensor, dlsT: torch.Tensor, ilbsT: torch.Tensor, iclusT: torch.Tensor):
    iclusT_unq = torch.unique(iclusT, return_inverse=False)
    dls_limt = torch.zeros_like(dlsT, device=dlsT.device)
    for iclu_unq in iclusT_unq:
        fltr = iclu_unq == iclusT
        dlsT_limt_i = hit_limt(censT=censT[fltr], dlsT=dlsT[fltr], ilbsT=ilbsT[fltr])
        dls_limt[fltr] = dlsT_limt_i
    return dls_limt


if __name__ == '__main__':
    device = torch.device('cuda:0')
    censT = torch.Tensor([[10, 10], [40, 15]]).to(device)
    dlsT = torch.full(size=(2, 4), fill_value=20, dtype=torch.float32).to(device)

    dlsT = nn.Parameter(dlsT)

    opt = torch.optim.SGD([dlsT], lr=1)
    for epoch in range(1000):
        opt.zero_grad()
        loss = hit_loss3(censT, dlsT)
        loss.backward()
        opt.step()
        print("epoch {:>3d}: loss = {:>8.3f}".format(epoch, loss))
    print(dlsT)
