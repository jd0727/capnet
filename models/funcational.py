from torch.autograd import Function

from utils import *


# <editor-fold desc='用于旋转检测的roi池化'>


def xywhasT_align(feats, ids_b, xywhas, size, spatial_scale=1.0, kernel_size=2):
    xywhuvs = xywhasT2xywhuvsT(xywhas)
    return xywhuvsT_align(feats, ids_b, xywhuvs, size, spatial_scale=spatial_scale, kernel_size=kernel_size)


def xywhuvsT_align(feats, ids_b, xywhuvs, size, spatial_scale=1.0, kernel_size=2):
    _, C, Hf, Wf = feats.size()
    Ns, _ = xywhuvs.size()
    Wr, Hr = size

    size = torch.Tensor(size).to(feats.device)
    feat_size = torch.Tensor((Wf, Hf)).to(feats.device)
    kernel_offset = create_meshT(kernel_size, kernel_size, feats.device) - kernel_size / 2

    xys, whs, coss, sins = \
        xywhuvs[:, None, :2] * spatial_scale, xywhuvs[:, None, 2:4] * spatial_scale, xywhuvs[:, 4], xywhuvs[:, 5]
    mats = torch.stack([torch.stack([coss, -sins], dim=1), torch.stack([sins, coss], dim=1)], dim=2)

    meshes = (create_meshT(Hr, Wr, device=feats.device) / size - 0.5) * whs
    meshes_proj = (xys + torch.bmm(meshes, mats))[:, :, None, :]
    ixys_samp = torch.floor(meshes_proj.detach() + kernel_offset).long()

    filter_valid = torch.all((ixys_samp >= 0) * (ixys_samp < feat_size), dim=3)
    dxys = 1 / (torch.abs(ixys_samp - meshes_proj + 0.5) ** 2 + 1e-16)
    ixys_samp = torch.where(filter_valid[..., None].expand_as(ixys_samp), ixys_samp,
                            torch.zeros_like(ixys_samp, dtype=torch.long, device=ixys_samp.device))

    pows = (torch.prod(dxys, dim=3) * filter_valid.float())
    pows = pows / (torch.sum(pows, dim=2, keepdim=True) + 1e-16)

    ids_b = ids_b.long()[:, None, None].expand(Ns, Wr * Hr, kernel_offset.size(0))
    samps = feats[ids_b, :, ixys_samp[..., 1], ixys_samp[..., 0]] * pows[..., None]
    samps = torch.sum(samps, dim=2).permute(0, 2, 1).view(Ns, C, Hr, Wr)
    return samps


# </editor-fold>

# <editor-fold desc='用于旋转检测的roi放置'>
def xywhasT_place(feats, ids_b, xywhas, size, spatial_scale=1.0, kernel_size=2):
    xywhuvs = xywhasT2xywhuvsT(xywhas)
    return xywhuvsT_place(feats, ids_b, xywhuvs, size, spatial_scale=spatial_scale, kernel_size=kernel_size)


def xywhuvsT_place(feats, ids_b, xywhuvs, size, spatial_scale=1.0, kernel_size=2):
    _, C, Hf, Wf = feats.size()
    Ns, _ = xywhuvs.size()
    Wr, Hr = size

    feat_size = torch.Tensor((Wf, Hf)).to(feats.device)
    kernel_offset = create_meshT(kernel_size, kernel_size, feats.device) - kernel_size / 2

    xys, whs, coss, sins = \
        xywhuvs[:, None, :2] * spatial_scale, xywhuvs[:, None, 2:4] * spatial_scale, xywhuvs[:, 4], xywhuvs[:, 5]
    mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)

    meshes_proj = torch.bmm(create_meshT(Hr, Wr, device=feats.device) - xys, mats) / whs
    filter_inbox = torch.all((meshes_proj > -0.5) * (meshes_proj < 0.5), dim=2)

    meshes_rproj = (meshes_proj[:, :, None, :] + 0.5) * feat_size
    ixys_samp = torch.floor(meshes_rproj.detach() + kernel_offset).long()

    dxys = 1 / (torch.abs(ixys_samp - meshes_rproj + 0.5) ** 2 + 1e-16)
    filter_valid = torch.all((ixys_samp >= 0) * (ixys_samp < feat_size), dim=3) * filter_inbox[:, :, None]
    ixys_samp = torch.where(filter_valid[..., None].expand_as(ixys_samp), ixys_samp,
                            torch.zeros_like(ixys_samp, dtype=torch.long, device=ixys_samp.device))

    pows = (torch.prod(dxys, dim=3) * filter_valid.float())
    pows = pows / (torch.sum(pows, dim=2, keepdim=True) + 1e-16)

    ids_b = ids_b.long()[:, None, None].expand(Ns, Wr * Hr, kernel_offset.size(0))
    samps = feats[ids_b, :, ixys_samp[..., 1], ixys_samp[..., 0]] * pows[..., None]
    samps = torch.sum(samps, dim=2).permute(0, 2, 1).view(Ns, C, Hr, Wr)
    return samps


# </editor-fold>

# <editor-fold desc='向特定方向最大值投影'>
def rproj(input, alphas, background=0):
    return RProj.apply(input, alphas, background)


class RProj(Function):
    @staticmethod
    def forward(ctx, feats, alphas, background):
        Nb, Hf, Wf = feats.size()
        L = math.ceil(math.sqrt(Wf ** 2 + Hf ** 2))
        thres = torch.min(feats).item() - 1

        coss, sins = torch.cos(alphas), torch.sin(alphas)
        vt = torch.stack([-sins, coss], dim=1)[:, None, :]
        ys, xs = arange2dT(Hf, Wf, device=feats.device)
        meshes = torch.stack([xs + (0.5 - Wf / 2), ys + (0.5 - Hf / 2)], dim=2).view(Wf * Hf, 2).expand(Nb, Wf * Hf, 2)
        idx_proj = torch.sum(meshes * vt, dim=2) + L / 2  # (Nb,Wf * Hf)
        idx_proj = torch.floor(idx_proj).long().view(Nb * Wf * Hf)
        val_proj = torch.full(size=(Nb * Wf * Hf, L), fill_value=thres, device=feats.device)

        feats_p = feats.view(Nb * Wf * Hf)
        val_proj.scatter_(dim=1, index=idx_proj[:, None], src=feats_p[:, None])

        val_proj = val_proj.view(Nb, Wf * Hf, L)
        val_proj, idx_pows = torch.max(val_proj, dim=1)
        filter_hit = val_proj > thres
        val_proj[~filter_hit] = background

        ctx.save_for_backward(idx_pows, feats)
        return val_proj

    @staticmethod
    def backward(ctx, grads_proj):
        idx_pows, feats = ctx.saved_tensors
        Nb, Hf, Wf = feats.size()
        _, L = grads_proj.size()
        grads_feats = torch.zeros((Nb, Hf * Wf), device=grads_proj.device)
        grads_feats.scatter_(dim=1, index=idx_pows, src=grads_proj)
        grads_feats = grads_feats.view(Nb, Hf, Wf)
        return (grads_feats, None, None)


# </editor-fold>

# <editor-fold desc='向特定方向最大投影后求区域均值'>

def mil_r(input, alphas, lmins, lmaxs, background=0, with_log=False):
    return MILR.apply(input, alphas, lmins, lmaxs, background, with_log)


class MILR(Function):
    @staticmethod
    def forward(ctx, feats, alphas, lmins, lmaxs, background, with_log):
        Nb, Hf, Wf = feats.size()
        L = math.ceil(math.sqrt(Wf ** 2 + Hf ** 2))
        thres = torch.min(feats).item() - 1

        coss, sins = torch.cos(alphas), torch.sin(alphas)
        vt = torch.stack([-sins, coss], dim=1)[:, None, :]
        ys, xs = arange2dT(Hf, Wf, device=feats.device)
        meshes = torch.stack([xs + (0.5 - Wf / 2), ys + (0.5 - Hf / 2)], dim=2).view(Wf * Hf, 2).expand(Nb, Wf * Hf, 2)
        idx_proj = torch.sum(meshes * vt, dim=2) + L / 2  # (Nb,Wf * Hf)
        idx_proj = torch.floor(idx_proj).long().view(Nb * Wf * Hf)

        feats_p = feats.view(Nb * Wf * Hf)
        idx_ori = torch.arange(Nb * Wf * Hf, device=feats.device)

        val_pos, idx_pows, pows = MILR.proj_v(
            Nb, Hf, Wf, L, idx_ori, idx_proj, feats_p, thres, background, lmins, lmaxs, with_log)
        ctx.save_for_backward(feats, idx_pows, pows)
        return val_pos

    @staticmethod
    def proj_v(Nb, Hf, Wf, L, idx_ori, idx_proj, feats_p, thres, background, lmins, lmaxs, with_log):
        val_proj = torch.full(size=(Nb * Wf * Hf, L), fill_value=thres, device=feats_p.device)
        val_proj[idx_ori, idx_proj] = feats_p
        val_proj = val_proj.view(Nb, Wf * Hf, L)
        val_proj, idx_pows = torch.max(val_proj, dim=1)
        filter_hit = val_proj > thres
        val_proj[~filter_hit] = background

        idx_l = (torch.arange(L, device=feats_p.device) + 0.5)[None, :].repeat(Nb, 1)  # (Nb,L)
        filter_l = ((idx_l > lmins) * (idx_l < lmaxs))
        filter_pos = filter_l * filter_hit

        if torch.any(filter_pos):
            sum_pos = torch.sum(filter_pos).detach()
            val_pos = val_proj[filter_pos]
            if with_log:
                pows_pos_sub = -1 / (val_proj + 1e-8) / sum_pos
                val_pos = -torch.sum(torch.log(val_pos + 1e-8)) / sum_pos
            else:
                pows_pos_sub = 1 / sum_pos
                val_pos = torch.sum(val_pos) / sum_pos
            pows = pows_pos_sub * filter_pos
        else:
            val_pos = torch.as_tensor(0).to(feats_p.device)
            pows = torch.zeros_like(idx_pows, device=feats_p.device)

        return val_pos, idx_pows, pows

    @staticmethod
    def backward(ctx, grads_pos):
        feats, idx_pows, pows = ctx.saved_tensors
        Nb, Hf, Wf = feats.size()
        grads_feats = torch.zeros((Nb, Hf * Wf), device=grads_pos.device)
        grads_feats.scatter_(dim=1, index=idx_pows, src=grads_pos * pows)
        grads_feats = grads_feats.view(Nb, Hf, Wf)
        return (grads_feats, None, None, None, None, None)


# </editor-fold>


# <editor-fold desc='旋转框内部双向最大投影后求区域均值'>
def xywhasT_mil(input, xywhas, background=0, with_log=False, spatial_scale=1.0, filter_in=None):
    xywhuvs = xywhasT2xywhuvsT(xywhas)
    return MILRRoi.apply(input, xywhuvs, background, with_log, spatial_scale)


def xywhuvsT_mil(input, xywhuvs, background=0, with_log=False, spatial_scale=1.0, filter_in=None):
    # if filter_in is None:
    #     filter_in = torch.full(size=input.size(), fill_value=True, device=input.device, dtype=torch.bool)
    return MILRRoi.apply(input, xywhuvs, background, with_log, spatial_scale)


class MILRRoi(Function):
    @staticmethod
    def forward(ctx, feats, xywhuvs, background, with_log, spatial_scale):
        Nb, Hf, Wf = feats.size()
        L = math.ceil(math.sqrt(Wf ** 2 + Hf ** 2))
        thres = torch.min(feats).item() - 1

        xys, whs, coss, sins = xywhuvs[:, None, 0:2] * spatial_scale, xywhuvs[:, None, 2:4] * spatial_scale, \
                               xywhuvs[:, 4], xywhuvs[:, 5]
        mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)
        xyc = torch.Tensor([Wf / 2, Hf / 2]).to(feats.device)
        xys_proj = (torch.bmm(xys - xyc, mats) + L / 2)

        xmins = xys_proj[..., 0] - whs[..., 0] / 2
        xmaxs = xys_proj[..., 0] + whs[..., 0] / 2
        ymins = xys_proj[..., 1] - whs[..., 1] / 2
        ymaxs = xys_proj[..., 1] + whs[..., 1] / 2

        ys, xs = arange2dT(Hf, Wf, device=feats.device)
        meshes = torch.stack([xs + 0.5, ys + 0.5], dim=2).view(Wf * Hf, 2).expand(Nb, Wf * Hf, 2)
        meshes_proj = torch.bmm(meshes - xyc, mats) + L / 2
        mproj_w = meshes_proj[..., 0]
        mproj_h = meshes_proj[..., 1]

        filter_rgn = (mproj_h > ymins.expand(Nb, Wf * Hf)) * (mproj_h < ymaxs.expand(Nb, Wf * Hf)) \
                     * (mproj_w > xmins.expand(Nb, Wf * Hf)) * (mproj_w < xmaxs.expand(Nb, Wf * Hf))

        feats_p = feats.view(Nb, Wf * Hf)[filter_rgn]
        idx_ori = torch.arange(Nb * Wf * Hf, device=feats.device).view(Nb, Wf * Hf)[filter_rgn]

        val_pos_w, idx_pows_w, pows_w = MILR.proj_v(
            Nb, Hf, Wf, L, idx_ori, torch.floor(mproj_w).long()[filter_rgn], feats_p,
            thres, background, xmins, xmaxs, with_log)

        val_pos_h, idx_pows_h, pows_h = MILR.proj_v(
            Nb, Hf, Wf, L, idx_ori, torch.floor(mproj_h).long()[filter_rgn], feats_p,
            thres, background, ymins, ymaxs, with_log)

        val_pos = (val_pos_w + val_pos_h) / 2
        # print('in loss', val_pos, torch.any(torch.isnan(pows_w)), torch.any(torch.isnan(pows_h)))
        ctx.save_for_backward(feats, idx_pows_w, pows_w, idx_pows_h, pows_h)
        return val_pos

    @staticmethod
    def backward(ctx, grads_pos):
        feats, idx_pows_w, pows_w, idx_pows_h, pows_h = ctx.saved_tensors
        Nb, Hf, Wf = feats.size()
        grads_feats = torch.zeros((Nb, Hf * Wf), device=grads_pos.device)
        grads_feats.scatter_add_(dim=1, index=idx_pows_w, src=grads_pos * pows_w / 2)
        grads_feats.scatter_add_(dim=1, index=idx_pows_h, src=grads_pos * pows_h / 2)
        grads_feats = grads_feats.view(Nb, Hf, Wf)
        return (grads_feats, None, None, None, None, None)


# </editor-fold>

# <editor-fold desc='水平框内部双向最大投影后求区域均值'>
def xyxysT_mil(input, xyxys, background=0, with_log=False, spatial_scale=1.0):
    return MILRoi.apply(input, xyxys, background, with_log, spatial_scale)


def xywhsT_mil(input, xywhs, background=0, with_log=False, spatial_scale=1.0):
    xyxys = xywhsT2xyxysT(xywhs)
    return MILRoi.apply(input, xyxys, background, with_log, spatial_scale)


class MILRoi(Function):
    @staticmethod
    def forward(ctx, feats, xyxys, background, with_log, spatial_scale):
        Nb, Hf, Wf = feats.size()
        ys, xs = arange2dT(Hf, Wf, device=feats.device)
        thres = torch.min(feats).item() - 1
        mesh_w = xs.to(feats.device).contiguous().view(Wf * Hf).expand(Nb, Wf * Hf)
        mesh_h = ys.to(feats.device).contiguous().view(Wf * Hf).expand(Nb, Wf * Hf)
        xyxys = xyxys * spatial_scale
        xmins = xyxys[:, 0:1]
        ymins = xyxys[:, 1:2]
        xmaxs = xyxys[:, 2:3]
        ymaxs = xyxys[:, 3:4]

        filter_rgn = (mesh_h > ymins.expand(Nb, Wf * Hf)) * (mesh_h < ymaxs.expand(Nb, Wf * Hf)) \
                     * (mesh_w > xmins.expand(Nb, Wf * Hf)) * (mesh_w < xmaxs.expand(Nb, Wf * Hf))

        feats_p = feats.view(Nb, Wf * Hf)[filter_rgn]
        idx_ori = torch.arange(Nb * Wf * Hf, device=feats.device).view(Nb, Wf * Hf)[filter_rgn]

        val_pos_w, idx_pows_w, pows_w = MILR.proj_v(
            Nb, Hf, Wf, Wf, idx_ori, mesh_w[filter_rgn], feats_p,
            thres, background, xmins, xmaxs, with_log)

        val_pos_h, idx_pows_h, pows_h = MILR.proj_v(
            Nb, Hf, Wf, Hf, idx_ori, mesh_h[filter_rgn], feats_p,
            thres, background, ymins, ymaxs, with_log)

        val_pos = (val_pos_w + val_pos_h) / 2
        ctx.save_for_backward(feats, idx_pows_w, pows_w, idx_pows_h, pows_h)
        return val_pos

    @staticmethod
    def backward(ctx, grads_pos):
        feats, idx_pows_w, pows_w, idx_pows_h, pows_h = ctx.saved_tensors
        Nb, Hf, Wf = feats.size()
        grads_feats = torch.zeros((Nb, Hf * Wf), device=grads_pos.device)
        grads_feats.scatter_add_(dim=1, index=idx_pows_w, src=grads_pos * pows_w / 2)
        grads_feats.scatter_add_(dim=1, index=idx_pows_h, src=grads_pos * pows_h / 2)
        grads_feats = grads_feats.view(Nb, Hf, Wf)
        return (grads_feats, None, None, None, None, None)


# </editor-fold>



# <editor-fold desc='基于邻近相似度的损失函数'>
def nei_loss_with_img(imgs, feats, filter_in, kernel_size=3, dilation=1, reduction='sum', theta=1.0, thres=2.0):
    Nb, Cf, Hf, Wf = feats.size()
    N, C, H, W = imgs.size()
    if not (Hf == H and Wf == W and Nb == N):
        print(feats.size(), imgs.size())
    assert Hf == H and Wf == W and Nb == N
    padding = (kernel_size - 1) * dilation // 2
    k_idx = kernel_size * (kernel_size // 2) + kernel_size // 2
    imgs_u = F.unfold(imgs, kernel_size=kernel_size, padding=padding, dilation=dilation)
    imgs_u = imgs_u.view(Nb, C, kernel_size * kernel_size, Hf * Wf)
    rnd_col = torch.cat([imgs_u[:, :, :k_idx], imgs_u[:, :, k_idx + 1:]], dim=2)
    cen_col = imgs_u[:, :, k_idx:k_idx + 1]
    pows_sim = torch.exp(-torch.linalg.norm(rnd_col - cen_col, dim=1, keepdim=False) / theta)

    feats_u = F.unfold(feats.view(Nb, Cf, Hf, Wf), kernel_size=kernel_size, padding=padding, dilation=dilation)
    feats_u = feats_u.view(Nb, Cf, kernel_size * kernel_size, Hf * Wf)
    rnd_val = torch.cat([feats_u[:, :, :k_idx], feats_u[:, :, k_idx + 1:]], dim=2)
    cen_val = feats_u[:, :, k_idx:k_idx + 1].repeat(1, 1, kernel_size * kernel_size - 1, 1)

    filter_in = filter_in.view(Nb, 1, Hf * Wf)
    filter_hit = filter_in * (pows_sim > thres)
    if not torch.any(filter_hit): return torch.as_tensor(0).to(feats.device)

    # kl_dist = cen_val * (cen_val.clamp_(min=0.001).log() - rnd_val.clamp_(min=0.001).log())
    kl_dist = cen_val * torch.log(cen_val.clamp_(min=0.001) / rnd_val.clamp_(min=0.001))
    kl_dist = torch.sum(kl_dist, dim=1)
    loss = torch.mean(kl_dist[filter_hit])
    # loss = torch.mean((torch.log(rnd_val + 1e-16) - torch.log(cen_val + 1e-16)) * rnd_val
    #                   + (torch.log(1 - rnd_val + 1e-16) - torch.log(1 - cen_val + 1e-16)) * (1 - rnd_val))
    return loss


def imgsT2unfold_simliar(imgsT, kernel_size=3, dilation=1):
    N, C, H, W = imgsT.size()
    padding = (kernel_size - 1) * dilation // 2
    k_idx = kernel_size * (kernel_size // 2) + kernel_size // 2
    imgs_u = F.pad(imgsT, (padding, padding, padding, padding), mode='reflect')
    imgs_u = F.unfold(imgs_u, kernel_size=kernel_size, padding=0, dilation=dilation)
    imgs_u = imgs_u.view(N, C, kernel_size * kernel_size, H * W)
    rnd_col = torch.cat([imgs_u[:, :, :k_idx], imgs_u[:, :, k_idx + 1:]], dim=2)
    cen_col = imgs_u[:, :, k_idx:k_idx + 1]
    pows_sim = torch.linalg.norm(rnd_col - cen_col, dim=1, keepdim=False)
    return pows_sim


def nei_loss_with_img2(imgs, feats, filter_in, kernel_size=3, dilation=1, reduction='sum', theta=1.0, **kwargs):
    Nb, Cf, Hf, Wf = feats.size()
    N, C, H, W = imgs.size()
    if not (Hf == H and Wf == W and Nb == N):
        print(feats.size(), imgs.size())
    assert Hf == H and Wf == W and Nb == N
    padding = (kernel_size - 1) * dilation // 2
    k_idx = kernel_size * (kernel_size // 2) + kernel_size // 2
    pows_sim = imgsT2unfold_simliar(imgs, kernel_size, dilation)

    feats_u = F.unfold(feats.view(Nb, Cf, Hf, Wf), kernel_size=kernel_size, padding=padding, dilation=dilation)
    feats_u = feats_u.view(Nb, Cf, kernel_size * kernel_size, Hf * Wf)
    rnd_val = torch.cat([feats_u[:, :, :k_idx], feats_u[:, :, k_idx + 1:]], dim=2)
    cen_val = feats_u[:, :, k_idx:k_idx + 1].repeat(1, 1, kernel_size * kernel_size - 1, 1)

    filter_in = filter_in.view(Nb, 1, Hf * Wf).expand(Nb, kernel_size * kernel_size - 1, Hf * Wf)
    thres = torch.sum(pows_sim * filter_in, dim=(1, 2), keepdim=True) \
            / torch.sum(filter_in, dim=(1, 2), keepdim=True).clamp_(min=1)
    filter_hit = filter_in * (pows_sim < thres)
    if not torch.any(filter_hit): return torch.as_tensor(0).to(feats.device)

    mse_loss = torch.abs(cen_val - rnd_val)
    mse_loss = torch.sum(mse_loss, dim=1)
    loss = torch.mean(mse_loss[filter_hit])
    return loss


def crf_loss_with_img(imgs, feats, filter_in, kernel_size=3, dilation=1, sxy=0.5, srgb=0.1, **kwargs):
    Nb, Cf, Hf, Wf = feats.size()
    N, C, H, W = imgs.size()
    assert Hf == H and Wf == W and Nb == N
    pad_s = (kernel_size - 1) * dilation // 2
    kxs, kys = arange2dT(kernel_size, kernel_size, feats.device)
    kernel_offset = torch.sqrt((kxs - kernel_size // 2) ** 2 + (kys - kernel_size // 2) ** 2)
    kernel_offset = torch.abs(kernel_offset).view(1, kernel_size * kernel_size, 1)

    imgs_u = F.pad(imgs, (pad_s, pad_s, pad_s, pad_s), mode='reflect')
    imgs_u = F.unfold(imgs_u, kernel_size=kernel_size, padding=0, dilation=dilation)
    imgs_u = imgs_u.view(N, C, kernel_size * kernel_size, H * W)
    imgs_cen = imgs.view(N, C, 1, H * W)
    pows_sim = torch.linalg.norm(imgs_cen - imgs_u, dim=1, keepdim=False)

    feats_u = F.unfold(feats, kernel_size=kernel_size, padding=pad_s, dilation=dilation)
    feats_u = feats_u.view(Nb, Cf, kernel_size * kernel_size, Hf * Wf)
    feats_cen = feats.view(Nb, Cf, 1, Hf * Wf)
    pairs_l1 = torch.sum(torch.abs(feats_u - feats_cen), dim=1)

    filter_in = filter_in.view(Nb, 1, Hf * Wf)
    if not torch.any(filter_in): return torch.as_tensor(0).to(feats.device)
    pows = torch.exp(-(pows_sim / srgb) ** 2 - (kernel_offset / sxy) ** 2) * filter_in

    loss = torch.sum(pows * pairs_l1) / torch.sum(pows)
    return loss


def crf_loss_with_img3(imgs, feats, filter_in, kernel_size=3, dilation=1, simga=None, **kwargs):
    Nb, Cf, Hf, Wf = feats.size()
    N, C, H, W = imgs.size()
    assert Hf == H and Wf == W and Nb == N
    pad_s = (kernel_size - 1) * dilation // 2

    imgs_u = F.pad(imgs, (pad_s, pad_s, pad_s, pad_s), mode='reflect')
    imgs_u = F.unfold(imgs_u, kernel_size=kernel_size, padding=0, dilation=dilation)
    imgs_u = imgs_u.view(N, C, kernel_size * kernel_size, H * W)
    imgs_cen = imgs.view(N, C, 1, H * W)
    pows_sim = torch.linalg.norm(imgs_cen - imgs_u, dim=1, keepdim=False)

    feats_u = F.unfold(feats.detach(), kernel_size=kernel_size, padding=pad_s, dilation=dilation)
    feats_u = feats_u.view(Nb, Cf, kernel_size * kernel_size, Hf * Wf)
    feats_cen = feats.view(Nb, Cf, 1, Hf * Wf)
    pairs_l1 = torch.sum(torch.abs(feats_u - feats_cen), dim=1)
    # pairs_l1 = torch.sum((feats_u - feats_cen) ** 2, dim=1)

    filter_in = filter_in.view(Nb, 1, Hf * Wf)
    if not torch.any(filter_in): return torch.as_tensor(0).to(feats.device)
    sum_val = torch.sum(filter_in, dim=(1, 2), keepdim=True).clamp_(min=1) * kernel_size * kernel_size
    if simga is None:
        simga = torch.sum(filter_in * pows_sim, dim=(1, 2), keepdim=True) / sum_val.clamp_(min=1) + 1e-5

    pows = torch.exp(-(pows_sim / simga) ** 2) * filter_in

    loss = torch.sum(pows * pairs_l1) / torch.sum(pows)
    return loss


def imgsT_rgb2hsv(imgsT: torch.Tensor) -> torch.Tensor:
    Nb, C, H, W = imgsT.size()
    max_val, max_ind = torch.max(imgsT, dim=1)
    min_val, _ = torch.min(imgsT, dim=1)
    detla_val = (max_val - min_val).clamp_(1e-5)
    max_val = max_val.clamp_(1e-5)
    rsT, gsT, bsT = imgsT[:, 0], imgsT[:, 1], imgsT[:, 2]
    hsT_buff = torch.stack([gsT - bsT, bsT - rsT, rsT - gsT], dim=3) / detla_val[..., None]
    hsT_buff = (hsT_buff + torch.Tensor([0, 2, 4]).to(rsT.device)) * np.pi / 3
    hsT = torch.gather(hsT_buff, dim=1, index=max_ind[..., None]).view(Nb, H, W) % np.pi
    imgsT_hsv = torch.stack([hsT, detla_val / max_val, max_val], dim=1)
    return imgsT_hsv


def hsv_dist(feat, dim):
    feat = torch.abs(feat)
    h, s, v = torch.chunk(feat, dim=dim, chunks=3)
    hd = torch.minimum(h, 2 * np.pi - h) / np.pi
    dist = hd ** 2 + s ** 2 + v ** 2
    return dist


def crf_loss_with_img4(imgs, feats, filter_in, kernel_size=3, dilation=1, srgb=0.1, **kwargs):
    Nb, Cf, Hf, Wf = feats.size()
    N, C, H, W = imgs.size()
    assert Hf == H and Wf == W and Nb == N
    pad_s = (kernel_size - 1) * dilation // 2

    imgs_u = F.pad(imgs, (pad_s, pad_s, pad_s, pad_s), mode='reflect')
    imgs_u = F.unfold(imgs_u, kernel_size=kernel_size, padding=0, dilation=dilation)
    imgs_u = imgs_u.view(N, C, kernel_size * kernel_size, H * W)
    imgs_cen = imgs.view(N, C, 1, H * W)
    pows_sim = torch.linalg.norm(imgs_cen - imgs_u, dim=1, keepdim=False)

    feats_u = F.unfold(feats.detach(), kernel_size=kernel_size, padding=pad_s, dilation=dilation)
    feats_u = feats_u.view(Nb, Cf, kernel_size * kernel_size, Hf * Wf)
    feats_cen = feats.view(Nb, Cf, 1, Hf * Wf)

    pairs_enp = feats_u * torch.log(feats_cen.clamp(min=1e-5)) + \
                (1 - feats_u) * torch.log((1 - feats_cen).clamp(min=1e-5))
    pairs_enp = -torch.sum(pairs_enp, dim=1)

    filter_in = filter_in.view(Nb, 1, Hf * Wf)
    if not torch.any(filter_in): return torch.as_tensor(0).to(feats.device)
    sum_val = torch.sum(filter_in, dim=(1, 2), keepdim=True).clamp_(min=1) * kernel_size * kernel_size
    mean_val = torch.sum(filter_in * pows_sim, dim=(1, 2), keepdim=True) / sum_val.clamp_(min=1) + 1e-5

    pows = torch.exp(-(pows_sim / mean_val) ** 2) * filter_in

    loss = torch.sum(pows * pairs_enp) / torch.sum(pows)
    return loss


# </editor-fold>


# <editor-fold desc='基于图像邻近相似度的损失函数'>
def pairwise_loss_with_img(imgs, feats, xyxys, kernel_size=3, dilation=1, reduction='sum', theta=2, thres=0.5,
                           lab_mode=False, sig_mode=True):
    Nb, Cf, Hf, Wf = feats.size()
    N, C, H, W = imgs.size()
    assert Hf == H and Wf == W and Nb == N
    imgs = imgsT_rgb2lab(imgs) if lab_mode else imgs
    padding = (kernel_size - 1) * dilation // 2
    k_idx = kernel_size * (kernel_size // 2) + kernel_size // 2
    imgs_u = F.unfold(imgs, kernel_size=kernel_size, padding=padding, dilation=dilation)
    imgs_u = imgs_u.view(Nb, C, kernel_size * kernel_size, Hf * Wf)
    rnd_col = torch.cat([imgs_u[:, :, :k_idx], imgs_u[:, :, k_idx + 1:]], dim=2)
    cen_col = imgs_u[:, :, k_idx:k_idx + 1]
    simailar = torch.exp(-torch.norm((rnd_col - cen_col), dim=1, keepdim=True) / theta)
    filter_s = simailar > thres

    ys, xs = arange2dT(Hf, Wf, device=feats.device)
    mesh_w = xs[None, ...].repeat(Nb, 1, 1).view(Nb, Wf * Hf) + 0.5
    mesh_h = ys[None, ...].repeat(Nb, 1, 1).view(Nb, Wf * Hf) + 0.5
    filter_in = (xyxys[:, 0:1] < mesh_w) * (xyxys[:, 2:3] > mesh_w) * (xyxys[:, 1:2] < mesh_h) * (
            xyxys[:, 3:4] > mesh_h)

    feats_u = F.unfold(feats.view(Nb, Cf, Hf, Wf), kernel_size=kernel_size, padding=padding, dilation=dilation)
    feats_u = feats_u.view(Nb, Cf, kernel_size * kernel_size, Hf * Wf)
    rnd_val = torch.cat([feats_u[:, :, :k_idx], feats_u[:, :, k_idx + 1:]], dim=2)
    cen_val = feats_u[:, :, k_idx:k_idx + 1]

    filter_hit = filter_in[:, None, None, :] * filter_s
    if not torch.any(filter_hit): return torch.as_tensor(0).to(feats.device)

    edges_pd = rnd_val * cen_val + (1 - rnd_val) * (1 - cen_val) if sig_mode else \
        torch.sum(rnd_val * cen_val, dim=1, keepdim=True)
    edges_pd = -torch.log(edges_pd[filter_hit] + 1e-16)
    loss = torch.mean(edges_pd) if reduction == 'mean' else torch.sum(edges_pd)
    return loss


# if __name__ == '__main__':
#     device = torch.device('cuda:0')
#     feats = nn.Parameter(torch.rand(size=(2, 20, 20), device=device))
#     imgs = torch.rand(size=(2, 3, 20, 20), device=device)
#     xyxys = nn.Parameter(torch.Tensor([[8, 3, 12, 15], [5, 8, 10, 16]])).to(device)
#     opt = torch.optim.SGD([feats], lr=1)
#     for epoch in range(2000):
#         opt.zero_grad()
#         loss = pairwise_loss_with_img(imgs, torch.sigmoid(feats), xyxys)
#         loss.backward()
#         opt.step()
#         print("epoch {:>3d}: loss = {:>8.3f}".format(epoch, loss))
#     featsN = torch.sigmoid(feats).detach().cpu().numpy()
#     imgsN = torch.sigmoid(imgs).detach().cpu().numpy()
#     plt.imshow(featsN[0, :, :])
#     plt.imshow(imgsN[0].transpose(1, 2, 0))


# </editor-fold>

# <editor-fold desc='最小二乘求区域主方向'>


def feats_xyxys2xywhas_pca(feats, xyxys, conf_thres=None, num_thres=5, wh_thres=0):
    Nb, Hf, Wf = feats.size()
    ys, xs = arange2dT(Hf, Wf, device=feats.device)
    meshes = torch.stack([xs + 0.5, ys + 0.5], dim=2).float().view(Wf * Hf, 2)
    feats = feats.view(Nb, Wf * Hf, 1)
    xywhas = []
    for feat, xyxy in zip(feats, xyxys):
        conf_thres_i = conf_thres if conf_thres is not None else torch.max(feat) / 2
        pnts = meshes[feat[..., 0] > conf_thres_i]
        if not pnts.size(0) > num_thres:
            xywhas.append(xyxyT2xywhaT_align(xyxy))
            continue
        pnts = pnts[(pnts[:, 1] > xyxy[1]) * (pnts[:, 1] < xyxy[3]) * \
                    (pnts[:, 0] > xyxy[0]) * (pnts[:, 0] < xyxy[2])]
        if not pnts.size(0) > num_thres:
            xywhas.append(xyxyT2xywhaT_align(xyxy))
            continue
        U, S, V = torch.pca_lowrank(pnts, q=None, center=True, niter=2)
        alpha = torch.atan2(V[1, 0], V[0, 0])
        pnts_cast = pnts @ V

        min_vals, _ = torch.min(pnts_cast, dim=0)
        max_vals, _ = torch.max(pnts_cast, dim=0)
        xy = (min_vals + max_vals) @ V.T / 2
        wh = max_vals - min_vals
        if not torch.all(wh > wh_thres):
            xywhas.append(xyxyT2xywhaT_align(xyxy))
            continue
        xywha = torch.cat([xy, max_vals - min_vals, torch.Tensor([alpha]).to(feats.device)])
        xywhas.append(xywha)
    xywhas = torch.stack(xywhas, dim=0)
    return xywhas


# def feats_xywhas2xywhas_pca(feats, xywhas, conf_thres=None, num_thres=5, wh_thres=0):
#     Nb, Hf, Wf = feats.size()
#     ys, xs = arange2d(Hf, Wf, device=feats.device)
#     meshes = torch.stack([xs + 0.5, ys + 0.5], dim=2).float().view(Wf * Hf, 2)
#     feats = feats.view(Nb, Wf * Hf, 1)
#     xywhas_reg = []
#     for feat, xywha in zip(feats, xywhas):
#         conf_thres_i = conf_thres if conf_thres is not None else torch.max(feat) / 2
#         pnts = meshes[feat[..., 0] > conf_thres_i]
#         if not pnts.size(0) > num_thres:
#             xywhas_reg.append(xywha)
#             continue
#         cos, sin = torch.cos(xywha[4]), torch.sin(xywha[4])
#         mat = torch.Tensor([[cos, sin], [-sin, cos]]).to(feat.device)
#         pnts_pcast = pnts @ mat.T
#         xy_pcast = xywha[:2] @ mat.T
#         xy_min = xy_pcast - xywha[2:4] / 2
#         xy_max = xy_pcast + xywha[2:4] / 2
#         pnts = pnts[(pnts_pcast[:, 1] > xy_min[1]) * (pnts_pcast[:, 1] < xy_max[1]) * \
#                     (pnts_pcast[:, 0] > xy_min[0]) * (pnts_pcast[:, 0] < xy_max[0])]
#         if not pnts.size(0) > num_thres:
#             xywhas_reg.append(xywha)
#             continue
#         U, S, V = torch.pca_lowrank(pnts, q=None, center=True, niter=2)
#         alpha = torch.atan2(V[1, 0], V[0, 0])
#         pnts_cast = pnts @ V
#
#         min_vals, _ = torch.min(pnts_cast, dim=0)
#         max_vals, _ = torch.max(pnts_cast, dim=0)
#         xy = (min_vals + max_vals) @ V.T / 2
#         wh = max_vals - min_vals
#         if not torch.all(wh > wh_thres):
#             xywhas_reg.append(xywha)
#             continue
#         xywha = torch.cat([xy, max_vals - min_vals, torch.Tensor([alpha]).to(feats.device)])
#         xywhas_reg.append(xywha)
#     xywhas_reg = torch.stack(xywhas_reg, dim=0)
#     return xywhas_reg


def feats_xywhas2xywhas_pca2(feats, xywhas, conf_thres=None, num_thres=5, wh_thres=0):
    Nb, Hf, Wf = feats.size()
    xys, whs, alphas = xywhas[:, None, 0:2], xywhas[:, None, 2:4], xywhas[:, 4]
    coss, sins = torch.cos(alphas), torch.sin(alphas)
    mats = torch.stack([torch.stack([coss, sins], dim=1), torch.stack([-sins, coss], dim=1)], dim=2)
    xys_proj = torch.bmm(xys, mats)

    xmins = xys_proj[..., 0] - whs[..., 0] / 2
    xmaxs = xys_proj[..., 0] + whs[..., 0] / 2
    ymins = xys_proj[..., 1] - whs[..., 1] / 2
    ymaxs = xys_proj[..., 1] + whs[..., 1] / 2

    ys, xs = arange2dT(Hf, Wf, device=feats.device)
    meshes = torch.stack([xs + 0.5, ys + 0.5], dim=2).view(Wf * Hf, 2)
    meshes_proj = torch.bmm(meshes.expand(Nb, Wf * Hf, 2), mats)
    mproj_w = meshes_proj[..., 0]
    mproj_h = meshes_proj[..., 1]

    filters_in = (mproj_h > ymins.expand(Nb, Wf * Hf)) * (mproj_h < ymaxs.expand(Nb, Wf * Hf)) * \
                 (mproj_w > xmins.expand(Nb, Wf * Hf)) * (mproj_w < xmaxs.expand(Nb, Wf * Hf))

    feats_p = feats.view(Nb, Wf * Hf)
    conf_thres = conf_thres if conf_thres is not None else torch.max(feats_p, dim=1, keepdim=True)[0] / 2
    filters_val = feats_p > conf_thres
    filters = filters_val * filters_in

    xywhas_reg = []
    for filter, xywha in zip(filters, xywhas):
        pnts = meshes[filter]
        if not pnts.size(0) > num_thres:
            xywhas_reg.append(xywha)
            continue
        U, S, V = torch.pca_lowrank(pnts, q=None, center=True, niter=2)
        alpha = torch.atan2(V[1, 0], V[0, 0])
        pnts_cast = pnts @ V
        min_vals, _ = torch.min(pnts_cast, dim=0)
        max_vals, _ = torch.max(pnts_cast, dim=0)
        xy = (min_vals + max_vals) @ V.T / 2
        wh = max_vals - min_vals
        if not torch.all(wh > wh_thres):
            xywhas_reg.append(xywha)
            continue
        xywha_reg = torch.cat([xy, max_vals - min_vals, torch.Tensor([alpha]).to(feats.device)])
        xywhas_reg.append(xywha_reg)
    xywhas_reg = torch.stack(xywhas_reg, dim=0)
    return xywhas_reg

# if __name__ == '__main__':
#     device = torch.device('cuda:0')
#     img_size=(100,80)
#     border = XYWHABorder([10, 10, 12, 3, np.pi * 5 / 6], size=img_size)
#     feat = imgN2imgT(np.array(border.maskP)[..., None] * 255)
#     feats = feat[:, 0].to(device).repeat(100, 1, 1)
#     # xyxys = nn.Parameter(torch.Tensor([[3, 3, 18, 15]])).to(device)
#     # xyxysN = xyxys.detach().cpu().numpy()
#     xywhas = nn.Parameter(torch.Tensor([[12, 12, 14, 5, np.pi * 5 / 6]])).to(device).repeat(100, 1)
#     xywhasN = xywhas.detach().cpu().numpy()
#     time1 = time.time()
#     for epoch in range(20):
#         xywhas_reg = feats_xywhas2xywhas_pca(feats, xywhas, conf_thres=None, num_thres=5, wh_thres=0)
#     time2 = time.time()
#     print(time2 - time1)
#
#     xywhasN_reg = xywhas_reg.detach().cpu().numpy()
#
#     featsN = feats.detach().cpu().numpy()
#     axis = show_label(XYWHABorder(xywhasN_reg[0], size=img_size))
#     # show_label(XYXYBorder(xyxysN[0], size=(20, 20)), axis=axis)
#     show_label(XYWHABorder(xywhasN[0], size=img_size), axis=axis)
#     plt.imshow(featsN[0, :, :], extent=(0, img_size[0], img_size[1], 0))
# </editor-fold>

# <editor-fold desc='使用guss函数放置mask'>


#
# if __name__ == '__main__':
#     device = torch.device('cuda:0')
#     img_size = (20, 20)
#     xywhas = torch.Tensor([[10, 10, 12,20, np.pi * 5 / 6]]).repeat(10, 1).to(device)
#     feats = xywhas2masks_cness(xywhas, mask_size=img_size)
#
#     xywhasN = xywhas.detach().cpu().numpy()
#     show_label(XYWHABorder(xywhasN[0], size=img_size))
#     featsN = feats.detach().cpu().numpy()
#     plt.imshow(featsN[0, :, :], extent=(0, img_size[0], img_size[1], 0))


# </editor-fold>
