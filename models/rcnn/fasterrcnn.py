from models.rcnn.modules import *


class FasterRCNN(MTorchModel):

    def __init__(self, backbone, classifier, device=None, pack=None):
        super(FasterRCNN, self).__init__(backbone=backbone, classifier=classifier, device=device, pack=pack)
        self.layers = self.backbone.layers
        self.anchors = self.backbone.anchors.detach().cpu().numpy()
        self.feat_size_clfr = classifier.feat_size

        self.whr_thresh_brd = 6.0  # 预采样长宽比
        self.whr_thresh_naro = 4.0
        self.radius_brd = 4.0
        self.radius_naro = 2.0

        self.pos_thresh = 0.7  # 先验框基于标签转化时正样本iou阈值
        self.neg_thresh = 0.3

        self.pos_num = 256  # 训练中一张图像中roi正样本采样数量
        self.neg_pos_ratio = 1  # 训练中roi负样本采样数量比例

        self.roi_num_max = 2000  # 推理中一张图像中最多roi数量
        self.base = np.exp(np.mean(np.log(self.layers[0].anchor_sizes.numpy())) * 2)

    @property
    def img_size(self):
        return self.backbone.rpn.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size
        self.anchors = self.backbone.anchors.detach().cpu().numpy()

    @property
    def num_cls(self):
        return self.classifier.num_cls

    @staticmethod
    def encode_batch(feat_list):
        ids_b = []
        for i, feat in enumerate(feat_list):
            ids_b.append(torch.full(fill_value=i, size=(feat.size(0),), device=feat.device))
        ids_b = torch.cat(ids_b, dim=0)
        feat_stack = torch.cat(feat_list, dim=0)
        return feat_stack, ids_b

    @staticmethod
    def decode_batch(feat_stack, ids_b, Nb):
        feat_list = []
        for i in range(Nb):
            feat_list.append(feat_stack[ids_b == i])
        return feat_list

    @staticmethod
    def levwise_pool(featmaps, xyxys, ids_b, img_size, base, feat_size=(7, 7)):
        xywhs = xyxysT2xywhsT(xyxys[..., :4])
        areas = xywhs[..., 2] * xywhs[..., 3]
        ids_xyxys = torch.cat([ids_b[:, None], xyxys], dim=1)
        areas_feat = torch.Tensor([featmap.size(2) * featmap.size(3) for featmap in featmaps])
        spatial_scales = list(torch.sqrt(areas_feat / (img_size[0] * img_size[1])))
        ids_f = torch.clamp(
            torch.round((torch.log(areas) / 2 - math.log(base)) / math.log(2)), min=0, max=len(featmaps) - 1)
        order = torch.arange(xyxys.size(0), device=xyxys.device)
        featmaps_roi = []
        order_resorted = []
        for i, (featmap, spatial_scale) in enumerate(zip(featmaps, spatial_scales)):
            featmap_roi = torchvision.ops.roi_align(
                input=featmap, boxes=ids_xyxys[ids_f == i], output_size=feat_size, spatial_scale=spatial_scale)
            featmaps_roi.append(featmap_roi)
            order_resorted.append(order[ids_f == i])

        order_resorted = torch.cat(order_resorted, dim=0)
        order_recv = torch.argsort(order_resorted)
        featmaps_roi = torch.cat(featmaps_roi, dim=0)
        featmaps_roi = featmaps_roi[order_recv]
        return featmaps_roi

    @staticmethod
    def single_pool(featmap, xyxys, ids_b, img_size, feat_size=(7, 7)):
        spatial_scale = math.sqrt(featmap.size(2) * featmap.size(3) / (img_size[0] * img_size[1]))
        ids_xyxys = torch.cat([ids_b[:, None], xyxys], dim=1)
        featmaps_roi = torchvision.ops.roi_align(input=featmap, boxes=ids_xyxys, output_size=feat_size,
                                                 spatial_scale=spatial_scale)
        return featmaps_roi

    def apply_classifier(self, featmaps, xyxys, ids_b):
        if len(self.layers) > 1:
            featmaps_roi = FasterRCNN.levwise_pool(featmaps, xyxys=xyxys, ids_b=ids_b, img_size=self.img_size,
                                                   feat_size=self.feat_size_clfr, base=self.base)
        else:
            featmaps_roi = FasterRCNN.single_pool(featmaps, xyxys=xyxys, ids_b=ids_b, img_size=self.img_size,
                                                  feat_size=self.feat_size_clfr)
        rois_regd = self.pkd_modules['classifier'](featmaps_roi, xyxysT2xywhsT(xyxys))
        return rois_regd

    @staticmethod
    def process_rois(rois, img_size, wh_thres=16.0, conf_thres=0.2, iou_thres=0.4,
                     nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, roi_num_max=128):
        Nb = rois.size(0)
        xyxys_roi = xywhsT2xyxysT(rois[..., :4])
        xyxys_roi[..., slice(0, 4, 2)] = torch.clamp(xyxys_roi[..., slice(0, 4, 2)], min=0, max=img_size[0])
        xyxys_roi[..., slice(1, 4, 2)] = torch.clamp(xyxys_roi[..., slice(1, 4, 2)], min=0, max=img_size[1])
        xywhs_roi = xyxysT2xywhsT(xyxys_roi)
        filters = torch.all(xywhs_roi[..., 2:] > wh_thres, dim=-1) * (rois[..., 4] > conf_thres)
        rois_nmsd = []
        for i in range(Nb):
            xyxys_roi_i = xyxys_roi[i, filters[i], :]
            confs_roi_i = rois[i, filters[i], 4]
            presv_inds = nms_xyxysT(xyxysT=xyxys_roi_i, confsT=confs_roi_i, cindsT=None,
                                    iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            presv_inds = presv_inds[:roi_num_max]
            xyxys_roi_i, confs_roi_i = xyxys_roi_i[presv_inds], confs_roi_i[presv_inds]
            rois_nmsd.append(torch.cat([xyxys_roi_i, confs_roi_i[..., None]], dim=-1))
        return rois_nmsd

    @staticmethod
    def process_rois_regd(rois_regd, img_size, num_cls, wh_thres=16.0, conf_thres=0.2, iou_thres=0.4,
                          nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU):
        rois_rdnd = []
        for roi_regd in rois_regd:
            xyxys_regd = xywhsT2xyxysT(roi_regd[..., :4])
            xyxys_regd[..., slice(0, 4, 2)] = torch.clamp(xyxys_regd[..., slice(0, 4, 2)], min=0, max=img_size[0])
            xyxys_regd[..., slice(1, 4, 2)] = torch.clamp(xyxys_regd[..., slice(1, 4, 2)], min=0, max=img_size[1])
            xywhs_regd = xyxysT2xywhsT(xyxys_regd)
            filters = torch.all(xywhs_regd[..., 2:] > wh_thres, dim=-1) * (roi_regd[..., 4] > conf_thres)
            xyxysT, confsT, cindsT = [], [], []
            for j in range(num_cls):
                xyxys_roi_j = xyxys_regd[filters[:, j], j]
                confs_roi_j = roi_regd[filters[:, j], j, 4]
                presv_inds = nms_xyxysT(xyxysT=xyxys_roi_j, confsT=confs_roi_j, cindsT=None,
                                        iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
                xyxys_roi_j, confs_roi_j = xyxys_roi_j[presv_inds], confs_roi_j[presv_inds]
                cinds_roi_j = torch.full(fill_value=j, size=(xyxys_roi_j.size(0),), device=xyxys_roi_j.device)
                xyxysT.append(xyxys_roi_j)
                confsT.append(confs_roi_j)
                cindsT.append(cinds_roi_j)
            xyxysT, confsT, cindsT = torch.cat(xyxysT, dim=0), torch.cat(confsT, dim=0), torch.cat(cindsT, dim=0)
            roi_rdnd = torch.cat([xyxysT, confsT[:, None], cindsT[:, None]], dim=1)
            rois_rdnd.append(roi_rdnd)
        return rois_rdnd

    def imgs2labels(self, imgs, wh_thres=16.0, rpn_conf_thres=0.2, rpn_iou_thres=0.4, roi_conf_thres=0.2,
                    roi_iou_thres=0.4, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, with_classifier=False,
                    cind2name=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        featmaps, rois = self.pkd_modules['backbone'](imgsT.to(self.device))
        Nb = imgsT.size(0)
        # ROI处理
        rois_nmsd = FasterRCNN.process_rois(
            rois=rois, wh_thres=wh_thres / 2 if with_classifier else wh_thres, conf_thres=rpn_conf_thres,
            iou_thres=rpn_iou_thres, nms_type=nms_type, iou_type=iou_type, img_size=self.img_size,
            roi_num_max=self.roi_num_max)

        if not with_classifier:  # 仅RPN
            labels = [BoxesLabel.from_xyxysT_confsT(
                xyxysT=roi_nmsd[:, :4], confsT=roi_nmsd[:, 4], img_size=self.img_size, num_cls=self.num_cls)
                for roi_nmsd in rois_nmsd]
            return labels_rescale(labels, imgs=imgs, ratios=1 / ratios)
        # 使用分类器
        rois_nmsd, ids_b = FasterRCNN.encode_batch(rois_nmsd)
        rois_regd = self.apply_classifier(featmaps, xyxys=rois_nmsd[..., :4], ids_b=ids_b)
        rois_regd[:, :, 4] = torch.softmax(rois_regd[:, :, 4], dim=1)  # 补充softmax
        rois_regd = FasterRCNN.decode_batch(rois_regd, ids_b, Nb)

        rois_rdnd = FasterRCNN.process_rois_regd(
            rois_regd=rois_regd, wh_thres=wh_thres, conf_thres=roi_conf_thres, iou_thres=roi_iou_thres,
            nms_type=nms_type, iou_type=iou_type, num_cls=self.num_cls, img_size=self.img_size)

        labels = [BoxesLabel.from_xyxysT_confsT_cindsT(
            xyxysT=roi_rdnd[:, :4], confsT=roi_rdnd[:, 4], cindsT=roi_rdnd[:, 5],
            img_size=self.img_size, num_cls=self.num_cls, cind2name=cind2name) for roi_rdnd in rois_rdnd]
        return labels_rescale(labels, imgs=imgs, ratios=1 / ratios)

    @staticmethod
    def layers_dual_match(xywhs_lb, layers, whr_thresh_brd=6.0, whr_thresh_naro=3.0,
                          radius_brd=4.0, radius_naro=2.0, ):
        num_lb = xywhs_lb.shape[0]
        if num_lb == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=bool)
        offset_lb = np.stack(np.meshgrid(
            np.arange(np.floor(-radius_brd), np.ceil(radius_brd) + 1),
            np.arange(np.floor(-radius_brd), np.ceil(radius_brd) + 1)), axis=2).reshape(-1, 2).astype(np.int32)
        # filter_rnd_naro = np.all(np.abs(offset_lb) <= radius_naro, axis=1)
        filter_rnd_naro = np.sqrt(np.sum(offset_lb ** 2, axis=1)) <= radius_naro
        filter_rnd_brd = np.sqrt(np.sum(offset_lb ** 2, axis=1)) <= radius_brd
        ratios = []
        filter_posi = []
        ids = []
        offset_layer = 0
        for k, layer in enumerate(layers):
            Na, stride, Wf, Hf, num_anchor = layer.Na, layer.stride, layer.Wf, layer.Hf, layer.num_anchor
            anchor_sizes = layer.anchor_sizes.numpy()
            # 长宽比过滤
            ratios_layer = xywhs_lb[:, None, 2:4] / anchor_sizes[None, :, :]
            # 位置过滤
            max_offsets = xywhs_lb[:, None, 2:4] / stride / 2
            offset_lb_layer = np.broadcast_to(offset_lb, (num_lb, Na, offset_lb.shape[0], 2))
            filter_offset = np.all((offset_lb_layer >= -max_offsets[:, :, None, :]) *
                                   (offset_lb_layer <= max_offsets[:, :, None, :]), axis=-1)
            # 中心点计算
            ixy = (xywhs_lb[:, None, None, :2] // stride).astype(np.int32) + offset_lb
            filter_map = (ixy[..., 0] >= 0) * (ixy[..., 0] < Wf) * (ixy[..., 1] >= 0) * (ixy[..., 1] < Hf)
            ids_layer = (ixy[..., 1] * Wf + ixy[..., 0]) * Na + np.arange(Na)[None, :, None]

            ratios.append(ratios_layer)
            filter_posi.append(filter_offset * filter_map * filter_rnd_brd)
            ids.append(ids_layer + offset_layer)
            offset_layer += num_anchor
        ratios = np.concatenate(ratios, axis=1)
        filter_posi = np.concatenate(filter_posi, axis=1)
        ids = np.concatenate(ids, axis=1)

        filter_ratio_bdr = np.all((ratios < whr_thresh_brd) * (ratios > 1 / whr_thresh_brd), axis=2)
        filter_ratio_naro = np.all((ratios < whr_thresh_naro) * (ratios > 1 / whr_thresh_naro), axis=2)
        idx_min = np.argmin(np.sum(np.abs(np.log(ratios)), axis=2), axis=1)
        filter_ratio_bdr[np.arange(num_lb), idx_min] = True
        filter_ratio_naro[np.arange(num_lb), idx_min] = True
        ids_lb_pos, idxl_ancr, idxl_rnd = np.nonzero(filter_posi * filter_ratio_bdr[..., None])

        ids_ancr_pos = ids[ids_lb_pos, idxl_ancr, idxl_rnd]
        candidtate = filter_rnd_naro[idxl_rnd] * filter_ratio_naro[ids_lb_pos, idxl_ancr]
        return ids_lb_pos, ids_ancr_pos, candidtate

    def labels2tars(self, labels, **kwargs):
        flgs_tg_unfold = [np.full(shape=0, dtype=bool, fill_value=False)]
        xywhs_tg_unfold = [np.zeros(shape=(0, 4), dtype=np.float32)]
        cinds_tg_unfold = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_neg = [np.zeros(shape=0, dtype=np.int32)]
        inds_neg = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            assert isinstance(label, InstsLabel)
            xywhs_lb, cinds_lb = label.export_xywhsN_cindsN()
            masks_im = label.export_masksN_enc(self.img_size, num_cls=self.num_cls)[None, :]
            # 每层稀疏采样
            ids_lb_pos, ids_ancr_pos, flgs_im = FasterRCNN.layers_dual_match(
                xywhs_lb, self.layers, whr_thresh_brd=self.whr_thresh_brd, whr_thresh_naro=self.whr_thresh_naro,
                radius_brd=self.radius_brd, radius_naro=self.radius_naro)
            # 计算负样本
            ids_ancr_neg = np.delete(np.arange(self.anchors.shape[0]), ids_ancr_pos)
            num_samp = self.neg_pos_ratio * max(ids_ancr_pos.shape[0], self.pos_num)
            samper = np.random.choice(a=ids_ancr_neg.shape[0], size=min(ids_ancr_neg.shape[0], num_samp), replace=False)
            ids_ancr_neg = ids_ancr_neg[samper]
            # 追加
            flgs_tg_unfold.append(flgs_im)
            xywhs_tg_unfold.append(xywhs_lb[ids_lb_pos])
            cinds_tg_unfold.append(cinds_lb[ids_lb_pos])
            inds_b_pos.append(np.full(fill_value=i, shape=ids_ancr_pos.shape[0]))
            inds_pos.append(ids_ancr_pos)
            inds_b_neg.append(np.full(fill_value=i, shape=ids_ancr_neg.shape[0]))
            inds_neg.append(ids_ancr_neg)

        flgs_tg_unfold = np.concatenate(flgs_tg_unfold, axis=0)
        xywhs_tg_unfold = np.concatenate(xywhs_tg_unfold, axis=0)
        cinds_tg_unfold = np.concatenate(cinds_tg_unfold, axis=0)

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_b_neg = np.concatenate(inds_b_neg, axis=0)
        inds_neg = np.concatenate(inds_neg, axis=0)

        targets = (
            inds_b_pos, inds_pos, inds_b_neg, inds_neg, flgs_tg_unfold, xywhs_tg_unfold, cinds_tg_unfold)
        return targets

    @staticmethod
    def xywh_loss(xywhs_pd, xywhs_tg):
        xy_loss = F.smooth_l1_loss(xywhs_pd[:, :2] / xywhs_tg[:, 2:], xywhs_tg[:, :2] / xywhs_tg[:, 2:],
                                   reduction='mean', beta=1 / 9)
        wh_loss = F.smooth_l1_loss(torch.log(xywhs_pd[:, 2:]), torch.log(xywhs_tg[:, 2:]),
                                   reduction='mean', beta=1 / 9)
        return (xy_loss + wh_loss) / 2

    @staticmethod
    @torch.no_grad()
    def balanced_samp(cinds: torch.Tensor, num_samp: int):
        if cinds.size(0) == 0:
            return torch.zeros(size=(0,), device=cinds.device, dtype=torch.long)
        cinds_u, cnts = torch.unique(cinds, return_counts=True)
        if cinds_u.size(0) == 1:
            return torch.randperm(cinds.size(0))[:num_samp]
        # 确定类别采样数量
        if num_samp >= cinds.size(0):
            return torch.arange(cinds.size(0))
        order = torch.argsort(cnts)
        cinds_u, cnts = cinds_u[order], cnts[order]
        high = torch.cumsum(cnts, dim=0) + torch.arange(cnts.size(0) - 1, -1, -1, device=cnts.device) * cnts
        low = torch.cat([torch.zeros(size=(1,), device=high.device), high[:-1]], dim=0)
        idx_samp = torch.nonzero((num_samp > low) * (num_samp <= high))[0]
        # 处理剩余采样指标
        fill_val = 0 if idx_samp == 0 else cnts[idx_samp - 1]
        divider = int(cinds_u.size(0) - idx_samp)
        lev, res = divmod(num_samp - int(low[idx_samp]), divider)
        cnts[idx_samp:] = fill_val + lev
        cnts[idx_samp:idx_samp + int(res)] += 1
        # 执行采样
        ids_samp = []
        for i, cind in enumerate(cinds_u):
            ids_i = torch.nonzero(cind == cinds)[:, 0]
            if i >= idx_samp:
                ids_i = ids_i[torch.randperm(ids_i.size(0))[:cnts[i]]]
            ids_samp.append(ids_i)
        ids_samp = torch.cat(ids_samp, dim=0)
        return ids_samp

    def backbone2loss(self, imgs, inds_b_pos, inds_pos, xywhs_tg_unfold, flgs_tg_unfold, inds_b_neg, inds_neg,
                      neg_pos_ratio=1):
        featmaps, rois = self.pkd_modules['backbone'](imgs)
        rois_pos = rois[inds_b_pos, inds_pos]
        xywhs_pd_neg = rois[inds_b_neg, inds_neg, :4]
        xywhs_pd_pos, confs_pos = rois_pos[..., :4], rois_pos[..., 4]
        # RPN部分
        if inds_b_pos.size(0) > 0:
            rpn_bdr_loss = FasterRCNN.xywh_loss(
                xywhs_pd=xywhs_pd_pos[flgs_tg_unfold], xywhs_tg=xywhs_tg_unfold[flgs_tg_unfold])
            rpn_pos_loss = -torch.mean(torch.log(confs_pos[flgs_tg_unfold] + 1e-16))
        else:
            rpn_bdr_loss = torch.as_tensor(0).to(rois.device)
            rpn_pos_loss = torch.as_tensor(0).to(rois.device)
        samper = torch.randperm(inds_b_neg.size(0))[:max(neg_pos_ratio * torch.sum(flgs_tg_unfold), 1)]
        inds_b_neg, inds_neg = inds_b_neg[samper], inds_neg[samper]
        if inds_b_neg.size(0) > 0:
            rpn_neg_loss = -torch.mean(torch.log(1 - rois[inds_b_neg, inds_neg, 4] + 1e-16))
        else:
            rpn_neg_loss = torch.as_tensor(0).to(rois.device)
        rpn_conf_loss = (rpn_pos_loss + rpn_neg_loss) / 2
        return rpn_conf_loss, rpn_bdr_loss, featmaps, xywhs_pd_pos, xywhs_pd_neg

    @staticmethod
    def match_refine(xywhs_pd, inds_b_pos, inds_pos, xywhs_tg_unfold, flgs_tg_unfold, pos_thresh, neg_thresh, Nb,
                     num_anchor):
        rpn_ious = ropr_arr_xywhsT(xywhs1=xywhs_pd, xywhs2=xywhs_tg_unfold, opr_type=OPR_TYPE.IOU)
        filter_near = (rpn_ious > pos_thresh)

        filter_appr = (rpn_ious > neg_thresh)
        unmatched = torch.full(size=(Nb, num_anchor), fill_value=True, device=xywhs_pd.device)
        unmatched[inds_b_pos[filter_appr], inds_pos[filter_appr]] = False
        filter_far = unmatched[inds_b_pos, inds_pos]
        return filter_near, filter_far

    @staticmethod
    def resamp_with_filter(inds_b_pos, xywhs_pd_pos, cinds_tg_unfold, xywhs_tg_unfold, inds_b_neg, xywhs_pd_neg,
                           filter_near, filter_far, num_pos=256, neg_pos_ratio=1):
        filter_near = torch.nonzero(filter_near, as_tuple=True)[0]
        samper_pos = torch.randperm(filter_near.size(0))[:num_pos]
        filter_near = filter_near[samper_pos]

        inds_b_neg, xywhs_pd_neg = torch.cat([inds_b_neg, inds_b_pos[filter_far]], dim=0), \
                                   torch.cat([xywhs_pd_neg, xywhs_pd_pos[filter_far]], dim=0)

        samper_neg = torch.randperm(inds_b_neg.size(0))[:neg_pos_ratio * filter_near.size(0)]
        inds_b_neg, xywhs_pd_neg = inds_b_neg[samper_neg], xywhs_pd_neg[samper_neg]

        inds_b_pos, cinds_tg_unfold, xywhs_tg_unfold, xywhs_pd_pos = \
            inds_b_pos[filter_near], cinds_tg_unfold[filter_near], xywhs_tg_unfold[filter_near], \
            xywhs_pd_pos[filter_near]
        return inds_b_pos, xywhs_pd_pos, cinds_tg_unfold, xywhs_tg_unfold, inds_b_neg, xywhs_pd_neg

    def classifier2loss(self, featmaps, inds_b_pos, xywhs_pd_pos, xywhs_pd_neg, cinds_tg_unfold, xywhs_tg_unfold,
                        inds_b_neg):
        xywhs_samp = torch.cat([xywhs_pd_pos, xywhs_pd_neg], dim=0)
        inds_b_samp = torch.cat([inds_b_pos, inds_b_neg], dim=0)
        cinds_tg_neg = torch.full(fill_value=self.num_cls, size=inds_b_neg.size(), device=inds_b_neg.device)
        if inds_b_pos.size(0) > 0:
            rois_regd_samp = self.apply_classifier(featmaps, xyxys=xywhsT2xyxysT(xywhs_samp), ids_b=inds_b_samp)
            xywhs_regd_clfr = rois_regd_samp[torch.arange(inds_b_pos.size(0)), cinds_tg_unfold, :4]
            cinds_tg_samp = torch.cat([cinds_tg_unfold, cinds_tg_neg], dim=0)
            roi_conf_loss = F.cross_entropy(rois_regd_samp[:, :, 4], cinds_tg_samp, reduction='mean')
            roi_bdr_loss = FasterRCNN.xywh_loss(xywhs_pd=xywhs_regd_clfr, xywhs_tg=xywhs_tg_unfold)
        else:
            roi_conf_loss = torch.as_tensor(0).to(xywhs_pd_pos.device)
            roi_bdr_loss = torch.as_tensor(0).to(xywhs_pd_pos.device)
        return roi_conf_loss, roi_bdr_loss

    def imgs_tars2loss(self, imgs, targets, with_classifier=True, with_masker=True):
        imgs = imgs.to(self.device)

        inds_b_pos, inds_pos, inds_b_neg, inds_neg, \
        flgs_tg_unfold, xywhs_tg_unfold, cinds_tg_unfold = targets

        inds_b_pos = torch.as_tensor(inds_b_pos, dtype=torch.long).to(imgs.device)
        inds_pos = torch.as_tensor(inds_pos, dtype=torch.long).to(imgs.device)
        inds_b_neg = torch.as_tensor(inds_b_neg, dtype=torch.long).to(imgs.device)
        inds_neg = torch.as_tensor(inds_neg, dtype=torch.long).to(imgs.device)

        flgs_tg_unfold = torch.as_tensor(flgs_tg_unfold, dtype=torch.bool).to(imgs.device)
        xywhs_tg_unfold = torch.as_tensor(xywhs_tg_unfold, dtype=torch.float).to(imgs.device)
        cinds_tg_unfold = torch.as_tensor(cinds_tg_unfold, dtype=torch.long).to(imgs.device)

        # 计算主干损失
        rpn_conf_loss, rpn_bdr_loss, featmaps, xywhs_pd_pos, xywhs_pd_neg = self.backbone2loss(
            imgs, inds_b_pos, inds_pos, xywhs_tg_unfold, flgs_tg_unfold, inds_b_neg, inds_neg, neg_pos_ratio=1)
        loss_dict = OrderedDict(rpn_conf=rpn_conf_loss, rpn_bdr=rpn_bdr_loss)
        if not with_classifier:
            return loss_dict
        #  RPN正负样本动态匹配调整
        filter_near, filter_far = FasterRCNN.match_refine(
            xywhs_pd_pos, inds_b_pos, inds_pos, xywhs_tg_unfold, flgs_tg_unfold, self.pos_thresh, self.neg_thresh,
            Nb=imgs.size(0), num_anchor=self.anchors.shape[0])
        inds_b_pos, xywhs_pd_pos, cinds_tg_unfold, xywhs_tg_unfold, inds_b_neg, xywhs_pd_neg = FasterRCNN.resamp_with_filter(
            inds_b_pos, xywhs_pd_pos, cinds_tg_unfold, xywhs_tg_unfold, inds_b_neg, xywhs_pd_neg, filter_near,
            filter_far, num_pos=self.pos_num * imgs.size(0), neg_pos_ratio=1)
        roi_conf_loss, roi_bdr_loss = self.classifier2loss(featmaps, inds_b_pos, xywhs_pd_pos, xywhs_pd_neg,
                                                           cinds_tg_unfold, xywhs_tg_unfold, inds_b_neg)
        loss_dict.update(OrderedDict(roi_bdr=roi_bdr_loss, roi_conf=roi_conf_loss))
        return loss_dict

    def export_onnx_trt(self, **kwargs):
        return None

    def export_onnx(self, onnx_pth_bkbn, onnx_pth_clfr, batch_size_bkbn=1, batch_size_clfr=300):
        return None

    @staticmethod
    def VGGA(num_cls=20, img_size=(224, 224), device=None, pack=None):
        backbone = RPNOneScaleVGGBkbn(**VGGBkbn.PARA_A, img_size=img_size, act=ACT.RELU,norm=NORM.BATCH)
        classifier = FasterRCNNMLPClassifier(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, in_channels=512, feat_size=(7, 7))
        return FasterRCNN(backbone=backbone, classifier=classifier, device=device, pack=pack)

    @staticmethod
    def ResNetR18(num_cls=20, img_size=(224, 224), device=None, pack=None):
        backbone = RPNMutiScaleResNetBkbn.R18(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH)
        classifier = FasterRCNNResNetClassifier.R18(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, feat_size=(14, 14))
        return FasterRCNN(backbone=backbone, classifier=classifier, device=device, pack=pack)

    @staticmethod
    def ResNetR50(num_cls=20, img_size=(224, 224), device=None, pack=None):
        backbone = RPNMutiScaleResNetBkbn.R50(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH)
        classifier = FasterRCNNResNetClassifier.R50(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, feat_size=(14, 14))
        return FasterRCNN(backbone=backbone, classifier=classifier, device=device, pack=pack)

    @staticmethod
    def ResNetR101(num_cls=20, img_size=(224, 224), device=None, pack=None):
        backbone = RPNMutiScaleResNetBkbn.R101(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH)
        classifier = FasterRCNNResNetClassifier.R101(num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH, feat_size=(14, 14))
        return FasterRCNN(backbone=backbone, classifier=classifier, device=device, pack=pack)

    @staticmethod
    def Const(num_cls=20, img_size=(224, 224), device=None, pack=None, batch_size=1):
        channels = (num_cls + 1) * 5 + num_cls
        backbone = FasterRCNNConstBkbn(batch_size=batch_size, img_size=img_size, out_channels=channels)
        classifier = FasterRCNNConstClassifier(num_cls=num_cls, in_channels=channels, feat_size=(7, 7))
        return FasterRCNN(backbone=backbone, classifier=classifier, device=device, pack=pack)


if __name__ == '__main__':
    model = FasterRCNN.ResNetR50(num_cls=20, img_size=(224, 224), device='cpu', pack=None)
    # x = torch.rand(2, 3, 224, 224)
    # y = model.imgs2boxss(x, with_classifier=True)
