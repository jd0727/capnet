from models.rcnn.fasterrcnn import *


class MaskRCNN(FasterRCNN):
    def __init__(self, backbone, classifier, masker, device=None, pack=None):
        super(MaskRCNN, self).__init__(backbone=backbone, classifier=classifier, device=device, pack=pack)
        # self.layers = self.backbone.layers
        # self.anchors = self.backbone.anchors.detach().cpu().numpy()
        # self.feat_size_clfr = classifier.feat_size
        self.add_module(masker=masker)
        self.feat_size_mskr = masker.feat_size
        self.mask_size = masker.mask_size

        self.whr_thresh_brd = 6.0  # 预采样长宽比
        self.whr_thresh_naro = 4.0
        self.radius_brd = 4.0
        self.radius_naro = 2.0

        self.pos_thresh = 0.7  # 先验框基于标签转化时正样本iou阈值
        self.neg_thresh = 0.3  # 先验框基于标签转化时负样本iou阈值

        self.pos_num = 256  # 训练中一张图像中roi正样本采样数量
        self.neg_pos_ratio = 1  # 训练中roi负样本采样数量比例

        self.roi_num_max = 2000  # 推理中一张图像中最多roi数量
        self.base = np.exp(np.mean(np.log(self.layers[0].anchor_sizes.numpy())) * 2)

    def apply_masker(self, featmaps, xyxys, ids_b):
        if len(self.layers) > 1:
            featmaps_roi = FasterRCNN.levwise_pool(featmaps, xyxys=xyxys, ids_b=ids_b, img_size=self.img_size,
                                                   feat_size=self.feat_size_clfr, base=self.base)
        else:
            featmaps_roi = FasterRCNN.single_pool(featmaps, xyxys=xyxys, ids_b=ids_b, img_size=self.img_size,
                                                  feat_size=self.feat_size_clfr)
        maskss = self.pkd_modules['masker'](featmaps_roi)
        return maskss

    @torch.no_grad()
    def imgs2labels(self, imgs, wh_thres=16.0, rpn_conf_thres=0.2, rpn_iou_thres=0.4, roi_conf_thres=0.2,
                    roi_iou_thres=0.4, masker_conf_thres=0.3, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU,
                    with_classifier=True, with_masker=True, cind2name=None, **kwargs):
        self.eval()
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        Nb = imgsT.size(0)
        featmaps, rois = self.pkd_modules['backbone'](imgsT.to(self.device))
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
        if not with_masker:  # 仅目标检测
            return labels_rescale(labels, imgs=imgs, ratios=1 / ratios)
        rois_rdnd, ids_b = FasterRCNN.encode_batch(rois_rdnd)
        maskss = self.apply_masker(featmaps, xyxys=rois_rdnd[..., :4], ids_b=ids_b)
        maskss = FasterRCNN.decode_batch(maskss, ids_b, Nb)
        labels = [InstsLabel.from_boxes_masksT_ref(
            boxes=boxes, masksT=masks, conf_thres=masker_conf_thres) for boxes, masks in zip(labels, maskss)]
        return labels_rescale(labels, imgs=imgs, ratios=1 / ratios)

    def labels2tars(self, labels, **kwargs):
        masks_tg = [np.zeros(shape=(0, self.img_size[1], self.img_size[0]), dtype=np.float32)]
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
            masks_tg.append(masks_im)
            flgs_tg_unfold.append(flgs_im)
            xywhs_tg_unfold.append(xywhs_lb[ids_lb_pos])
            cinds_tg_unfold.append(cinds_lb[ids_lb_pos])
            inds_b_pos.append(np.full(fill_value=i, shape=ids_ancr_pos.shape[0]))
            inds_pos.append(ids_ancr_pos)
            inds_b_neg.append(np.full(fill_value=i, shape=ids_ancr_neg.shape[0]))
            inds_neg.append(ids_ancr_neg)

        masks_tg = np.concatenate(masks_tg, axis=0)
        flgs_tg_unfold = np.concatenate(flgs_tg_unfold, axis=0)
        xywhs_tg_unfold = np.concatenate(xywhs_tg_unfold, axis=0)
        cinds_tg_unfold = np.concatenate(cinds_tg_unfold, axis=0)

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_b_neg = np.concatenate(inds_b_neg, axis=0)
        inds_neg = np.concatenate(inds_neg, axis=0)

        targets = (
            inds_b_pos, inds_pos, masks_tg, inds_b_neg, inds_neg, flgs_tg_unfold, xywhs_tg_unfold, cinds_tg_unfold)
        return targets

    def masker2loss(self, featmaps, inds_b_pos, xywhs_pd_pos, masks_tg):
        if inds_b_pos.size(0) > 0:
            xyxys_pos = xywhsT2xyxysT(xywhs_pd_pos)
            with torch.no_grad():
                masks_tg_chot = torch.zeros(
                    size=(masks_tg.size(0), self.num_cls + 1, masks_tg.size(1), masks_tg.size(2)),
                    device=masks_tg.device)
                filler = torch.ones_like(masks_tg, device=masks_tg.device, dtype=torch.float32)
                masks_tg_chot.scatter_(dim=1, index=masks_tg[:, None, :, :], src=filler[:, None, :, :])
                masks_tg_unfold = torchvision.ops.roi_align(
                    masks_tg_chot, boxes=torch.cat([inds_b_pos[:, None], xyxys_pos], dim=1),
                    output_size=self.mask_size, spatial_scale=1.0)[:, :self.num_cls]
            masks_pd = self.apply_masker(featmaps, xyxys=xyxys_pos, ids_b=inds_b_pos)
            roi_mask_loss = F.binary_cross_entropy(masks_pd, masks_tg_unfold, reduction='mean')
        else:
            roi_mask_loss = torch.as_tensor(0).to(masks_tg.device)
        return roi_mask_loss

    def imgs_tars2loss(self, imgs, targets, with_classifier=True, with_masker=True):
        imgs = imgs.to(self.device)

        inds_b_pos, inds_pos, masks_tg, inds_b_neg, inds_neg, \
        flgs_tg_unfold, xywhs_tg_unfold, cinds_tg_unfold = targets

        inds_b_pos = torch.as_tensor(inds_b_pos, dtype=torch.long).to(imgs.device)
        inds_pos = torch.as_tensor(inds_pos, dtype=torch.long).to(imgs.device)
        inds_b_neg = torch.as_tensor(inds_b_neg, dtype=torch.long).to(imgs.device)
        inds_neg = torch.as_tensor(inds_neg, dtype=torch.long).to(imgs.device)
        masks_tg = torch.as_tensor(masks_tg, dtype=torch.long).to(imgs.device)

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
        if not with_masker:
            return loss_dict
        roi_mask_loss = self.masker2loss(featmaps, inds_b_pos, xywhs_pd_pos, masks_tg)
        loss_dict.update(OrderedDict(roi_mask=roi_mask_loss))
        return loss_dict

    def export_onnx(self, onnx_dir, prifix, batch_size_bkbn=1, batch_size_inner=300):
        W, H = self.img_size
        onnx_pth_bkbn = os.path.join(onnx_dir, prifix + '_bkbn')
        model2onnx(self.backbone, onnx_pth_bkbn, input_size=(batch_size_bkbn, 3, H, W))
        # Wf, Hf = self.classifier.feat_size
        # Cf = self.classifier.in_channels
        # onnx_pth_clfr = os.path.join(onnx_dir, prifix + '_clfr')
        # model2onnx(self.classifier, onnx_pth_clfr, input_size=(batch_size_inner, Cf, Hf, Wf))
        # Wf, Hf = self.masker.feat_size
        # Cf = self.masker.in_channels
        # onnx_pth_mskr = os.path.join(onnx_dir, prifix + '_mskr')
        # model2onnx(self.masker, onnx_pth_mskr, input_size=(batch_size_inner, Cf, Hf, Wf))
        return self

    def export_onnx_trt(self, **kwargs):
        pass

    @staticmethod
    def ResNetR18(num_cls=20, img_size=(224, 224), device=None, pack=None, anchor_ratio=8):
        backbone = RPNMutiScaleResNetBkbn.R18(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=anchor_ratio)
        channels = ResNetBkbn.PARA_R18['channels']
        classifier = FasterRCNNMLPClassifier(in_channels=channels, num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH)
        masker = MaskRCNNRepeatConvMasker(in_channels=channels, num_cls=num_cls, repeat_num=4, act=ACT.RELU,norm=NORM.BATCH)
        return MaskRCNN(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=pack)

    @staticmethod
    def ResNetR50(num_cls=20, img_size=(224, 224), device=None, pack=None, anchor_ratio=8):
        backbone = RPNMutiScaleResNetBkbn.R50(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=anchor_ratio)
        channels = ResNetBkbn.PARA_R50['channels']
        classifier = FasterRCNNMLPClassifier(in_channels=channels, num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH)
        masker = MaskRCNNRepeatConvMasker(in_channels=channels, num_cls=num_cls, repeat_num=4, act=ACT.RELU,norm=NORM.BATCH)
        return MaskRCNN(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=pack)

    @staticmethod
    def ResNetR101(num_cls=20, img_size=(224, 224), device=None, pack=None, anchor_ratio=8):
        backbone = RPNMutiScaleResNetBkbn.R101(img_size=img_size, act=ACT.RELU,norm=NORM.BATCH, anchor_ratio=anchor_ratio)
        channels = ResNetBkbn.PARA_R101['channels']
        classifier = FasterRCNNMLPClassifier(in_channels=channels, num_cls=num_cls, act=ACT.RELU,norm=NORM.BATCH)
        masker = MaskRCNNRepeatConvMasker(in_channels=channels, num_cls=num_cls, repeat_num=4, act=ACT.RELU,norm=NORM.BATCH)
        return MaskRCNN(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=pack)

    @staticmethod
    def Const(num_cls=20, img_size=(224, 224), device=None, batch_size=1, anchor_ratio=8):
        channels = (num_cls + 1) * 5 + num_cls
        backbone = MaskRCNNConstBkbn(img_size=img_size, batch_size=batch_size, out_channels=channels,
                                     anchor_ratio=anchor_ratio)
        classifier = FasterRCNNConstClassifier(num_cls=num_cls, in_channels=channels)
        masker = MaskRCNNConstMasker(num_cls=num_cls, in_channels=channels)
        return MaskRCNN(backbone=backbone, classifier=classifier, masker=masker, device=device, pack=PACK.NONE)


if __name__ == '__main__':
    model = MaskRCNN.ResNetR101(img_size=(800, 800), num_cls=20, device='cpu')
    # model.img_size = (256, 256)
    model.export_onnx(onnx_dir='/home/user/JD/Public/export', prifix='mskr101', batch_size_bkbn=1,
                      batch_size_inner=1)

# if __name__ == '__main__':
#     arr = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3]).to(torch.device('cuda:0'))
#     for i in range(10):
#         ids_samp = MaskRCNN.balanced_samp(arr, num_samp=6)
#         print(arr[ids_samp])
