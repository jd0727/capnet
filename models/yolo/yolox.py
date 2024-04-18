from models.base.darknet import DarkNetV5Bkbn
from models.yolo.modules import *
from models.yolo.yolov5 import YoloV5DownStream, YoloV5UpStream, YoloV5Main
from models.modules import *


class YoloXLayer(AnchorImgLayer):
    def __init__(self, in_channels, inner_channels, stride, num_cls, img_size=(512, 512), act=ACT.LK, norm=NORM.BATCH):
        super().__init__(stride=stride, anchor_sizes=((stride, stride)), img_size=img_size)
        self.num_cls = num_cls
        self.stem = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, act=act, norm=norm)
        self.xywh_conf = nn.Sequential(
            Ck3s1NA(in_channels=inner_channels, out_channels=inner_channels, act=act, norm=norm),
            Ck3s1NA(in_channels=inner_channels, out_channels=inner_channels, act=act, norm=norm)
        )
        self.xywh = Ck1(in_channels=inner_channels, out_channels=4)
        self.conf = Ck1(in_channels=inner_channels, out_channels=1)
        self.cind = nn.Sequential(
            Ck3s1NA(in_channels=inner_channels, out_channels=inner_channels, act=act, norm=norm),
            Ck3s1NA(in_channels=inner_channels, out_channels=inner_channels, act=act, norm=norm),
            Ck1(in_channels=inner_channels, out_channels=self.num_cls)
        )
        init_sig(bias=self.conf.conv.bias, prior_prob=0.01)

    def forward(self, featmap):
        featmap = self.stem(featmap)
        xywh_conf = self.xywh_conf(featmap)
        xywh = self.xywh(xywh_conf)
        conf = self.conf(xywh_conf)
        cind = self.cind(featmap)
        pred = YoloXLayer.decode(xywh=xywh, conf=conf, cind=cind, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
                                 stride=self.stride, num_cls=self.num_cls)
        return pred

    @staticmethod
    def decode(xywh, conf, cind, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(xywh.device, non_blocking=True)
        wh_offset = wh_offset.to(xywh.device, non_blocking=True)
        Hf, Wf, _ = xy_offset.size()
        cind = cind.permute(0, 2, 3, 1)
        xywh = xywh.permute(0, 2, 3, 1)
        conf = conf.permute(0, 2, 3, 1)

        x = (xywh[..., 0] + 0.5 + xy_offset[..., 0]) * stride
        y = (xywh[..., 1] + 0.5 + xy_offset[..., 1]) * stride
        w = torch.exp(xywh[..., 2]) * wh_offset[..., 0]
        h = torch.exp(xywh[..., 3]) * wh_offset[..., 1]

        cind = torch.sigmoid(cind)
        conf = torch.sigmoid(conf)
        pred = torch.cat([x[..., None], y[..., None], w[..., None], h[..., None], conf, cind], dim=-1).contiguous()
        pred = pred.reshape(-1, Wf * Hf, num_cls + 5)
        return pred


class YoloXMain(DarkNetV5Bkbn):

    def __init__(self, Module, channels, repeat_nums, num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(256, 256),
                 in_channels=3):
        super(YoloXMain, self).__init__(Module, channels, repeat_nums, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        feat_channelss = (channels * 4, channels * 8, channels * 16)
        down_channelss = (channels * 4, channels * 4, channels * 8)
        up_channelss = (channels * 4, channels * 8, channels * 16)
        self.down = YoloV5DownStream(Module, feat_channelss, down_channelss, repeat_num=repeat_nums[-1], act=act,
                                     norm=norm)
        self.up = YoloV5UpStream(Module, down_channelss, up_channelss, repeat_num=repeat_nums[-1], act=act, norm=norm)
        self.layers = self.layers = nn.ModuleList([
            YoloXLayer(in_channels=channels * 4, inner_channels=channels * 4, stride=8,
                       num_cls=num_cls, img_size=img_size, act=act, norm=norm),
            YoloXLayer(in_channels=channels * 8, inner_channels=channels * 4, stride=16,
                       num_cls=num_cls, img_size=img_size, act=act, norm=norm),
            YoloXLayer(in_channels=channels * 16, inner_channels=channels * 4, stride=32,
                       num_cls=num_cls, img_size=img_size, act=act, norm=norm)
        ])

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        for layer in self.layers:
            layer.img_size = img_size

    def forward(self, imgs):
        feats0 = self.pre(imgs)
        feats1 = self.stage1(feats0)
        feats2 = self.stage2(feats1)
        feats3 = self.stage3(feats2)
        feats4 = self.stage4(feats3)
        feats = (feats2, feats3, feats4)
        feats = self.down(feats)
        feats = self.up(feats)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def Nano(num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(512, 512)):
        return YoloXMain(**YoloV5Main.NANO_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def Small(num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(512, 512)):
        return YoloXMain(**YoloV5Main.SMALL_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def Medium(num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(512, 512)):
        return YoloXMain(**YoloV5Main.MEDIUM_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def Large(num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(512, 512)):
        return YoloXMain(**YoloV5Main.LARGE_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.LK, norm=NORM.BATCH, img_size=(512, 512)):
        return YoloXMain(**YoloV5Main.XLARGE_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size)


class YoloXConstLayer(AnchorImgLayer):
    def __init__(self, batch_size, stride, num_cls, img_size=(512, 512)):
        super().__init__(stride=stride, anchor_sizes=((stride, stride)), img_size=img_size)
        self.num_cls = num_cls
        self.featmap = nn.Parameter(torch.zeros(batch_size, num_cls + 5, self.Hf, self.Wf))
        init_sig(bias=self.featmap[:, 4, :, :], prior_prob=0.01)

    def forward(self, featmap):
        featmap = self.featmap
        cls = featmap[:, 5:, :, :]
        box = featmap[:, :4, :, :]
        conf = featmap[:, 4:5, :, :]
        pred = YoloXLayer.decode(
            xywh=box, conf=conf, cind=cls,
            xy_offset=self.xy_offset, wh_offset=self.wh_offset, stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloXConstMain(nn.Module):
    def __init__(self, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloXConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloXConstLayer(batch_size=batch_size, stride=8, num_cls=num_cls, img_size=img_size),
            YoloXConstLayer(batch_size=batch_size, stride=16, num_cls=num_cls, img_size=img_size),
            YoloXConstLayer(batch_size=batch_size, stride=32, num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloX(YoloFrame):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super().__init__(backbone=backbone, device=device, pack=pack)
        self.layers = backbone.layers
        self.with_xywh_loss = False
        self.radius = 2.5

    # 动态匹配策略
    def labels2tars(self, labels, **kwargs):
        targets = []
        for i, label in enumerate(self.labels):
            inds_ancr = [np.zeros(shape=0, dtype=np.int32)]
            inds_lb = [np.zeros(shape=0, dtype=np.int32)]
            inds_layer = [np.zeros(shape=0, dtype=np.int32)]
            mask = [np.full(shape=0, fill_value=False)]
            xyxys, chots = label.export_xyxysN_chotsN(num_cls=self.num_cls)
            xywhs = xyxysN2xywhsN(xyxys)

            num_lb = xywhs.shape[0]
            offset_layer = 0
            for j, layer in enumerate(self.layers):
                num_anchor_layer = layer.num_anchor
                stride, Wf, Hf = layer.stride, layer.Wf, layer.Hf
                # 生成偏移
                xs, ys = np.meshgrid(np.arange(Wf), np.arange(Hf))
                offset = np.concatenate([xs[:, :, None, None], ys[:, :, None, None]], axis=-1)  # (Hf, Wf, 1, 2)
                # 匹配gt包含的区域
                grids_lb = np.broadcast_to(xyxys / stride - 0.5, shape=(Hf, Wf, num_lb, 4))
                masks_lb = np.concatenate([grids_lb[..., :2] - offset, offset - grids_lb[..., 2:]], axis=-1)
                masks_lb = np.all(masks_lb < 0, axis=-1)
                # 匹配以gt为中心，step半径的区域
                grids_lbr = np.broadcast_to(xywhs[:, :2] / stride - 0.5, shape=(Hf, Wf, num_lb, 2))
                masks_lbr = np.concatenate([grids_lbr - offset, offset - grids_lbr], axis=-1)
                masks_lbr = np.all(masks_lbr < self.radius, axis=-1)
                # 求交集并集
                iy, ix, ids_lb = np.nonzero(masks_lb | masks_lbr)
                ids = iy * Wf + ix
                mask_j = (masks_lb & masks_lbr)[iy, ix, ids_lb]
                # 添加
                inds_layer.append(np.full(fill_value=j, shape=len(ids)))
                inds_lb.append(ids_lb)
                inds_ancr.append(offset_layer + ids)
                mask.append(mask_j)
                offset_layer = offset_layer + num_anchor_layer

            inds_ancr = np.concatenate(inds_ancr, axis=0)
            inds_ancr, indinds_ancr = np.unique(inds_ancr, return_inverse=True)
            inds_lb = np.concatenate(inds_lb, axis=0)
            mask = np.concatenate(mask, axis=0)
            inds_layer = np.concatenate(inds_layer, axis=0)
            targets.append((inds_ancr, indinds_ancr, inds_lb, mask, xywhs, chots, inds_layer))
        return targets

    # 动态计算loss
    def imgs_tars2loss(self, imgs, targets, with_xywh_loss=False, **kwargs):
        imgs = imgs.to(self.device)
        preds = self.pkd_modules['backbone'](imgs)
        # 动态匹配
        xywh_pred = []
        cind_pred = []
        conf_pred_pos = []
        xywh = []
        cind = []
        conf = []
        num_ancr = preds.size(1)
        for i, (ids_ancr, idids_ancr, ids_lb, mask, xywh_i, chot_i, ids_layer) in enumerate(targets):
            # 无目标情况
            if ids_ancr.shape[0] == 0:
                conf.append(torch.zeros(size=(num_ancr,), device=preds.device))
                continue
            with torch.no_grad():
                xywh_i = torch.as_tensor(xywh_i, dtype=torch.float).to(preds.device, non_blocking=True)
                chot_i = torch.as_tensor(chot_i, dtype=torch.float).to(preds.device, non_blocking=True)
                ids_lb = torch.as_tensor(ids_lb, dtype=torch.long).to(preds.device, non_blocking=True)
                ids_ancr = torch.as_tensor(ids_ancr, dtype=torch.long).to(preds.device, non_blocking=True)
                idids_ancr = torch.as_tensor(idids_ancr, dtype=torch.long).to(preds.device, non_blocking=True)
                mask = torch.as_tensor(mask, dtype=torch.float).to(preds.device, non_blocking=True)

                predR = preds[i, ids_ancr[idids_ancr], :]
                predR_xywh, predR_conf, predR_cls = predR[:, :4], predR[:, 4:5], predR[:, 5:]
                # 计算cost
                ious = ropr_arr_xywhsT(predR_xywh, xywh_i[ids_lb], opr_type=OPR_TYPE.IOU)
                iou_loss = -torch.log(ious + 1e-16)
                cls_loss = F.binary_cross_entropy(torch.sqrt(predR_conf * predR_cls), chot_i[ids_lb])
                cost = cls_loss + iou_loss * 3 + (1 - mask) * 10000  # 使同时具备中心特性的pbox优先级更高
                # 确定匹配数量 k个最小cost匹配
                iou_mat = torch.zeros(size=(xywh_i.size(0), ids_ancr.size(0)), device=preds.device)
                iou_mat[ids_lb, idids_ancr] = ious
                ks = torch.sum(torch.topk(iou_mat, k=10, dim=1)[0], dim=1).int() + 1
                # 选取
                cost_mat = torch.full(size=(xywh_i.size(0), ids_ancr.size(0)), fill_value=40000.0, device=preds.device)
                cost_mat[ids_lb, idids_ancr] = cost
                for j in range(xywh_i.size(0)):
                    ids = torch.topk(cost_mat[j], largest=False, k=ks[j].item())[1]
                    cost_mat[j, ids] -= 20000
                    cost_mat[j, ids[0]] -= 10000
                # 去除重复
                _, idids_pbox_pos = torch.nonzero(cost_mat < 0, as_tuple=True)
                idids_pbox_pos = torch.unique(idids_pbox_pos)
                ids_gt_pos = torch.argmin(cost_mat[:, idids_pbox_pos], dim=0)
                ids_pbox_pos = ids_ancr[idids_pbox_pos]

                conf_i = torch.zeros(size=(num_ancr,), device=preds.device)
                conf_i[ids_pbox_pos] = 1
                conf.append(conf_i)
                xywh.append(xywh_i[ids_gt_pos])
                cind.append(chot_i[ids_gt_pos])
                # print(i, ids_gt_pos, predd[i, ids_pbox_pos, 4])
            predM = preds[i, ids_pbox_pos, :]
            xywh_pred.append(predM[:, :4])
            cind_pred.append(predM[:, 5:])
            conf_pred_pos.append(predM[:, 4])

        xywh_pred = torch.cat(xywh_pred, dim=0)
        cind_pred = torch.cat(cind_pred, dim=0)
        conf_pred_pos = torch.cat(conf_pred_pos, dim=0)
        xywh = torch.cat(xywh, dim=0)
        cind = torch.cat(cind, dim=0)
        conf = torch.stack(conf, dim=0)

        pred_conf = preds[:, :, 4]
        Nm = xywh.size(0)
        # iou损失
        ious = ropr_arr_xywhsT(xywh_pred, xywh, opr_type=OPR_TYPE.IOU)
        iou_loss = torch.sum(1 - ious ** 2) / Nm * 5
        # print(ids_gt_pos)
        # 分类损失
        cls_loss = F.binary_cross_entropy(cind_pred, cind * ious.detach()[:, None], reduction='sum') / Nm
        # cls_loss = F.binary_cross_entropy(chots_pred, chots_tg, reduction='sum') / Nm
        # 目标检出损失
        # obj_loss = F.binary_cross_entropy(pred_conf, conf, reduction='sum') / Nm
        pos_loss = -torch.sum(torch.log(conf_pred_pos + 1e-16)) / Nm * 5
        neg_loss = -torch.sum((1 - conf) * torch.log(1 - pred_conf)) / Nm
        losses = OrderedDict(pos=pos_loss, neg=neg_loss, iou=iou_loss, cls=cls_loss)
        if with_xywh_loss:  # xywh损失
            xy_loss = F.l1_loss(xywh_pred[:, :2], xywh[:, :2], reduction='sum') / Nm
            wh_loss = F.l1_loss(torch.log(xywh_pred[:, 2:]), torch.log(xywh[:, 2:]), reduction='sum') / Nm
            losses['xywhs_tg'] = xy_loss + wh_loss
        return losses

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloXMain.Nano(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size)
        return YoloX(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloXMain.Small(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size)
        return YoloX(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloXMain.Medium(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size)
        return YoloX(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloXMain.Large(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size)
        return YoloX(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def XLarge(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloXMain.XLarge(num_cls=num_cls, act=ACT.LK, norm=NORM.BATCH, img_size=img_size)
        return YoloX(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloXConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size)
        return YoloX(backbone=backbone, device=device, pack=PACK.NONE)

    # @staticmethod
    # def ONNX(onnx_pth, device=None):
    #     from deploy.onnx import ONNXModule
    #     backbone = ONNXModule(onnx_pth=onnx_pth, device=device)
    #     norm = (backbone.input_size[3], backbone.input_size[2])
    #     return YoloInferN(backbone=backbone, norm=norm)
    #
    # @staticmethod
    # def TRT(trt_pth):
    #     from deploy.trt import TRTModule
    #     backbone = TRTModule(trt_pth=trt_pth)
    #     norm = (backbone.input_size[3], backbone.input_size[2])
    #     return YoloInferN(backbone=backbone, norm=norm)


if __name__ == '__main__':
    model = YoloX.Std(img_size=(416, 416), device=1)
    imgs = torch.rand(2, 3, 416, 416)
    y = model(imgs)
