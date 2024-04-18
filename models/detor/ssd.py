from models.base.resnet import ResNetBkbn, Bottleneck
from models.base.vgg import VGGBkbn
from models.modules import *
from models.template import OneStageTorchModel
from utils.frame import *


class L2Norm(nn.Module):
    def __init__(self, channels, scale):
        super(L2Norm, self).__init__()
        self.channels = channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSDLayer(AnchorLayer):
    def __init__(self, in_channels, anchor_sizes, stride, num_cls=20, feat_size=(0, 0)):
        super(SSDLayer, self).__init__(anchor_sizes=anchor_sizes, stride=stride, feat_size=feat_size)
        self.num_cls = num_cls
        self.reg_xywh = Ck1(in_channels=in_channels, out_channels=self.Na * 4)
        self.reg_iconf_chot = Ck1(in_channels=in_channels, out_channels=self.Na * (num_cls + 1))

    @staticmethod
    def layers(in_channelss, min_sizes, max_sizes, ratioss, strides, feat_sizes, num_cls):
        layers = nn.ModuleList()
        for in_channels, min_size, max_size, ratios, stride, feat_size in \
                zip(in_channelss, min_sizes, max_sizes, ratioss, strides, feat_sizes):
            layers.append(SSDLayer(
                in_channels=in_channels, stride=stride, num_cls=num_cls, feat_size=feat_size,
                anchor_sizes=SSDLayer.generate_anchor_sizes(min_size=min_size, max_size=max_size, ratios=ratios)))
        return layers

    @staticmethod
    def generate_anchor_sizes(min_size, max_size, ratios):
        anchors = []
        anchors.append([min_size, min_size])
        wid = math.sqrt(max_size * min_size)
        anchors.append([wid, wid])
        ratios = ratios if isinstance(ratios, Iterable) else [ratios]
        for ratio in ratios:
            ratio = math.sqrt(ratio)
            anchors.append([min_size * ratio, min_size / ratio])
            anchors.append([min_size / ratio, min_size * ratio])
        return anchors

    def forward(self, featmap):
        reg_xywh = self.reg_xywh(featmap)
        reg_iconf_chot = self.reg_iconf_chot(featmap)
        pred = SSDLayer.decode(
            reg_xywh=reg_xywh, reg_iconf_chot=reg_iconf_chot, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred

    @staticmethod
    def decode(reg_xywh, reg_iconf_chot, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(reg_xywh.device, non_blocking=True)
        wh_offset = wh_offset.to(reg_xywh.device, non_blocking=True)
        Hf, Wf, Na, _ = xy_offset.size()
        reg_xywh = reg_xywh.permute(0, 2, 3, 1)
        reg_xywh = reg_xywh.reshape(-1, Hf, Wf, Na, 4)

        x = (reg_xywh[..., 0] * 0.1 + xy_offset[..., 0]) * stride
        y = (reg_xywh[..., 1] * 0.1 + xy_offset[..., 1]) * stride
        w = torch.exp(reg_xywh[..., 2] * 0.2) * wh_offset[..., 0]
        h = torch.exp(reg_xywh[..., 3] * 0.2) * wh_offset[..., 1]

        reg_iconf_chot = reg_iconf_chot.permute(0, 2, 3, 1)
        reg_iconf_chot = reg_iconf_chot.reshape(-1, Hf, Wf, Na, num_cls + 1)
        # reg_iconf_chot = torch.softmax(reg_iconf_chot, dim=-1)

        pred = torch.cat([x[..., None], y[..., None], w[..., None], h[..., None], reg_iconf_chot], dim=-1)
        pred = torch.reshape(pred, shape=(-1, Hf * Wf * Na, num_cls + 5))
        return pred


class SSDConstLayer(AnchorLayer):
    def __init__(self, anchor_sizes, stride, batch_size=1, num_cls=20, feat_size=(0, 0)):
        super(SSDConstLayer, self).__init__(anchor_sizes=anchor_sizes, stride=stride, feat_size=feat_size)
        self.num_cls = num_cls
        self.featmap = nn.Parameter(torch.zeros(batch_size, self.Na * (num_cls + 5), self.Hf, self.Wf))

    @staticmethod
    def layers(min_sizes, max_sizes, ratioss, strides, feat_sizes, num_cls, batch_size=1):
        layers = nn.ModuleList()
        for min_size, max_size, ratios, stride, feat_size in \
                zip(min_sizes, max_sizes, ratioss, strides, feat_sizes):
            layers.append(SSDConstLayer(
                stride=stride, num_cls=num_cls, feat_size=feat_size, batch_size=batch_size,
                anchor_sizes=SSDLayer.generate_anchor_sizes(min_size=min_size, max_size=max_size, ratios=ratios)))
        return layers

    def forward(self, featmap):
        reg_xywh = self.featmap[:, :self.Na * 4, :, :]
        reg_iconf_chot = self.featmap[:, self.Na * 4:, :, :]
        pred = SSDLayer.decode(
            reg_xywh=reg_xywh, reg_iconf_chot=reg_iconf_chot, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred


class SSDVGGMain(VGGBkbn):
    def __init__(self, repeat_nums, channelss, strides, num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        super(SSDVGGMain, self).__init__(repeat_nums, channelss, strides, act)
        self.num_cls = num_cls
        self.post2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Ck3NA(in_channels=512, out_channels=1024, stride=1, dilation=6, act=act, norm=norm),
            Ck1NA(in_channels=1024, out_channels=1024, act=act, norm=norm)
        )
        self.post3 = nn.Sequential(
            Ck1NA(in_channels=1024, out_channels=256, act=act, norm=norm),
            Ck3NA(in_channels=256, out_channels=512, stride=2, act=act, norm=norm)
        )
        self.post4 = nn.Sequential(
            Ck1NA(in_channels=512, out_channels=128, act=act, norm=norm),
            Ck3NA(in_channels=128, out_channels=256, stride=2, act=act, norm=norm)
        )
        self.post5 = nn.Sequential(
            Ck1A(in_channels=256, out_channels=128, act=act),
            Ck3A(in_channels=128, out_channels=256, act=act)
        )
        self.post6 = nn.Sequential(
            Ck1A(in_channels=256, out_channels=128, act=act),
            Ck3A(in_channels=128, out_channels=256, act=act)
        )
        self.layers = SSDLayer.layers(
            in_channelss=(512, 1024, 512, 256, 256, 256),
            min_sizes=(30, 60, 111, 162, 213, 264),
            max_sizes=(60, 111, 162, 213, 264, 315),
            ratioss=(2, (2, 3), (2, 3), (2, 3), 2, 2),
            strides=(8, 16, 32, 64, 100, 300),
            feat_sizes=SSDVGGMain.calc_feat_sizes(img_size),
            num_cls=num_cls
        )
        self.img_size = img_size

    @staticmethod
    def calc_feat_sizes(img_size):
        W, H = img_size
        feat_sizes = []
        for stride in (8, 16, 32, 64):
            feat_sizes.append([int(math.ceil(W / stride)), int(math.ceil(H / stride))])
        feat_sizes.append((feat_sizes[-1][0] - 2, feat_sizes[-1][1] - 2))
        feat_sizes.append((feat_sizes[-1][0] - 2, feat_sizes[-1][1] - 2))
        return feat_sizes

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        feat_sizes = SSDVGGMain.calc_feat_sizes(img_size)
        for layer, feat_size in zip(self.layers, feat_sizes):
            layer.feat_size = feat_size

    def forward(self, imgs):
        feat = self.stage1(imgs)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat1 = self.stage4(feat)
        feat2 = self.stage5(feat1)
        feat2 = self.post2(feat2)
        feat3 = self.post3(feat2)
        feat4 = self.post4(feat3)
        feat5 = self.post5(feat4)
        feat6 = self.post6(feat5)
        feats = (feat1, feat2, feat3, feat4, feat5, feat6)
        preds = [layer(feat) for layer, feat in zip(self.layers, feats)]
        preds = torch.cat(preds, dim=1)
        return preds

    @staticmethod
    def A(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDVGGMain(**VGGBkbn.PARA_A, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def D(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDVGGMain(**VGGBkbn.PARA_D, num_cls=num_cls, act=act, norm=norm, img_size=img_size)


class SSDResNetMain(ResNetBkbn):
    def __init__(self, Module, repeat_nums, channels, num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        super(SSDResNetMain, self).__init__(Module, repeat_nums, channels, act=act, norm=norm)
        self.num_cls = num_cls
        self._img_size = img_size
        self.post4 = Bottleneck(in_channels=channels * 8, out_channels=channels * 4, stride=2, act=act, norm=norm)
        self.post5 = CNA(in_channels=channels * 4, out_channels=channels * 2, kernel_size=3,
                         stride=1, act=act, norm=norm)
        self.post6 = CNA(in_channels=channels * 2, out_channels=channels, kernel_size=3,
                         stride=1, act=act, norm=norm)
        self.norm = L2Norm(channels=channels * 2, scale=20)
        self.layers = SSDLayer.layers(
            in_channelss=(channels * 2, channels * 4, channels * 8, channels * 4, channels * 2, channels),
            min_sizes=(30, 60, 111, 162, 213, 264),
            max_sizes=(60, 111, 162, 213, 264, 315),
            ratioss=(2, (2, 3), (2, 3), (2, 3), 2, 2),
            strides=(8, 16, 32, 64, 100, 300),
            feat_sizes=SSDResNetMain.calc_feat_sizes(img_size),
            num_cls=num_cls,
        )

    @staticmethod
    def calc_feat_sizes(img_size):
        W, H = img_size
        feat_sizes = []
        for stride in (8, 16, 32, 64):
            feat_sizes.append([int(math.ceil(W / stride)), int(math.ceil(H / stride))])
        feat_sizes.append((feat_sizes[-1][0] - 2, feat_sizes[-1][1] - 2))
        feat_sizes.append((feat_sizes[-1][0] - 2, feat_sizes[-1][1] - 2))
        return feat_sizes

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        feat_sizes = SSDResNetMain.calc_feat_sizes(img_size)
        for layer, feat_size in zip(self.layers, feat_sizes):
            layer.feat_size = feat_size

    def forward(self, imgs):
        feat = self.pre(imgs)
        feat = self.stage1(feat)
        feat1 = self.stage2(feat)
        feat2 = self.stage3(feat1)
        feat3 = self.stage4(feat2)
        feat1 = self.norm(feat1)
        feat4 = self.post4(feat3)
        feat5 = self.post5(feat4)
        feat6 = self.post6(feat5)
        feats = (feat1, feat2, feat3, feat4, feat5, feat6)
        pred = [layer(feat) for layer, feat in zip(self.layers, feats)]
        pred = torch.cat(pred, dim=1)
        return pred

    @staticmethod
    def R18(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDResNetMain(**ResNetBkbn.PARA_R18, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def R34(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDResNetMain(**ResNetBkbn.PARA_R34, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def R50(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDResNetMain(**ResNetBkbn.PARA_R50, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def R101(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDResNetMain(**ResNetBkbn.PARA_R101, num_cls=num_cls, act=act, norm=norm, img_size=img_size)

    @staticmethod
    def R152(num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(300, 300)):
        return SSDResNetMain(**ResNetBkbn.PARA_R152, num_cls=num_cls, act=act, norm=norm, img_size=img_size)


class SSDConstMain(nn.Module):
    def __init__(self, num_cls=20, img_size=(300, 300), batch_size=1):
        super(SSDConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = SSDConstLayer.layers(
            min_sizes=(30, 60, 111, 162, 213, 264),
            max_sizes=(60, 111, 162, 213, 264, 315),
            ratioss=(2, (2, 3), (2, 3), (2, 3), 2, 2),
            strides=(8, 16, 32, 64, 100, 300),
            feat_sizes=SSDResNetMain.calc_feat_sizes(img_size),
            num_cls=num_cls, batch_size=batch_size
        )

    def forward(self, imgs):
        pred = []
        for layer in self.layers:
            pred.append(layer(None))
        pred = torch.cat(pred, dim=1)
        return pred


class SSD(OneStageTorchModel):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(SSD, self).__init__(backbone=backbone, device=device, pack=pack)
        self._generate_anchors()

    def _generate_anchors(self):
        anchors = torch.cat([layer.anchors for layer in self.backbone.layers], dim=0)
        self.anchors = anchors.detach().cpu().numpy()
        self.num_ancr = self.anchors.shape[0]
        return None

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def img_size(self):
        return self.backbone.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size
        self._generate_anchors()

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True, nms_type=NMS_TYPE.HARD,
                    iou_type=IOU_TYPE.IOU, cind2name=None):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        preds = self.pkd_modules['backbone'](imgsT.to(self.device))
        preds[:, :, 4:] = torch.softmax(preds[:, :, 4:], dim=2)
        confssT, cindssT = torch.max(preds[:, :, 5:], dim=2)
        confssT[preds[:, :, 4] > confssT] = 0
        xyxyssT = xywhsT2xyxysT(preds[:, :, :4])

        labels = []
        for xyxysT, confsT, cindsT in zip(xyxyssT, confssT, cindssT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
            prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=cindsT if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
            boxs = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name, img_size=self.img_size,
                num_cls=self.num_cls)
            labels.append(boxs)
        return labels_rescale(labels, imgs, 1 / ratios)

    def labels2tars(self, labels, pos_thresh=0.5, ignore_thresh=0.225):
        cinds_tg = [np.zeros(shape=(0, self.num_cls))]
        xywhs_tg = [np.zeros(shape=(0, 4))]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_neg = [np.zeros(shape=0, dtype=np.int32)]
        inds_neg = [np.zeros(shape=0, dtype=np.int32)]
        for i, label in enumerate(labels):
            xyxys_lb, cinds_lb = label.export_xyxysN_cindsN()
            xywhs_lb = xyxysN2xywhsN(xyxys_lb)
            num_lb = len(label)
            if num_lb == 0:
                inds_b_neg.append(np.full(fill_value=i, shape=self.num_ancr))
                inds_neg.append(np.arange(self.num_ancr))
                continue
            iou_mat = ropr_mat_xyxysN(xyxys_lb, self.anchors, opr_type=OPR_TYPE.IOU)
            # 最大匹配
            ids_gt_max = np.arange(num_lb)
            ids_pbx_max = []
            for j in range(num_lb):
                id = np.argmax(iou_mat[j])
                ids_pbx_max.append(id)
                iou_mat[:, id] = -np.inf
            ids_pbx_max = np.array(ids_pbx_max)
            # 阈值匹配
            ids_gt_iou, ids_pbx_iou = np.nonzero(iou_mat > pos_thresh)
            ids_lb = np.concatenate([ids_gt_max, ids_gt_iou], axis=0)
            ids_pbx = np.concatenate([ids_pbx_max, ids_pbx_iou], axis=0)

            cinds_tg.append(cinds_lb[ids_lb])
            xywhs_tg.append(xywhs_lb[ids_lb])
            inds_b_pos.append(np.full(fill_value=i, shape=ids_lb.shape[0]))
            inds_pos.append(ids_pbx)
            inds_b_neg.append(np.full(fill_value=i, shape=self.num_ancr - ids_lb.shape[0]))
            inds_neg.append(np.delete(np.arange(self.num_ancr), ids_pbx))

        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        inds_b_neg = np.concatenate(inds_b_neg, axis=0)
        inds_neg = np.concatenate(inds_neg, axis=0)
        xywhs_tg = np.concatenate(xywhs_tg, axis=0)
        cinds_tg = np.concatenate(cinds_tg, axis=0)
        target = (inds_b_pos, inds_pos, xywhs_tg, cinds_tg, inds_b_neg, inds_neg)
        return target

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        inds_b_pos, inds_pos, xywhs_tg, cinds_tg, inds_b_neg, inds_neg = targets
        xywhs_tg = torch.as_tensor(xywhs_tg, dtype=torch.float).to(self.device, non_blocking=True)
        cinds_tg = torch.as_tensor(cinds_tg, dtype=torch.long).to(self.device, non_blocking=True)
        inds_b_neg = torch.as_tensor(inds_b_neg, dtype=torch.long).to(self.device, non_blocking=True)
        inds_neg = torch.as_tensor(inds_neg, dtype=torch.long).to(self.device, non_blocking=True)
        inds_b_pos = torch.as_tensor(inds_b_pos, dtype=torch.long).to(self.device, non_blocking=True)
        inds_pos = torch.as_tensor(inds_pos, dtype=torch.long).to(self.device, non_blocking=True)

        xywhs_pd = preds[inds_b_pos, inds_pos, :4]

        xy_loss = F.smooth_l1_loss(xywhs_pd[:, :2] / xywhs_tg[:, 2:] / 0.1,
                                   xywhs_tg[:, :2] / xywhs_tg[:, 2:] / 0.1, reduction='mean')

        wh_loss = F.smooth_l1_loss(torch.log(xywhs_pd[:, 2:].clamp(min=0.001)) / 0.2,
                                   torch.log(xywhs_tg[:, 2:].clamp(min=0.001)) / 0.2, reduction='mean')

        xywh_loss = (xy_loss + wh_loss)

        iconf_chots_pd_sft = torch.softmax(preds[:, :, 4:], dim=2)
        confs_neg_pd = iconf_chots_pd_sft[inds_b_neg, inds_neg]
        pos_num = max(len(inds_pos), 1)
        order = torch.argsort(confs_neg_pd, descending=False)
        order = order[:pos_num * 3]  # 控制正负样本数1比3
        inds_b_select = torch.cat([inds_b_pos, inds_b_neg[order]], dim=0)
        inds_select = torch.cat([inds_pos, inds_neg[order]], dim=0)
        cinds_tg_select = torch.cat(
            [cinds_tg + 1, torch.full(size=(order.size(0),), fill_value=0, device=preds.device)])

        cls_loss = F.cross_entropy(preds[inds_b_select, inds_select, 4:], cinds_tg_select, reduction='mean') * 3
        return OrderedDict(xywh=xywh_loss, cls=cls_loss)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = SSDConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=PACK.NONE)

    @staticmethod
    def VGGA(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDVGGMain.A(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def VGGD(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDVGGMain.D(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR18(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDResNetMain.R18(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR34(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDResNetMain.R34(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR50(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDResNetMain.R50(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR101(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDResNetMain.R101(num_cls=num_cls, act=ACT.RELU, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR152(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = SSDResNetMain.R152(num_cls=num_cls, act=ACT.RELU, img_size=img_size)
        return SSD(backbone=backbone, device=device, pack=pack)


if __name__ == '__main__':
    pass
