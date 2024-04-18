from models.template import OneStageTorchModel, IndependentInferableModel
from utils.frame import *


class YoloFrame(OneStageTorchModel, IndependentInferableModel):

    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(YoloFrame, self).__init__(backbone=backbone, device=device, pack=pack)

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def img_size(self):
        return self.backbone.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size

    @property
    def num_anchor(self):
        return np.sum([layer.num_anchor for layer in self.backbone.layers])

    @property
    def anchors(self):
        return torch.cat([layer.anchors for layer in self.backbone.layers], dim=0)

    @torch.no_grad()
    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True, num_presv=3000,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None, **kwargs):
        self.eval()
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size, device=self.device)
        preds = self.pkd_modules['backbone'](imgsT)
        xywhssT, confssT, chotssT = preds.split((4, 1, self.num_cls), dim=-1)
        max_vals, cindssT = torch.max(torch.sigmoid(chotssT), dim=-1)
        confssT = confssT[..., 0] * max_vals
        xyxyssT = xyxysT_clip(xywhsT2xyxysT(xywhssT), xyxyN_rgn=np.array(self.img_size))
        labels = []
        for xyxysT, confsT, cindsT in zip(xyxyssT, confssT, cindssT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_msks], confsT[prsv_msks], cindsT[prsv_msks]
            prsv_inds = nms_xyxysT(xyxysT=xyxysT, confsT=confsT, cindsT=cindsT if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type, num_presv=num_presv)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=self.img_size))
                continue
            xyxysT, confsT, cindsT = xyxysT[prsv_inds], confsT[prsv_inds], cindsT[prsv_inds]
            boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, cind2name=cind2name, img_size=self.img_size,
                num_cls=self.num_cls)
            labels.append(boxes)
        return labels_rescale(labels, imgs2img_sizes(imgs), 1 / ratios)


class YoloInferN(IndependentInferableModel):

    def __init__(self, backbone, img_size, num_cls):
        super().__init__()
        self.backbone = backbone
        self._img_size = img_size
        self._num_cls = num_cls

    @property
    def num_cls(self):
        return self._num_cls

    @property
    def img_size(self):
        return self._img_size

    def imgs2labels(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True,
                    nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cind2name=None):
        img_size = (imgs.shape[3], imgs.shape[2])
        preds = self.backbone(imgs)
        max_vals = np.max(preds[..., 5:], axis=-1)
        cindssN = np.argmax(preds[..., 5:], axis=-1)
        confssN = preds[..., 4] * max_vals
        xyxyssN = xywhsN2xyxysN(preds[..., :4])
        labels = []
        for xyxysN, confsN, cindsN in zip(xyxyssN, confssN, cindssN):
            prsv_msks = confsN > conf_thres
            if not np.any(prsv_msks):
                labels.append(BoxesLabel(img_size=img_size))
                continue
            xyxysN, confsN, cindsN = xyxysN[prsv_msks], confsN[prsv_msks], cindsN[prsv_msks]
            prsv_inds = nms_xyxysN(xyxysN=xyxysN, confsN=confsN, cindsN=cindsN if by_cls else None,
                                   iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            if len(prsv_inds) == 0:
                labels.append(BoxesLabel(img_size=img_size))
                continue
            xyxysN, confsN, cindsN = xyxysN[prsv_inds], confsN[prsv_inds], cindsN[prsv_inds]
            boxs = BoxesLabel.from_xyxysN_confsN_cindsN(xyxysN=xyxysN, cindsN=cindsN, confsN=confsN,
                                                        cind2name=cind2name,
                                                        img_size=img_size, num_cls=self.num_cls)
            labels.append(boxs)
        return labels
