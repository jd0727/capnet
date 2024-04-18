from models.base.hourglass import EXKP
from models.base.resnet import ResNetBkbn
from models.modules import *
from models.template import OneStageTorchModel
from utils.frame import *


class CenterLayer(AnchorImgLayer):
    def __init__(self, in_channels, inner_channels, stride=4, num_cls=80, img_size=(0, 0), act=ACT.RELU):
        super(CenterLayer, self).__init__(anchor_sizes=(img_size), stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.cls = nn.Sequential(
            Ck3A(in_channels=in_channels, out_channels=inner_channels, act=act),
            Ck1(in_channels=inner_channels, out_channels=num_cls)
        )
        self.cls[1].conv.bias.data.fill_(-2.19)
        self.wh = nn.Sequential(
            Ck3A(in_channels=in_channels, out_channels=inner_channels, act=act),
            Ck1(in_channels=inner_channels, out_channels=2)
        )
        self.xy = nn.Sequential(
            Ck3A(in_channels=in_channels, out_channels=inner_channels, act=act),
            Ck1(in_channels=inner_channels, out_channels=2)
        )

    @staticmethod
    def decode(xy, wh, chot, xy_offset, wh_offset, stride):
        xy_offset = xy_offset.to(xy.device, non_blocking=True)
        xy = xy.permute(0, 2, 3, 1)  # (Nb,H,W,c)
        xy = (torch.sigmoid(xy) + xy_offset) * stride

        wh_offset = wh_offset.to(wh.device, non_blocking=True)
        wh = wh.permute(0, 2, 3, 1)
        wh = torch.sigmoid(wh) * wh_offset

        chot = chot.permute(0, 2, 3, 1)
        chot = torch.sigmoid(chot)
        pred = torch.cat([xy, wh, chot], dim=-1).contiguous()
        return pred

    def forward(self, featmap):
        xy = self.xy(featmap)
        wh = self.wh(featmap)
        chot = self.cls(featmap)
        pred = CenterLayer.decode(
            xy=xy, wh=wh, chot=chot,
            xy_offset=self.xy_offset, wh_offset=self.wh_offset, stride=self.stride)
        return pred


class CenterConstLayer(AnchorImgLayer):
    def __init__(self, batch_size=1, stride=4, num_cls=80, img_size=(512, 512)):
        super(CenterConstLayer, self).__init__(anchor_sizes=(img_size), stride=stride, img_size=img_size)
        self.featmap = nn.Parameter(torch.zeros(batch_size, num_cls + 4, self.Hf, self.Wf))
        nn.init.constant_(self.featmap[:, 4:, :, :], -2.69)

    def forward(self, featmap):
        xy = self.featmap[:, :2, :, :]
        wh = self.featmap[:, 2:4, :, :]
        chot = self.featmap[:, 4:, :, :]
        pred = CenterLayer.decode(
            xy=xy, wh=wh, chot=chot,
            xy_offset=self.xy_offset, wh_offset=self.wh_offset, stride=self.stride)
        return pred


class CenterConstMain(nn.Module):
    def __init__(self, num_cls=80, img_size=(416, 416), batch_size=5):
        super(CenterConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layer = CenterConstLayer(batch_size=batch_size, stride=4, num_cls=num_cls, img_size=img_size)

    def forward(self, imgs):
        return self.layer(None)


class CenterHourgMain(EXKP, ImageONNXExportable):

    def __init__(self, nstack, channelss, repeat_nums, cps_channels=256, num_cls=80, img_size=(416, 352),
                 act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        super(CenterHourgMain, self).__init__(nstack=nstack, channelss=channelss, repeat_nums=repeat_nums,
                                              cps_channels=cps_channels, act=act, norm=norm, in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self.layer = CenterLayer(in_channels=cps_channels, inner_channels=channelss[0], num_cls=num_cls, act=act,
                                 img_size=img_size)

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        self.layer.img_size = img_size

    def forward(self, imgs):
        featmap = super(CenterHourgMain, self).forward(imgs)
        featmap = self.layer(featmap)
        return featmap

    @staticmethod
    def Small(num_cls=80, act=ACT.RELU, norm=NORM.BATCH, img_size=(416, 352), in_channels=3):
        return CenterHourgMain(**EXKP.SMALL_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                               in_channels=in_channels)

    @staticmethod
    def Large(num_cls=80, act=ACT.RELU, norm=NORM.BATCH, img_size=(416, 352), in_channels=3):
        return CenterHourgMain(**EXKP.LARGE_PARA, num_cls=num_cls, act=act, norm=norm, img_size=img_size,
                               in_channels=in_channels)


class CenterResNetMain(ResNetBkbn):
    def __init__(self, Module, repeat_nums, channels, num_cls=20, act=ACT.RELU, norm=NORM.BATCH, img_size=(416, 352),
                 in_channels=3):
        super(CenterResNetMain, self).__init__(Module, repeat_nums, channels, act=act, norm=norm,
                                               in_channels=in_channels)
        self.num_cls = num_cls
        self._img_size = img_size
        self.deconvs = nn.Sequential(
            CTpaNA(in_channels=channels * 8, out_channels=256, kernel_size=4, stride=2, norm=norm, act=act),
            CTpaNA(in_channels=256, out_channels=256, kernel_size=4, stride=2, norm=norm, act=act),
            CTpaNA(in_channels=256, out_channels=256, kernel_size=4, stride=2, norm=norm, act=act)
        )
        self.layer = CenterLayer(in_channels=256, inner_channels=64, num_cls=num_cls, act=act,
                                 stride=4, img_size=img_size)

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        self.layer.img_size = img_size

    def forward(self, imgs):
        feat = super(CenterResNetMain, self).forward(imgs)
        feat = self.deconvs(feat)
        pred = self.layer(feat)
        return pred

    @staticmethod
    def R18(num_cls=80, img_size=(256, 256), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return CenterResNetMain(**ResNetBkbn.PARA_R18, img_size=img_size, num_cls=num_cls, act=act, norm=norm,
                                in_channels=in_channels)

    @staticmethod
    def R34(num_cls=80, img_size=(256, 256), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return CenterResNetMain(**ResNetBkbn.PARA_R34, img_size=img_size, num_cls=num_cls, act=act, norm=norm,
                                in_channels=in_channels)

    @staticmethod
    def R50(num_cls=80, img_size=(256, 256), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return CenterResNetMain(**ResNetBkbn.PARA_R50, img_size=img_size, num_cls=num_cls, act=act, norm=norm,
                                in_channels=in_channels)

    @staticmethod
    def R101(num_cls=80, img_size=(256, 256), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return CenterResNetMain(**ResNetBkbn.PARA_R101, img_size=img_size, num_cls=num_cls, act=act, norm=norm,
                                in_channels=in_channels)

    @staticmethod
    def R152(num_cls=80, img_size=(256, 256), act=ACT.RELU, norm=NORM.BATCH, in_channels=3):
        return CenterResNetMain(**ResNetBkbn.PARA_R152, img_size=img_size, num_cls=num_cls, act=act, norm=norm,
                                in_channels=in_channels)


class Center(OneStageTorchModel):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(Center, self).__init__(backbone=backbone, device=device, pack=pack)
        self.layer = self.backbone.layer

    @property
    def num_cls(self):
        return self.backbone.num_cls

    @property
    def img_size(self):
        return self.backbone.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size

    def imgs2labels(self, imgs, top_num=100, conf_thres=0.1, cind2name=None, **kwargs):
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        preds = self.pkd_modules['backbone'](imgsT.to(self.device))
        assert isinstance(preds, torch.Tensor), 'err type'
        labels = []
        for i in range(preds.size(0)):
            xywhsT, chotsT = preds[i, ..., :4], preds[i, ..., 4:]
            inds = torch.nonzero(chotsT == F.max_pool2d(chotsT, kernel_size=3, padding=1, stride=1))
            inds_y, inds_x, cindsT = inds[:, 0], inds[:, 1], inds[:, 2]
            confsT = chotsT[inds_y, inds_x, cindsT]
            order = torch.argsort(confsT, descending=True)[:top_num]
            # 置信度截断
            order = order[confsT[order] > conf_thres]
            # 选取输出
            inds_y, inds_x, cindsT, confsT = inds_y[order], inds_x[order], cindsT[order], confsT[order]
            xywhsT = xywhsT[inds_y, inds_x, :]
            # 转化box
            xyxysT = xywhsT2xyxysT(xywhsT)
            boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, img_size=self.img_size,
                num_cls=self.num_cls, cind2name=cind2name)
            labels.append(boxes)
        return labels_rescale(labels, imgs, 1 / ratios)

    # 计算置信度半径
    @staticmethod
    def calc_radius(w, h, iou_thres=0.7):
        a1 = 1
        b1 = w + h
        c1 = w * h * (1 - iou_thres) / (1 + iou_thres)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / (2 * a1)
        a2 = 4
        b2 = 2 * (w + h)
        c2 = (1 - iou_thres) * w * h
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / (2 * a2)
        a3 = 4 * iou_thres
        b3 = -2 * iou_thres * (w + h)
        c3 = (iou_thres - 1) * w * h
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return int(np.ceil(min(r1, r2, r3)))

    # 画高斯分布圆形
    @staticmethod
    def put_gauss(heat_map, ix, iy, w, h, iou_thres=0.7):
        ix, iy, w, h = int(ix), int(iy), int(w), int(h)
        radius = Center.calc_radius(w, h, iou_thres=iou_thres)
        diameter = 2 * radius + 1
        sigma = diameter / 6
        xs, ys = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
        mask = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
        mask[mask < np.finfo(mask.dtype).eps * mask.max()] = 0
        # 填充
        left, right = min(ix, radius), min(heat_map.shape[1] - ix, radius + 1)
        top, bottom = min(iy, radius), min(heat_map.shape[0] - iy, radius + 1)
        heat_map[iy - top:iy + bottom, ix - left:ix + right] += \
            mask[radius - top:radius + bottom, radius - left:radius + right]
        return heat_map

    def labels2tars(self, labels, iou_thres=0.7):
        Wf, Hf, stride = self.layer.Wf, self.layer.Hf, self.layer.stride
        inds_xy = [np.zeros(shape=(0, 2))]
        inds_c = [np.zeros(shape=0)]
        xywhs_tg = [np.zeros(shape=(0, 4))]
        hmaps_tg = [np.zeros(shape=(0, Hf, Wf))]
        inds_b_c = [np.zeros(shape=0)]
        inds_b_xy = [np.zeros(shape=0)]
        for i, label in enumerate(labels):
            xywhs, cinds = label.export_xywhsN_cindsN(label)
            if len(label) == 0:
                continue
            ids_xy = xywhs[:, :2] // stride
            inds_xy.append(ids_xy)
            xywhs_tg.append(xywhs)
            hmap_dicts = {}
            for n in range(len(cinds)):
                if cinds[n] not in hmap_dicts.keys():
                    hmap_dicts[cinds[n]] = np.zeros(shape=(Hf, Wf))
                hmap = hmap_dicts[cinds[n]]
                Center.put_gauss(hmap, ix=ids_xy[n, 0], iy=ids_xy[n, 1],
                                 w=xywhs[n, 2], h=xywhs[n, 3], iou_thres=iou_thres)
            inds_c.append(np.array(list(hmap_dicts.keys())))
            hmaps_tg.append(np.stack(list(hmap_dicts.values()), axis=0))
            inds_b_c.append(np.full(fill_value=i, shape=len(hmap_dicts)))
            inds_b_xy.append(np.full(fill_value=i, shape=len(ids_xy)))

        inds_xy = np.concatenate(inds_xy, axis=0)
        inds_b_xy = np.concatenate(inds_b_xy, axis=0)
        xywhs_tg = np.concatenate(xywhs_tg, axis=0)
        inds_c = np.concatenate(inds_c, axis=0)
        inds_b_c = np.concatenate(inds_b_c, axis=0)
        hmaps_tg = np.concatenate(hmaps_tg, axis=0)
        return inds_b_xy, inds_xy, xywhs_tg, inds_b_c, inds_c, hmaps_tg

    def imgs_tars2loss(self, imgs, targets, alpha=2, beta=2):
        imgs = imgs.to(self.device)
        preds = self.pkd_modules['backbone'](imgs)
        inds_b_xy, inds_xy, xywhs_tg, inds_b_c, inds_c, hmaps_tg = targets
        xywhs_tg = torch.as_tensor(xywhs_tg, dtype=torch.float).to(preds.device, non_blocking=True)
        hmaps_tg = torch.as_tensor(hmaps_tg, dtype=torch.float).to(preds.device, non_blocking=True)
        xywhs_pd, hmaps_pd = preds[..., :4], preds[..., 4:]
        # 目标检出损失
        hmaps_pd = hmaps_pd[inds_b_c, :, :, inds_c]
        pos_mask = hmaps_tg == 1
        pos_loss = -torch.mean(
            torch.pow(1 - hmaps_pd[pos_mask], alpha) *
            torch.log(hmaps_pd[pos_mask] + 1e-16)
        ) * 10
        neg_mask = hmaps_tg < 1
        neg_loss = -torch.mean(
            torch.pow(hmaps_pd[neg_mask], beta) *
            torch.log(1 - hmaps_pd[neg_mask] + 1e-16)
        ) * 10
        # 边框损失
        xywhs_pd = xywhs_pd[inds_b_xy, inds_xy[:, 1], inds_xy[:, 0], :]
        wh_loss = F.smooth_l1_loss(xywhs_pd[..., 2:], xywhs_tg[..., 2:], reduction='mean') * 0.1
        xy_loss = F.smooth_l1_loss(xywhs_pd[..., :2], xywhs_tg[..., :2], reduction='mean')
        return OrderedDict(pos=pos_loss, neg=neg_loss, xy=xy_loss, wh=wh_loss)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = CenterConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size)
        return Center(backbone=backbone, device=device, pack=PACK.NONE)

    @staticmethod
    def HourgSmall(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterHourgMain.Small(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                         in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def HourgLarge(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterHourgMain.Small(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                         in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR18(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterResNetMain.R18(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                        in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR34(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterResNetMain.R34(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                        in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR50(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterResNetMain.R50(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                        in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR101(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterResNetMain.R101(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                         in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def ResNetR152(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO, in_channels=3):
        backbone = CenterResNetMain.R152(num_cls=num_cls, act=ACT.RELU, norm=NORM.BATCH, img_size=img_size,
                                         in_channels=in_channels)
        return Center(backbone=backbone, device=device, pack=pack)


if __name__ == '__main__':
    model = Center.ResNetR18(device='cpu', img_size=(960, 960))
    x = torch.zeros(1, 3, 960, 960)
    y = model.backbone(x)

    # norm = (416, 416)
    # batch_size = 3
