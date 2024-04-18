from scipy.optimize import linear_sum_assignment

from data.processing import _genrgns_pyramid
from models.modules import *
from models.template import *


class ItemMixorMain(SequenceONNXExportable):
    @property
    def length(self):
        return self._num_input

    @property
    def num_input(self):
        return self._num_input

    @property
    def num_query(self):
        return self._num_query

    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features

    def __init__(self, in_features, out_features, query_channels=128, num_input=200,
                 num_query=100, numl_encoder=2, numl_decoder=2,
                 features_fedfwd=1024, ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._num_input = num_input
        self._num_query = num_query
        self.cvtor_in = nn.Linear(in_features=self.in_features, out_features=query_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=query_channels, nhead=8,
                                                   dim_feedforward=features_fedfwd, dropout=0.1,
                                                   batch_first=True, )
        encoder_norm = nn.LayerNorm(query_channels, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=numl_encoder, norm=encoder_norm)

        # self.trans = nn.Transformer(d_model=query_channels, nhead=8, num_encoder_layers=numl_encoder,
        #                             num_decoder_layers=numl_decoder, dim_feedforward=features_fedfwd,
        #                             batch_first=True, dropout=0.1)
        self.query_embed = nn.Embedding(num_query, query_channels)
        self.cvtor_out = nn.Linear(in_features=query_channels, out_features=self.out_features)

    @property
    def num_cls(self):
        return self._num_cls

    def forward(self, x):
        feat = self.cvtor_in(x)
        # tgt = self.query_embed.weight[None].repeat(x.size(0), 1, 1)
        # feat = self.trans(feat, tgt=tgt)
        feat = self.encoder(feat)
        y = self.cvtor_out(feat)
        return y

    PARA_SMALL = dict(numl_encoder=2, numl_decoder=2, features_fedfwd=1024, query_channels=128, )
    PARA_MEDIUM = dict(numl_encoder=4, numl_decoder=4, features_fedfwd=1024, query_channels=128, )
    PARA_LARGE = dict(numl_encoder=6, numl_decoder=6, features_fedfwd=2048, query_channels=256, )

    @staticmethod
    def Small(in_features, out_features, num_input=200, num_query=100):
        return ItemMixorMain(**ItemMixorMain.PARA_SMALL, in_features=in_features, num_query=num_query,
                             out_features=out_features, num_input=num_input)

    @staticmethod
    def Medium(in_features, out_features, num_input=200, num_query=100):
        return ItemMixorMain(**ItemMixorMain.PARA_MEDIUM, in_features=in_features, num_query=num_query,
                             out_features=out_features, num_input=num_input)

    @staticmethod
    def Large(in_features, out_features, num_input=200, num_query=100):
        return ItemMixorMain(**ItemMixorMain.PARA_LARGE, in_features=in_features, num_query=num_query,
                             out_features=out_features, num_input=num_input)


if __name__ == '__main__':
    model = ItemMixorMain.Small(in_features=10, out_features=20)
    model.export_onnx('./buff')


class ItemMixorConstMain(nn.Module):
    def __init__(self, out_features, batch_size=1, num_query=100, num_input=200):
        super().__init__()
        self.num_input = num_input
        self.num_query = num_query
        self.feat = nn.Parameter(torch.rand(size=(batch_size, num_query, out_features)))

    def forward(self, x):
        return self.feat


def xyxysN_chotsN_cutpry_random(img_rgn, xyxys, chots, piece_sizes=((512, 512), (256, 256)),
                                over_laps=((128, 128), (64, 64)), xyxy_jitter=0, chot_jitter=0.0):
    xyxys_rgn = _genrgns_pyramid(img_rgn, piece_sizesN=np.array(piece_sizes),
                                 over_lapsN=np.array(over_laps), with_clip=True).astype(np.int32)
    xyxys_int = np.concatenate([np.maximum(xyxys[:, None, :2], xyxys_rgn[:, :2]),
                                np.minimum(xyxys[:, None, 2:4], xyxys_rgn[:, 2:4])], axis=2)
    xyxys_int = xyxys_int + np.random.randint(low=-xyxy_jitter, high=xyxy_jitter + 1, size=xyxys_int.shape)
    fltr_valid = np.all(xyxys_int[..., 2:4] - xyxys_int[..., :2] > 0, axis=2)
    chots_ext = chots[:, None, :] * (1 - chot_jitter) + \
                chot_jitter * np.random.rand(chots.shape[0], xyxys_rgn.shape[0], chots.shape[1])
    ids_lb, ids_ptch = np.nonzero(fltr_valid)
    return xyxys_int[ids_lb, ids_ptch], chots_ext[ids_lb, ids_ptch], xyxys_rgn[ids_ptch]


def label_cutpry_random(label, num_cls, piece_sizes=((512, 512), (256, 256)),
                        over_laps=((128, 128), (64, 64)), xyxy_jitter=0, chot_jitter=0.0):
    xyxys = label.export_xyxysN()
    chots = label.export_chotsN(num_cls=num_cls)
    img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
    xyxys_pie, chots_pie, xyxys_ptch_pie = xyxysN_chotsN_cutpry_random(
        img_rgn, xyxys, chots, piece_sizes=piece_sizes,
        over_laps=over_laps, xyxy_jitter=xyxy_jitter, chot_jitter=chot_jitter)
    label_cutd = label.empty()
    for xyxy_pie, chot_pie, xyxy_ptch_pie in zip(xyxys_pie, chots_pie, xyxys_ptch_pie):
        category = OneHotCategory(chotN=chot_pie)
        border = XYXYBorder(xyxy_pie, size=label.img_size)
        item = BoxItem(category=category, border=border, xyxy_ptch=xyxy_ptch_pie)
        label_cutd.append(item)
    return label_cutd


class ItemMixor(OneStageTorchModel):
    def __init__(self, backbone, num_cls=10, img_size=(1024, 1024), device=None, pack=PACK.AUTO):
        super(OneStageTorchModel, self).__init__(backbone=backbone, device=device, pack=pack)
        self.piece_sizes = ((512, 512), (256, 256))
        self.over_laps = ((128, 128), (64, 64))
        self.cost_weight_iou = 2
        self.cost_weight_l1 = 5
        self.cost_weight_cls = 1
        self.img_size = img_size
        self._num_cls = num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    @property
    def num_input(self):
        return self.backbone.num_input

    @property
    def num_query(self):
        return self.backbone.num_query

    @property
    def num_cls(self):
        return self._num_cls

    def _pad_to(self, arr, length, pad_val=0):
        len_cur, features = arr.shape[-2:]
        if len_cur >= length:
            return arr[..., :length, :]
        else:
            pad_width = ((0, 0), (0, length - len_cur), (0, 0))
            return np.pad(arr, pad_width=pad_width, mode='constant', constant_values=pad_val)

    def _align_input(self, xyxys_loc, xyxys_ptch, chots_loc):
        max_input = max([v.shape[-2] for v in xyxys_loc]) if len(xyxys_loc) > 0 else 0
        max_input = min(max_input, self.num_input)
        xyxys_loc.append(np.zeros(shape=(0, max_input, 4)))
        xyxys_ptch.append(np.zeros(shape=(0, max_input, 4)))
        chots_loc.append(np.zeros(shape=(0, max_input, self.num_cls)))
        xyxys_loc = np.concatenate([self._pad_to(arr, max_input) for arr in xyxys_loc], axis=0)
        xyxys_ptch = np.concatenate([self._pad_to(arr, max_input) for arr in xyxys_ptch], axis=0)
        chots_loc = np.concatenate([self._pad_to(arr, max_input) for arr in chots_loc], axis=0)
        return xyxys_loc, xyxys_ptch, chots_loc

    def labels2tars(self, labels, **kwargs):
        xyxys_loc = []
        xyxys_ptch = []
        chots_loc = []
        xyxys_gol = []
        chots_gol = []
        for i, label in enumerate(labels):
            xyxys = label.export_xyxysN()
            chots = label.export_chotsN(num_cls=self.num_cls)
            img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
            xyxys_pie, chots_pie, xyxys_ptch_pie = xyxysN_chotsN_cutpry_random(
                img_rgn, xyxys, chots, piece_sizes=((512, 512), (256, 256)),
                over_laps=((128, 128), (64, 64)), xyxy_jitter=16, chot_jitter=0.5)
            xyxys_loc.append(xyxys_pie[None])
            chots_loc.append(chots_pie[None])
            xyxys_ptch.append(xyxys_ptch_pie[None])
            xyxys_gol.append(xyxys)
            chots_gol.append(chots)
        xyxys_loc, xyxys_ptch, chots_loc = self._align_input(xyxys_loc, xyxys_ptch, chots_loc)
        return xyxys_loc, xyxys_ptch, chots_loc, xyxys_gol, chots_gol

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        self.train()
        xyxyss_loc, xyxyss_ptch, chotss_loc, xyxyss_gol, chotss_gol = arrsN2arrsT(targets, device=self.device)
        scaler = np.sqrt(np.prod(self.img_size))
        feats_in = torch.cat([xyxyss_loc / scaler, xyxyss_ptch / scaler, chotss_loc], dim=-1)
        feats_out = self.pkd_modules['backbone'](feats_in)
        xyxyss_pd, chotss_pd = feats_out.split((4, self.num_cls + 1), dim=2)
        chotss_pd_sft = torch.softmax(chotss_pd, dim=2)[..., :self.num_cls]
        xyxyss_pd = xyxyss_pd * scaler

        iou_loss = torch.as_tensor(0).to(self.device, non_blocking=True)
        cls_loss = torch.as_tensor(0).to(self.device, non_blocking=True)
        l1_loss = torch.as_tensor(0).to(self.device, non_blocking=True)
        num_mtch = 0
        for i, (xyxys_pd, chots_pd, chots_pd_sft, xyxys_gol, chots_gol) in enumerate(
                zip(xyxyss_pd, chotss_pd, chotss_pd_sft, xyxyss_gol, chotss_gol)):
            cinds_tg_align = torch.full(
                size=(chots_pd.size(0),), fill_value=self.num_cls, device=self.device, dtype=torch.long)
            cinds_tg = torch.argmax(chots_gol, dim=1)
            if cinds_tg.size(0) > 0:
                cost_cls = -chots_pd_sft[:, cinds_tg]
                cost_iou = 1 - ropr_mat_xyxysT(xyxys_pd, xyxys_gol, opr_type=OPR_TYPE.IOU)
                cost_l1 = torch.cdist(xyxys_pd, xyxys_gol, p=1) / scaler
                cost = cost_cls * self.cost_weight_cls + cost_iou * self.cost_weight_iou + cost_l1 * self.cost_weight_l1
                # print(cost)
                inds_pd, inds_tg = linear_sum_assignment(cost.detach().cpu().numpy())

                iou_loss = iou_loss + torch.sum(cost_iou[inds_pd, inds_tg])
                l1_loss = l1_loss + torch.sum(cost_l1[inds_pd, inds_tg])
                cinds_tg_align[inds_pd] = cinds_tg[inds_tg]
                num_mtch = num_mtch + inds_pd.shape[0]
            # 分类损失
            cls_loss = cls_loss + F.cross_entropy(chots_pd, cinds_tg_align, reduction='mean')

        num_mtch = max(num_mtch, 1)
        iou_loss = iou_loss / num_mtch
        l1_loss = l1_loss / num_mtch
        cls_loss = cls_loss / xyxyss_loc.size(0)
        return OrderedDict(cls=cls_loss, l1=l1_loss * 5, iou=iou_loss * 2)

    def labels_mix(self, labels, conf_thres=0.5, cind2name=None):
        self.eval()
        scaler = np.sqrt(np.prod(self.img_size))
        xyxys_loc = []
        xyxys_ptch = []
        chots_loc = []
        for label in labels:
            xyxys_loc.append(label.export_xyxysN()[None])
            chots_loc.append(label.export_chotsN(num_cls=self.num_cls)[None])
            xyxys_ptch.append(label.export_valsN(key='xyxy_ptch', default=np.zeros(4))[None])
        xyxys_loc, xyxys_ptch, chots_loc = \
            arrsN2arrsT(self._align_input(xyxys_loc, xyxys_ptch, chots_loc), device=self.device)
        feats_in = torch.cat([xyxys_loc / scaler, xyxys_ptch / scaler, chots_loc], dim=-1)
        feats_out = self.pkd_modules['backbone'](feats_in)
        xyxyss_pd, chotss_pd = feats_out.split((4, self.num_cls + 1), dim=2)
        chotss_pd = torch.softmax(chotss_pd, dim=2)[..., :self.num_cls]
        xyxyss_pd = xyxyss_pd * scaler
        labels_mixd = []
        for label, chots_pd, xyxys_pd in zip(labels, chotss_pd, xyxyss_pd):
            confs = torch.max(chots_pd, dim=1)[0]
            mask_presv = confs > conf_thres
            chots_pd, xyxys_pd = chots_pd[mask_presv], xyxys_pd[mask_presv]
            boxes = BoxesLabel.from_xyxysT_chotsT(xyxys_pd, chots_pd, img_size=label.img_size,
                                                  num_cls=self.num_cls, cind2name=cind2name)
            labels_mixd.append(boxes)
        return labels_mixd

    @staticmethod
    def Small(device=None, pack=PACK.AUTO, num_cls=10, img_size=(224, 224), num_input=200, num_query=100):
        backbone = ItemMixorMain.Small(in_features=num_cls + 8, num_query=num_query, num_input=num_input,
                                       out_features=num_cls + 5)
        return ItemMixor(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def Large(device=None, pack=PACK.AUTO, num_cls=10, img_size=(224, 224), num_input=200, num_query=100):
        backbone = ItemMixorMain.Large(in_features=num_cls + 8, num_query=num_query, num_input=num_input,
                                       out_features=num_cls + 5)
        return ItemMixor(backbone=backbone, device=device, pack=pack, num_cls=num_cls, img_size=img_size)

    @staticmethod
    def Const(device=None, batch_size=1, num_cls=10, img_size=(224, 224), num_input=200, num_query=100):
        backbone = ItemMixorConstMain(batch_size=batch_size, out_features=num_cls + 5,
                                      num_input=num_input, num_query=num_query)
        return ItemMixor(backbone=backbone, device=device, pack=PACK.NONE, num_cls=num_cls, img_size=img_size)

# if __name__ == '__main__':
#     model = PyramidMixorMain(num_cls=20, )
#     model.export_onnx('./buff')
