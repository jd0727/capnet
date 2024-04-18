from typing import Optional, Any

from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

from models.base import ResNetBkbn
from models.base.modules import MLP
from models.modules import *
from models.template import *
from utils import ImageONNXExportable


#  <editor-fold desc='获取空间编码'>
def _pos_embd_from_mask(mask: torch.Tensor, features: int, temperature: int = 10000):
    N, H, W = mask.size()
    y_accu = mask.cumsum(dim=1, dtype=torch.float32)
    x_accu = mask.cumsum(dim=2, dtype=torch.float32)

    dim_t = torch.arange(features // 2, dtype=torch.float32, device=mask.device) / (features // 2)
    dim_t = torch.pow(temperature, dim_t)
    freq_x = x_accu[:, :, :, None] / dim_t
    freq_y = y_accu[:, :, :, None] / dim_t
    pos_embd = torch.cat((freq_x.sin(), freq_x.cos(), freq_y.sin(), freq_y.cos()), dim=-1).permute(0, 3, 1, 2)
    return pos_embd


def _pos_embd_from_size(size: tuple, features: int, temperature: int = 10000):
    W, H = size
    y_accu, x_accu = arange2dT(H, W)
    dim_t = torch.arange(features // 2, dtype=torch.float32) / (features // 2)
    dim_t = torch.pow(temperature, dim_t)
    freq_x = x_accu[:, :, None] / dim_t
    freq_y = y_accu[:, :, None] / dim_t
    pos_embd = torch.cat((freq_x.sin(), freq_x.cos(), freq_y.sin(), freq_y.cos()), dim=-1).permute(2, 0, 1)
    return pos_embd


# if __name__ == '__main__':
#     embd=_pos_embd_from_size((20,20),features=256)


def _with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


# </editor-fold>


#  <editor-fold desc='空间编码的Transformer'>


class TransformerDecoderLayerWithPE(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayerWithPE, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos_embd: Optional[Tensor] = None,
                src_pos_embd: Optional[Tensor] = None):
        q = k = _with_pos_embed(tgt, tgt_pos_embd)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=_with_pos_embed(tgt, tgt_pos_embd),
                                   key=_with_pos_embed(memory, src_pos_embd),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoderLayerWithPE(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayerWithPE, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerWithPE, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, src_pos_embd: Optional[Tensor] = None) -> Tensor:
        q = k = _with_pos_embed(src, src_pos_embd)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderWithPE(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderWithPE, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, src_pos_embd: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, src_pos_embd=src_pos_embd)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderWithPE(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderWithPE, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos_embd: Optional[Tensor] = None,
                src_pos_embd: Optional[Tensor] = None
                ) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_pos_embd=tgt_pos_embd,
                         src_pos_embd=src_pos_embd)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerWithPE(nn.Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerWithPE, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayerWithPE(d_model, nhead, dim_feedforward, dropout,
                                                          activation, layer_norm_eps, batch_first,
                                                          **factory_kwargs)
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoderWithPE(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayerWithPE(d_model, nhead, dim_feedforward, dropout,
                                                          activation, layer_norm_eps, batch_first,
                                                          **factory_kwargs)
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoderWithPE(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                src_pos_embd: Optional[Tensor] = None,
                tgt_pos_embd: Optional[Tensor] = None) -> Tensor:

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, src_pos_embd=src_pos_embd)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              src_pos_embd=src_pos_embd, tgt_pos_embd=tgt_pos_embd)
        return output


# </editor-fold>


class SizeScaleLayer(nn.Module):

    def __init__(self, img_size):
        super(SizeScaleLayer, self).__init__()
        self.img_size = img_size

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        self.scaler = torch.as_tensor(list(img_size) * 2)

    @staticmethod
    def xywh_decode(xywh, scaler):
        return torch.sigmoid(xywh) * scaler.to(xywh.device)


class DETRLayer(SizeScaleLayer):
    def __init__(self, in_features, num_cls=20, img_size=(224, 224)):
        super(DETRLayer, self).__init__(img_size)
        self.chot = MLP(in_features=in_features, out_features=num_cls, inner_featuress=in_features)
        self.xywh = MLP(in_features=in_features, out_features=4, inner_featuress=in_features)

    def forward(self, features):
        chot = self.chot(features)
        xywh = self.xywh(features)
        xywh = SizeScaleLayer.xywh_decode(xywh, self.scaler)
        chot_xyxy = torch.cat([xywh, chot], dim=2)
        return chot_xyxy


class DETRConstLayer(SizeScaleLayer):

    def __init__(self, batch_size=1, num_queries=100, num_cls=20, img_size=(224, 224)):
        super(DETRConstLayer, self).__init__(img_size)
        self.chot = nn.Parameter(torch.zeros(batch_size, num_queries, num_cls))
        self.xywh = nn.Parameter(torch.zeros(batch_size, num_queries, 4))

    def forward(self, features):
        xywh = SizeScaleLayer.xywh_decode(self.xywh.to(features.device), self.scaler)
        chot = self.chot.to(features.device)
        return torch.cat([xywh, chot], dim=2)


class DETRConstMain(nn.Module):
    def __init__(self, batch_size=1, num_query=100, num_cls=20, img_size=(224, 224)):
        super().__init__()
        self.layer = DETRConstLayer(batch_size, num_query, num_cls=num_cls, img_size=img_size)
        self.num_cls = num_cls
        self.img_size = img_size

    def forward(self, imgs):
        return self.layer(imgs)


class DETRResNetMain(ResNetBkbn, ImageONNXExportable):

    def __init__(self, Module, repeat_nums, channels=64, act=ACT.RELU, in_channels=3, query_channels=128,
                 num_query=100, num_cls=20, img_size=(224, 224), numl_encoder=2, numl_decoder=2,
                 features_fedfwd=1024, ):
        super(DETRResNetMain, self).__init__(Module, repeat_nums, channels=channels, act=act, in_channels=in_channels)
        self.projtor = Ck1s1(in_channels=channels * 8, out_channels=query_channels)
        self.query_embed = nn.Embedding(num_query, query_channels)
        self.transformer = TransformerWithPE(
            d_model=query_channels, nhead=8, num_encoder_layers=numl_encoder, num_decoder_layers=numl_decoder,
            dim_feedforward=features_fedfwd, batch_first=True, dropout=0.1)
        self.layer = DETRLayer(num_cls=num_cls, img_size=img_size, in_features=query_channels)
        self._in_channels = in_channels
        self.query_channels = query_channels
        self.img_size = img_size
        self.num_cls = num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size
        self.pos_embed = _pos_embd_from_size(
            size=(img_size[0] // 32, img_size[1] // 32), features=self.query_channels // 2)

    @property
    def in_channels(self):
        return self._in_channels

    def forward(self, imgs):
        feat4 = super(DETRResNetMain, self).forward(imgs)
        feat4_proj = self.projtor(feat4)

        pos_embed = self.pos_embed.to(imgs.device).expand(feat4_proj.size())
        src_embed = pos_embed.flatten(2).permute(0, 2, 1)

        tgt_embed = self.query_embed.weight[None].repeat(imgs.size(0), 1, 1)
        src = feat4_proj.flatten(2).permute(0, 2, 1)
        tgt = torch.zeros_like(tgt_embed, dtype=torch.float32, device=imgs.device)

        hs = self.transformer(src=src, tgt=tgt, src_pos_embd=src_embed, tgt_pos_embd=tgt_embed)
        return self.layer(hs)

    PARA_R18 = dict(ResNetBkbn.PARA_R18, numl_encoder=3, numl_decoder=3, features_fedfwd=1024, query_channels=256)
    PARA_R34 = dict(ResNetBkbn.PARA_R34, numl_encoder=4, numl_decoder=4, features_fedfwd=1024, query_channels=256)
    PARA_R50 = dict(ResNetBkbn.PARA_R50, numl_encoder=6, numl_decoder=6, features_fedfwd=2048, query_channels=256)
    PARA_R101 = dict(ResNetBkbn.PARA_R50, numl_encoder=6, numl_decoder=6, features_fedfwd=2048, query_channels=256)

    @staticmethod
    def R18(act=ACT.RELU, num_cls=10, img_size=(224, 224), in_channels=3, num_queries=100):
        return DETRResNetMain(**DETRResNetMain.PARA_R18, act=act, num_cls=num_cls, num_query=num_queries,
                              img_size=img_size, in_channels=in_channels)

    @staticmethod
    def R34(act=ACT.RELU, num_cls=10, img_size=(224, 224), in_channels=3, num_queries=100):
        return DETRResNetMain(**DETRResNetMain.PARA_R34, act=act, num_cls=num_cls, num_query=num_queries,
                              img_size=img_size, in_channels=in_channels)

    @staticmethod
    def R50(act=ACT.RELU, num_cls=10, img_size=(224, 224), in_channels=3, num_queries=100, ):
        return DETRResNetMain(**DETRResNetMain.PARA_R50, act=act, num_cls=num_cls, num_query=num_queries,
                              img_size=img_size, in_channels=in_channels)

    @staticmethod
    def R101(act=ACT.RELU, num_cls=10, img_size=(224, 224), in_channels=3, num_queries=100, ):
        return DETRResNetMain(**DETRResNetMain.PARA_R101, act=act, num_cls=num_cls, num_query=num_queries,
                              img_size=img_size, in_channels=in_channels)


class DETR(OneStageTorchModel, IndependentInferableModel):

    def __init__(self, backbone, device=None, pack=PACK.AUTO, num_cls=20):
        super(DETR, self).__init__(backbone=backbone, device=device, pack=pack)
        self._num_cls = num_cls
        self.cost_weight_iou = 2
        self.cost_weight_l1 = 5
        self.cost_weight_cls = 1

    @property
    def img_size(self):
        return self.backbone.img_size

    @property
    def num_cls(self):
        return self._num_cls

    def imgs2labels(self, imgs, cind2name=None, conf_thres=0.5, **kwargs):
        self.eval()
        imgsT, ratios = imgs2imgsT(imgs, img_size=self.img_size)
        preds = self.pkd_modules['backbone'](imgsT.to(self.device))
        xywhssT, chotssT = torch.split(preds, split_size_or_sections=(4, self.num_cls + 1), dim=2)
        xyxyssT = xyxysT_clip(xywhsT2xyxysT(xywhssT), xyxyN_rgn=np.array(self.img_size))
        chotssT = torch.softmax(chotssT, dim=2)[..., :self.num_cls]
        confssT, cindssT = torch.max(chotssT, dim=2)
        labels = []
        for i, (xyxysT, confsT, cindsT) in enumerate(zip(xyxyssT, confssT, cindssT)):
            prsv_mask = confsT > conf_thres
            xyxysT, confsT, cindsT = xyxysT[prsv_mask], confsT[prsv_mask], cindsT[prsv_mask]
            boxes = BoxesLabel.from_xyxysT_confsT_cindsT(
                xyxysT=xyxysT, confsT=confsT, cindsT=cindsT, img_size=self.img_size, num_cls=self.num_cls,
                cind2name=cind2name)
            labels.append(boxes)
        return labels_rescale(labels, imgs2img_sizes(imgs), 1 / ratios)

    def labels2tars(self, labels, **kwargs):
        xywhss_tg = []
        cindss_tg = []
        for label in labels:
            xywhs = label.export_xywhsN()
            cinds = label.export_cindsN()
            xywhss_tg.append(xywhs)
            cindss_tg.append(cinds)
        return xywhss_tg, cindss_tg

    def imgs_tars2loss(self, imgs, targets, **kwargs):
        self.train()
        xywhss_tg, cindss_tg = arrsN2arrsT(targets, device=self.device)
        preds = self.pkd_modules['backbone'](imgs.to(self.device))
        xywhss_pd, chotss_pd = torch.split(preds, split_size_or_sections=(4, self.num_cls + 1), dim=2)
        chotss_pd_sft = torch.softmax(chotss_pd, dim=2)
        xywhss_pd = xywhss_pd.contiguous()
        scaler = np.sqrt(np.prod(self.img_size))

        iou_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        cls_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        l1_loss = torch.as_tensor(0).to(preds.device, non_blocking=True)
        num_mtch = 0
        for i, (xywhs_tg, cinds_tg, xywhs_pd, chots_pd, chots_pd_sft) in enumerate(
                zip(xywhss_tg, cindss_tg, xywhss_pd, chotss_pd, chotss_pd_sft)):
            cinds_tg_align = torch.full(
                size=(chots_pd.size(0),), fill_value=self.num_cls, device=self.device, dtype=torch.long)
            if cinds_tg.size(0) > 0:
                cost_cls = -chots_pd_sft[:, cinds_tg]
                cost_iou = 1 - ropr_mat_xywhsT(xywhs_pd, xywhs_tg, opr_type=OPR_TYPE.IOU)
                cost_l1 = torch.cdist(xywhs_pd, xywhs_tg, p=1) / scaler
                cost = cost_cls * self.cost_weight_cls + cost_iou * self.cost_weight_iou + cost_l1 * self.cost_weight_l1
                inds_pd, inds_tg = linear_sum_assignment(cost.detach().cpu().numpy())
                # print(cost_iou)
                iou_loss = iou_loss + torch.sum(cost_iou[inds_pd, inds_tg])
                l1_loss = l1_loss + torch.sum(cost_l1[inds_pd, inds_tg])
                cinds_tg_align[inds_pd] = cinds_tg[inds_tg]
                num_mtch = num_mtch + inds_pd.shape[0]
            # 分类损失
            cls_loss = cls_loss + F.cross_entropy(chots_pd, cinds_tg_align, reduction='mean')

        num_mtch = max(num_mtch, 1)
        iou_loss = iou_loss / num_mtch
        l1_loss = l1_loss / num_mtch
        cls_loss = cls_loss / imgs.size(0)
        return OrderedDict(cls=cls_loss, l1=l1_loss * 5, iou=iou_loss * 2)

    @staticmethod
    def ResNetR34(device=None, pack=PACK.AUTO, num_cls=10, img_size=(224, 224), in_channels=3, num_queries=100):
        backbone = DETRResNetMain.R34(act=ACT.RELU, num_cls=num_cls + 1, img_size=img_size, in_channels=in_channels,
                                      num_queries=num_queries)
        return DETR(backbone, device=device, pack=pack, num_cls=num_cls)

    @staticmethod
    def ResNetR50(device=None, pack=PACK.AUTO, num_cls=10, img_size=(224, 224), in_channels=3, num_queries=100):
        backbone = DETRResNetMain.R50(act=ACT.RELU, num_cls=num_cls + 1, img_size=img_size, in_channels=in_channels,
                                      num_queries=num_queries)
        return DETR(backbone, device=device, pack=pack, num_cls=num_cls)

    @staticmethod
    def Const(device=None, batch_size=1, num_cls=10, img_size=(224, 224)):
        backbone = DETRConstMain(batch_size=batch_size, num_cls=num_cls + 1, img_size=img_size)
        return DETR(backbone, device=device, pack=PACK.NONE, num_cls=num_cls)

# if __name__ == '__main__':


#     mask = torch.rand(4, 224, 124) > 0.5  # [bs,h,w]
#     pos_emb = PositionEmbeddingSine(128)
#     res = pos_emb(mask)  # [bs,dim,h,w]
#     print(res.shape)

# if __name__ == '__main__':
#     model = DETRResNetMain.R50()
#     # x = torch.rand(size=(1, 3, 224, 224))
#     # y = model(x)
#     model.export_onnx('./buff')

# if __name__ == '__main__':
#     model = TransformerWithPE(d_model=128, num_head=8, num_encoder_layers=2,
#                               num_decoder_layers=2, dim_feedforward=1024, batch_first=True)
#     src = torch.rand(1, 10, 128)
#     tgt = torch.rand(1, 10, 128)
#     src_embd = torch.rand(1, 10, 128)
#     tgt_embd = torch.rand(1, 10, 128)
#     output = model(src, tgt, src_pos_embd=src_embd, tgt_pos_embd=tgt_embd)
#     print(output.size())
