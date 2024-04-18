from utils import *


# <editor-fold desc='图像读取'>
def imgwt_pil(img_pth):
    return Image.open(img_pth).convert('RGB')


# </editor-fold>

def encode_meta_xyxy(meta, xyxy):
    return meta + '_' + '_'.join(['%04d' % v for v in xyxy])


def decode_meta_xyxy(meta):
    meta_p = meta.split('_')
    xyxy = np.array([int(v) for v in meta_p[-4:]])
    meta = '_'.join(meta_p[:-4])
    return meta, xyxy


def meta_merge_dir(file_dir):
    mapper = {}
    if os.path.isdir(file_dir):
        file_names = os.listdir(file_dir)
    else:
        file_names = load_txt(file_dir)
    for file_name in file_names:
        meta, xyxy = decode_meta_xyxy(file_name.split('.')[0])
        mapper[meta] = mapper.get(meta, []) + [xyxy]
    return mapper


class ItemFilterBasic(ItemFilterValueEqual):

    def __init__(self, cls_names=None, thres=-1, **kwargs):
        self.cls_names = cls_names
        self.thres = thres
        ItemFilterValueEqual.__init__(self, **kwargs)

    def __call__(self, item):
        if not ItemFilterValueEqual.__call__(self, item):
            return False
        if self.cls_names is not None and item['name'] not in self.cls_names:
            return False
        if self.thres > 0 and item.measure < self.thres:
            return False
        return True


class Extractor(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, items, xyxy_rgn, **kwargs):
        pass


class ExtractorClip(Extractor):
    def __init__(self, fltr=None):
        self.fltr = fltr

    def __call__(self, items, xyxy_rgn, index=None, **kwargs):
        items_fltd = []
        xyxy_rgn = xyxy_rgn.astype(np.int32)
        patch_size = (xyxy_rgn[2:4] - xyxy_rgn[:2])
        for item in items if self.fltr is None else filter(self.fltr, items):
            item_cp = copy.deepcopy(item)
            if index is not None:
                item_cp['main'] = item == items[index]
            if item_cp.clip(xyxy_rgn).measure < 1:
                continue
            item_cp.linear_(bias=-xyxy_rgn[:2], size=patch_size)
            if isinstance(item_cp, InstItem):
                item_cp.align()
            items_fltd.append(item_cp)
        return items_fltd


class ExtractorIndex(Extractor):
    def __init__(self, ):
        pass

    def __call__(self, items, xyxy_rgn, index=None, **kwargs):
        xyxy_rgn = xyxy_rgn.astype(np.int32)
        patch_size = (xyxy_rgn[2:4] - xyxy_rgn[:2])
        item = copy.deepcopy(items[index])
        item.linear_(bias=-xyxy_rgn[:2], size=patch_size)
        return [item]


# 检测框样本区域扩展策略
def _xyxyN_expand(xyxy: np.ndarray, xyxyN_rgn: np.ndarray, expend_ratio: float = 1.1,
                  as_square: bool = False, with_clip: bool = False, min_expand: int = 3) -> np.ndarray:
    xywh = xyxyN2xywhN(xyxy)
    xywh[2:4] = np.maximum(xywh[2:4] * expend_ratio, xywh[2:4] + 2 * min_expand)
    if as_square:
        xywh[2:4] = np.max(xywh[2:4])
    xywh = np.round(xywh).astype(np.int32)
    xyxy = xywhN2xyxyN(xywhN=xywh)
    if with_clip:
        xyxy = xyxyN_clip(xyxy, xyxyN_rgn=xyxyN_rgn)
    return xyxy


def _xyxysN_expand(xyxys: np.ndarray, xyxyN_rgn: np.ndarray, expend_ratio: float = 1.1,
                   as_square: bool = False, with_clip: bool = False, min_expand: int = 3) -> np.ndarray:
    xywhs = xyxysN2xywhsN(xyxys)
    xywhs[:, 2:4] = np.maximum(xywhs[:, 2:4] * expend_ratio, xywhs[:, 2:4] + 2 * min_expand)
    if as_square:
        xywhs[:, 2:4] = np.max(xywhs[:, 2:4], keepdims=True, axis=1)
    xywhs = np.round(xywhs).astype(np.int32)
    xyxys = xywhsN2xyxysN(xywhs)
    if with_clip:
        xyxys = xyxysN_clip(xyxys, xyxyN_rgn=xyxyN_rgn)
    return xyxys


# 子区域裁剪策略
def _genrgns_persize(xyxyN: np.ndarray, piece_sizeN: np.ndarray = np.array([640, 640]),
                     over_lapN: np.ndarray = np.array([100, 100]),
                     with_clip: bool = True, ) -> np.ndarray:
    step_size = piece_sizeN - over_lapN
    full_size = xyxyN[2:4] - xyxyN[:2]
    assert np.all(step_size > 0), 'size err'
    nwh = np.ceil((full_size - over_lapN) / step_size).astype(np.int32)
    ofys, ofxs = arange2dN(nwh[1], nwh[0])
    offsets = np.stack([ofxs, ofys], axis=2).reshape(-1, 2)
    xy1s = offsets * step_size + xyxyN[:2]
    rgns = np.concatenate([xy1s, xy1s + piece_sizeN], axis=1)

    if with_clip:
        rgns = xyxysN_clip(rgns, xyxyN_rgn=xyxyN)
    return rgns


def _genrgns_pyramid(xyxyN: np.ndarray, piece_sizesN: np.ndarray = np.array([[640, 640], [320, 320]]),
                     over_lapsN: np.ndarray = np.array([[100, 100], [50, 50]]),
                     with_clip: bool = True, unique: bool = True, ) -> np.ndarray:
    rgns = [np.zeros(shape=(0, 4), dtype=np.float32)]
    for piece_size, over_lap in zip(piece_sizesN, over_lapsN):
        rgns_i = _genrgns_persize(xyxyN, piece_sizeN=piece_size, over_lapN=over_lap, with_clip=with_clip)
        rgns.append(rgns_i)
    rgns = np.concatenate(rgns, axis=0)
    if unique:
        rgns = np.unique(rgns, axis=0)
    return rgns


def img2piece_persize(img, items, piece_size=(640, 640), over_lap=(100, 100), ignore_empty=True,
                      with_clip=True, fltr=None, meta_encoder=None, extractor=None):
    meta_encoder = encode_meta_xyxy if meta_encoder is None else meta_encoder
    extractor = ExtractorClip() if extractor is None else extractor
    imgP = img2imgP(img)
    items_fltd = items.filt(fltr=fltr)
    assert isinstance(items_fltd, ImageItemsLabel), 'fmt err ' + items_fltd.__class__.__name__
    img_rgn = np.array([0, 0, imgP.size[0], imgP.size[1]])
    piece_rgns = _genrgns_persize(img_rgn, piece_sizeN=np.array(piece_size),
                                  over_lapN=np.array(over_lap), with_clip=with_clip).astype(np.int32)
    pieces = []
    plabels = []
    for piece_rgn in piece_rgns:
        meta_piece = meta_encoder(items_fltd.meta, piece_rgn)
        piece_size_j = tuple(piece_rgn[2:] - piece_rgn[:2])
        pitems = items_fltd.__class__(img_size=piece_size_j, meta=meta_piece)
        pitems += extractor(items_fltd, xyxy_rgn=piece_rgn)
        if ignore_empty and len(pitems) == 0:
            continue
        piece = imgP.crop(list(piece_rgn))
        plabels.append(pitems)
        pieces.append(piece)
    return pieces, plabels


def img2piece_pyramid(img, items, piece_sizes=((640, 640), (320, 320)), over_laps=((100, 100), (50, 50)),
                      ignore_empty=True, unique=True,
                      with_clip=True, fltr=None, meta_encoder=None, extractor=None):
    meta_encoder = encode_meta_xyxy if meta_encoder is None else meta_encoder
    extractor = ExtractorClip() if extractor is None else extractor
    imgP = img2imgP(img)
    items_fltd = items.filt(fltr=fltr)
    assert isinstance(items_fltd, ImageItemsLabel), 'fmt err ' + items_fltd.__class__.__name__
    img_rgn = np.array([0, 0, imgP.size[0], imgP.size[1]])
    piece_rgns = _genrgns_pyramid(img_rgn, piece_sizesN=np.array(piece_sizes), unique=unique,
                                  over_lapsN=np.array(over_laps), with_clip=with_clip).astype(np.int32)
    pieces = []
    plabels = []
    for piece_rgn in piece_rgns:
        meta_piece = meta_encoder(items_fltd.meta, piece_rgn)
        piece_size_j = tuple(piece_rgn[2:] - piece_rgn[:2])
        pitems = items_fltd.__class__(img_size=piece_size_j, meta=meta_piece)
        pitems += extractor(items_fltd, xyxy_rgn=piece_rgn)
        if ignore_empty and len(pitems) == 0:
            continue
        piece = imgP.crop(list(piece_rgn))
        plabels.append(pitems)
        pieces.append(piece)
    return pieces, plabels


def img2piece_perbox(img, items, expend_ratio=1.2, with_clip=False, as_square=False,
                     fltr=None, ignore_empty=True, meta_encoder=None, extractor=None):
    meta_encoder = encode_meta_xyxy if meta_encoder is None else meta_encoder
    extractor = ExtractorClip() if extractor is None else extractor
    imgP = img2imgP(img)
    items_fltd = items.filt(fltr=fltr)
    xyxys, cinds = items_fltd.export_xyxysN(), items_fltd.export_cindsN()
    img_rgn = np.array([0, 0, imgP.size[0], imgP.size[1]])
    piece_rgns = _xyxysN_expand(
        xyxys, xyxyN_rgn=img_rgn, expend_ratio=expend_ratio, as_square=as_square, with_clip=with_clip)
    pieces = []
    plabels = []
    for j, item in enumerate(items_fltd):
        xyxy_rgn = piece_rgns[j]
        meta_piece = meta_encoder(items_fltd.meta, xyxy_rgn)
        piece_size_j = tuple(xyxy_rgn[2:4] - xyxy_rgn[:2])
        pitems = items_fltd.__class__(img_size=piece_size_j, meta=meta_piece)
        pitems += extractor(items, xyxy_rgn=xyxy_rgn, index=items.index(item))
        if ignore_empty and len(pitems) == 0:
            continue
        piece = img.crop(list(xyxy_rgn))
        plabels.append(pitems)
        pieces.append(piece)
    return pieces, plabels


def img2img_filt(img, items, fltr=None, ignore_empty=True):
    items_fltd = items.filt(fltr=fltr)
    if ignore_empty and len(items_fltd) == 0:
        return [], []
    else:
        img = img2imgP(img)
        return [img], [items_fltd]


def img2background(img, items, min_size=0, max_size=16, repeat_num=1.0, fltr=None):
    pieces = []
    plabels = []
    items_fltd = items.filt(fltr=fltr)
    xyxys_fltd = items_fltd.export_xyxysN()
    img = img2imgP(img)
    for j in range(int(np.ceil(repeat_num))):
        if j + np.random.rand() > repeat_num: continue
        size_ratio = np.random.rand()
        cur_size = int(size_ratio * max_size + (1 - size_ratio) * min_size)
        offset = ((np.array(img.size) - cur_size) * np.random.rand(2)).astype(np.int32)
        xyxy = np.concatenate([offset, offset + cur_size], axis=0)
        iareas = ropr_arr_xyxysN(np.repeat(xyxy[None, :], axis=0, repeats=xyxys_fltd.shape[0]), xyxys_fltd,
                                 opr_type=OPR_TYPE.IAREA)
        if np.any(iareas > 0): continue

        piece = img.crop(xyxy)
        meta = encode_meta_xyxy(items.meta, xyxy)
        pitems = items.__class__(img_size=(cur_size, cur_size), meta=meta)
        plabels.append(pitems)
        pieces.append(piece)
    return pieces, plabels


def dataset2background(dataset, bkgd_dir, min_size=0, max_size=16, repeat_num=1.0,
                       fltr=None, broadcast=BROADCAST):
    ensure_folder_pth(bkgd_dir)
    broadcast('Start convert [ ' + dataset.__class__.__name__ + ' ]')
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces_i, plabels_i = img2background(img, items, min_size=min_size, max_size=max_size,
                                             repeat_num=repeat_num, fltr=fltr)
        for piece_i, plabel_i in zip(pieces_i, plabels_i):
            piece_pth = os.path.join(bkgd_dir, plabel_i.meta + '.jpg')
            piece_i.save(piece_pth)

    broadcast('Convert complete')
    return None


def dataset2piece_persize(dataset, piece_size=(640, 640), over_lap=(100, 100), with_clip=False,
                          fltr=None, ignore_empty=True, broadcast=BROADCAST):
    pieces = []
    plabels = []
    broadcast('Start convert [ ' + dataset.__class__.__name__ + ' ]')
    for i, (img, items) in MEnumerate(dataset):
        pieces_i, plabels_i = img2piece_persize(img, items, piece_size=piece_size,
                                                over_lap=over_lap, fltr=fltr, ignore_empty=ignore_empty,
                                                with_clip=with_clip)
        pieces += pieces_i
        plabels += plabels_i
    broadcast('Convert complete')
    return pieces, plabels


_BORDER_CMPLX = {
    XYXYBorder: 0,
    XYWHBorder: 0,
    XYWHABorder: 1,
    XLYLBorder: 2,

}


def _select_border_type(boxes: BoxesLabel):
    border_type = XYXYBorder
    cmplx = _BORDER_CMPLX[border_type]
    for box in boxes:
        border_type_cur = box.border.__class__
        cmplx_cur = _BORDER_CMPLX[border_type_cur]
        if cmplx_cur > cmplx:
            border_type = border_type_cur
            cmplx = cmplx_cur
    return border_type


def _boxes_nms(boxes: BoxesLabel, border_type=XYXYBorder, iou_thres=0.0, by_cls=True, iou_type=IOU_TYPE.IOU, ):
    cindsN, confsN = boxes.export_cindsN_confsN()
    border_type = border_type if border_type is not None else _select_border_type(boxes)
    cindsN = cindsN if by_cls else None
    if border_type == XYXYBorder or border_type == XYWHBorder:
        xyxysN = boxes.export_xyxysN()
        prsv_inds = nms_xyxysN(xyxysN, confsN, cindsN=cindsN, iou_thres=iou_thres,
                               nms_type=NMS_TYPE.HARD, iou_type=iou_type, )
    else:
        xlylNs = boxes.export_xlyls()
        prsv_inds = nms_xlylNs(xlylNs, confsN, cindsN, iou_thres=iou_thres, iou_type=IOU_TYPE.IOU)
    return prsv_inds


def piece_merge_img(plabels, xyxy_rgns, iou_thres=0.0, by_cls=True, meta=None, iou_type=IOU_TYPE.IOU, ):
    max_size = np.array([-np.inf, -np.inf])
    label = []
    for plabel, xyxy_rgn in zip(plabels, xyxy_rgns):
        max_size = np.maximum(max_size, xyxy_rgn[2:])
        cur_size = np.array(plabel.img_size)
        rgn_size = xyxy_rgn[2:] - xyxy_rgn[:2]
        plabel.linear_(bias=xyxy_rgn[:2], scale=rgn_size / cur_size, size=tuple(xyxy_rgn[2:]))
        label += plabel

    img_size = tuple(max_size.astype(np.int32))
    if iou_thres > 0:
        boxes_label = BoxesLabel(label, img_size=img_size, meta=meta)
        prsv_inds = _boxes_nms(boxes_label, iou_thres=iou_thres, by_cls=by_cls, iou_type=iou_type)
        label = [label[i] for i in prsv_inds]

    label = ImageItemsLabel(label, img_size=img_size, meta=meta)
    return label


def piece_merge(plabels, iou_thres=0.0, meta_decoder=None, broadcast=BROADCAST):
    meta_dict = {}
    meta_decoder = decode_meta_xyxy if meta_decoder is None else meta_decoder
    broadcast('Cluster label for every image')
    plabels = copy.deepcopy(plabels)
    for plabel in plabels:
        meta, xyxy_rgn = meta_decoder(plabel.meta)
        rgn_plb = (xyxy_rgn, plabel)
        if meta in meta_dict.keys():
            meta_dict[meta].append(rgn_plb)
        else:
            meta_dict[meta] = [rgn_plb]
    labels = []
    broadcast('Merge label for every image')
    for i, (meta, rgn_plbs) in MEnumerate(meta_dict.items(), broadcast=broadcast):
        rgns, plabels = zip(*rgn_plbs)
        label = piece_merge_img(plabels=plabels, xyxy_rgns=rgns, iou_thres=iou_thres, meta=meta)
        labels.append(label)
    return labels

# if __name__ == '__main__':
#     uname = platform.uname()
#     print(uname)
