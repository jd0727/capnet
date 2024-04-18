import os
from collections import Counter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from data.processing import *
from utils import *
import xml.etree.ElementTree as ET
import ast

# <editor-fold desc='ds xml标签转化'>

XMLRD_MAPPER = OrderedDict()


def xmlrd_registor(*tags):
    def wrapper(_xmlrd):
        for tag in tags:
            XMLRD_MAPPER[tag] = _xmlrd
        return _xmlrd

    return wrapper


def xmlrd_items(node, **kwargs):
    items = []
    for i, sub_node in enumerate(list(node)):
        if node.tag == 'annotation' and sub_node.tag == 'annotation':
            continue
        if sub_node.tag in XMLRD_MAPPER.keys():
            reader = XMLRD_MAPPER[sub_node.tag]
            item = reader(sub_node, **kwargs)
            items.append(item)
    return items


XMLWT_MAPPER = OrderedDict()


def xmlwt_registor(*item_types):
    def wrapper(_xmlwt):
        for item_type in item_types:
            XMLWT_MAPPER[item_type] = _xmlwt
        return _xmlwt

    return wrapper


def xmlwt_item(node, item, **kwargs):
    writer = XMLWT_MAPPER[item.__class__]
    sub_node = writer(node, item, **kwargs)
    return sub_node


def xmlwt_items(node, items, **kwargs):
    for item in items:
        writer = XMLWT_MAPPER[item.__class__]
        writer(node, item, **kwargs)
    return True


TAG_ROOT = 'annotation'
TAG_META = 'filename'
TAG_SIZE = 'size'
TAG_WIDTH = 'width'
TAG_HEIGHT = 'height'
TAG_DEPTH = 'depth'

TAG_XYXY = 'bndbox'
TAGS_XYXY_ITEM = ('xmin', 'ymin', 'xmax', 'ymax')

TAG_XYWH = 'bndboxw'
TAGS_XYWH_ITEM = ('cx', 'cy', 'w', 'h')

TAG_XYWHA = 'robndbox'
TAGS_XYWHA_ITEM = ('cx', 'cy', 'w', 'h', 'angle')

TAG_XLYL = 'polygon'

TAG_BOXITEM = 'object'
TAG_BOXREFITEM = 'object_ref'
TAGS_BOXITEM = (TAG_BOXITEM, TAG_BOXREFITEM)

TAG_NAME = 'name'
TAG_DIFFICULT = 'difficult'
TAG_TRUNCATED = 'truncated'
KEY_NAME = TAG_NAME
KEY_DIFFICULT = TAG_DIFFICULT
KEY_TRUNCATED = TAG_TRUNCATED
TAGS_IGNORE_BOXITEM = (TAG_NAME, TAG_DIFFICULT, TAG_TRUNCATED, TAG_XYXY, TAG_XYWH, TAG_XLYL, TAG_XYWHA)
TAGS_IGNORE_LABEL = (TAG_BOXITEM, TAG_BOXREFITEM, TAG_SIZE, TAG_META)
KEYS_IGNORE_BOXITEM = (KEY_NAME, KEY_DIFFICULT, KEY_TRUNCATED)


def _xmlrd_img_size(node):
    size = node.find(TAG_SIZE)
    img_size = (int(size.find(TAG_WIDTH).text), int(size.find(TAG_HEIGHT).text))
    return img_size


def _xmlwt_img_size(node, img_szie, depth=3):
    size = ET.SubElement(node, TAG_SIZE)
    ET.SubElement(size, TAG_WIDTH).text = str(int(img_szie[0]))
    ET.SubElement(size, TAG_HEIGHT).text = str(int(img_szie[1]))
    ET.SubElement(size, TAG_DEPTH).text = str(int(depth))
    return node


def _xmlrd_meta(node):
    meta = node.find(TAG_META).text.split('.')[0]
    return meta


def _xmlwt_meta(node, meta, img_extend='jpg'):
    ET.SubElement(node, TAG_META).text = ensure_extend(meta, img_extend)
    return node


def _xmlrd_bool(node, tag, default=False):
    val = node.find(tag)
    val = int(val.text) == 1 if val is not None else default
    return val


def _xmlwt_bool(node, tag, val):
    ET.SubElement(node, tag).text = '1' if val else '0'
    return val


def _xmlrd_dict(node, ignore_tages=None):
    full_dict = {}
    for sub_node in node:
        if ignore_tages is not None and sub_node.tag in ignore_tages:
            continue
        if len(sub_node) == 0:
            try:
                val = ast.literal_eval(sub_node.text)
            except Exception as e:
                val = sub_node.text
            full_dict[sub_node.tag] = val
        else:
            full_dict[sub_node.tag] = _xmlrd_dict(sub_node, ignore_tages=ignore_tages)
    return full_dict


def _xmlwt_dict(node, dct, ignore_keys=None):
    node = ET.Element(node) if isinstance(node, str) else node
    if isinstance(dct, dict):
        for key, val in dct.items():
            if ignore_keys is not None and key in ignore_keys:
                continue
            sub_node = ET.SubElement(node, key) if node.find(key) is None else node.find(key)
            _xmlwt_dict(node=sub_node, dct=val, ignore_keys=ignore_keys)
    else:
        dct = dct.tolist() if isinstance(dct, np.ndarray) and dct.size > 1 else dct
        node.text = str(dct)
    return node


@xmlrd_registor(TAG_XYXY)
def _xmlrd_border_xyxy(node, size, **kwargs):
    xyxy = np.array([float(node.find(item).text) for item in TAGS_XYXY_ITEM])
    border = XYXYBorder(xyxyN=xyxy, size=size)
    return border


@xmlwt_registor(XYXYBorder)
def _xmlwt_border_xyxy(node, border):
    sub_node = ET.SubElement(node, TAG_XYXY)
    for i, item in enumerate(TAGS_XYXY_ITEM):
        ET.SubElement(sub_node, item).text = str(border.xyxyN[i])
    return sub_node


@xmlrd_registor(TAG_XYWHA)
def _xmlrd_border_xywha(node, size, **kwargs):
    xywha = np.array([float(node.find(item).text) for item in TAGS_XYWHA_ITEM])
    border = XYWHABorder(xywhaN=xywha, size=size)
    return border


@xmlwt_registor(XYWHABorder)
def _xmlwt_border_xywha(node, border):
    sub_node = ET.SubElement(node, TAG_XYWHA)
    for i, item in enumerate(TAGS_XYWHA_ITEM):
        ET.SubElement(sub_node, item).text = str(border.xywhaN[i])
    return sub_node


@xmlrd_registor(TAG_XYWH)
def _xmlrd_border_xywh(node, size, **kwargs):
    xywh = np.array([float(node.find(item).text) for item in TAGS_XYWH_ITEM])
    border = XYWHBorder(xywhN=xywh, size=size)
    return border


@xmlwt_registor(XYWHBorder)
def _xmlwt_border_xywh(node, border):
    sub_node = ET.SubElement(node, TAG_XYWH)
    for i, item in enumerate(TAGS_XYWH_ITEM):
        ET.SubElement(sub_node, item).text = str(border.xywhN[i])
    return sub_node


@xmlrd_registor(TAG_XLYL)
def _xmlrd_border_xlyl(node, size, **kwargs):
    vals = ast.literal_eval(node.text)
    xlyl = np.array(vals).reshape(-1, 2)
    border = XLYLBorder(xlylN=xlyl, size=size)
    return border


@xmlwt_registor(XLYLBorder)
def _xmlwt_border_xlyl(node, border):
    xlyl = border.xlylN.reshape(-1)
    vals = str(list(xlyl))
    sub_node = ET.SubElement(node, TAG_XLYL)
    sub_node.text = vals
    return sub_node


@xmlrd_registor(TAG_BOXITEM)
def _xmlrd_boxitem(node, size, name2cind=None, num_cls=1, **kwargs):
    name = node.find(TAG_NAME).text
    cind = name2cind(name) if name2cind is not None else 0
    category = IndexCategory(cindN=cind, num_cls=num_cls, conf=1)

    attrs = _xmlrd_dict(node, ignore_tages=TAGS_IGNORE_BOXITEM)
    attrs[KEY_DIFFICULT] = _xmlrd_bool(node, tag=TAG_DIFFICULT)
    attrs[KEY_TRUNCATED] = _xmlrd_bool(node, tag=TAG_TRUNCATED)
    attrs[KEY_NAME] = name

    border = xmlrd_items(node, size=size)[0]
    box = BoxItem(border=border, category=category, **attrs)
    return box


@xmlrd_registor(TAG_BOXREFITEM)
def _xmlrd_boxrefitem(node, size, name2cind=None, num_cls=1, **kwargs):
    name = node.find(TAG_NAME).text
    cind = name2cind(name) if name2cind is not None else 0
    category = IndexCategory(cindN=cind, num_cls=num_cls, conf=1)

    attrs = _xmlrd_dict(node, ignore_tages=TAGS_IGNORE_BOXITEM)
    attrs[KEY_DIFFICULT] = _xmlrd_bool(node, tag=TAG_DIFFICULT)
    attrs[KEY_TRUNCATED] = _xmlrd_bool(node, tag=TAG_TRUNCATED)
    attrs[KEY_NAME] = name

    border, border_ref = xmlrd_items(node, size=size)
    box = BoxRefItem(border=border, border_ref=border_ref, category=category, **attrs)
    return box


@xmlrd_registor(TAG_ROOT)
def _xmlrd_boxes(node, name2cind=None, num_cls=1, ):
    meta = _xmlrd_meta(node)
    img_size = _xmlrd_img_size(node)
    boxes = xmlrd_items(node, size=img_size, name2cind=name2cind, num_cls=num_cls)
    attrs = _xmlrd_dict(node, ignore_tages=TAGS_IGNORE_LABEL)
    boxes = BoxesLabel(boxes, meta=meta, img_size=img_size, **attrs)
    return boxes


@xmlwt_registor(BoxItem, InstItem)
def _xmlwt_boxitem(node, box, **kwargs):
    obj = ET.SubElement(node, TAG_BOXITEM)
    xmlwt_item(obj, box.border)
    ET.SubElement(obj, TAG_NAME).text = box.get(KEY_NAME, 'Unknown')
    _xmlwt_bool(obj, tag=TAG_DIFFICULT, val=box.get(KEY_DIFFICULT, False))
    _xmlwt_bool(obj, tag=TAG_TRUNCATED, val=box.get(KEY_TRUNCATED, False))
    _xmlwt_dict(obj, box, ignore_keys=KEYS_IGNORE_BOXITEM)
    return node


@xmlwt_registor(BoxRefItem, InstRefItem)
def _xmlwt_boxrefitem(node, box, **kwargs):
    obj = ET.SubElement(node, TAG_BOXREFITEM)
    xmlwt_item(obj, box.border)
    xmlwt_item(obj, box.border_ref)
    ET.SubElement(obj, TAG_NAME).text = box.get(KEY_NAME, 'Unknown')
    _xmlwt_bool(obj, tag=TAG_DIFFICULT, val=box.get(KEY_DIFFICULT, False))
    _xmlwt_bool(obj, tag=TAG_TRUNCATED, val=box.get(KEY_TRUNCATED, False))
    _xmlwt_dict(obj, box, ignore_keys=KEYS_IGNORE_BOXITEM)
    return obj


@xmlwt_registor(BoxesLabel, InstsLabel)
def _xmlwt_boxes(node, label, **kwargs):
    sub_node = ET.SubElement(node, TAG_ROOT)
    _xmlwt_dict(sub_node, label.kwargs)
    _xmlwt_meta(sub_node, label.meta)
    _xmlwt_img_size(sub_node, label.img_size)
    xmlwt_items(sub_node, label)
    return sub_node


# 修饰node标注
def pretty_node(element, indent='\t', newline='\n', level=0):
    if element:
        # 如果element的text没有内容
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_node(subelement, indent, newline, level=level + 1)
    return True


# </editor-fold>

# <editor-fold desc='ds mask标签转化'>
MASKRD_MAPPER = {}
MASKWT_MAPPER = {}


def maskwt_registor(*extends):
    def wrapper(_maskwt):
        for extend in extends:
            MASKWT_MAPPER[extend] = _maskwt
        return _maskwt

    return wrapper


def maskwt(items, mask_pth, **kwargs):
    _maskwt = MASKWT_MAPPER[mask_pth.split('.')[-1]]
    return _maskwt(items, mask_pth, **kwargs)


def maskrd_registor(*extends):
    def wrapper(_maskrd):
        for extend in extends:
            MASKRD_MAPPER[extend] = _maskrd
        return _maskrd

    return wrapper


def maskrd(mask_pth, **kwargs):
    _maskrd = MASKRD_MAPPER[mask_pth.split('.')[-1]]
    return _maskrd(mask_pth, **kwargs)


@maskwt_registor('jpg', 'png')
def _maskwt_pil(items, mask_pth, colors, **kwargs):
    maskN = np.zeros(shape=(items.img_size[1], items.img_size[0], 3))
    for item in items:
        cind = IndexCategory.convert(item.category).cindN
        maskN[item.rgn.maskNb, :3] = colors[cind]
    maskP = imgN2imgP(maskN)
    maskP.save(mask_pth)
    return True


@maskrd_registor('jpg', 'png')
def _maskrd_pil(mask_pth, colors, num_cls, **kwargs):
    colors = [colors[i] for i in range(num_cls)]
    maskN = np.array(Image.open(mask_pth).convert('RGB'))
    maskN = np.all(maskN == np.array(colors), axis=3)
    has_cate = np.any(maskN, axis=(0, 1))
    items = SegsLabel(img_size=(maskN.shape[1], maskN.shape[0]))
    for cind in np.where(has_cate)[0]:
        rgn = AbsBoolRegion(maskN[:, :, cind])
        cate = IndexCategory(cindN=cind, num_cls=len(has_cate), conf=1)
        items.append(SegItem(rgn=rgn, category=cate))
    return items


@maskwt_registor('npy')
def _maskwt_npy(items, mask_pth, num_cls, **kwargs):
    maskN = np.zeros(shape=(items.img_size[1], items.img_size[0], num_cls))
    for item in items:
        cind = IndexCategory.convert(item.category).cindN
        rgn = AbsValRegion.convert(item.rgn)
        maskN[..., cind] = np.maximum(maskN[..., cind], rgn.maskN)
    np.save(mask_pth, maskN)
    return True


@maskrd_registor('npy')
def _maskrd_npy(mask_pth, **kwargs):
    maskN = np.load(mask_pth)
    has_cate = np.any(maskN, axis=(0, 1))
    items = SegsLabel(img_size=(maskN.shape[1], maskN.shape[0]))
    for cind in np.where(has_cate)[0]:
        rgn = AbsBoolRegion(maskN[:, :, cind])
        cate = IndexCategory(cindN=cind, num_cls=len(has_cate), conf=1)
        items.append(SegItem(rgn=rgn, category=cate))
    return items


@maskwt_registor('pkl')
def _maskwt_pkl(items, mask_pth, **kwargs):
    load_pkl(mask_pth, items)
    return True


@maskrd_registor('pkl')
def _maskrd_pkl(mask_pth, **kwargs):
    return load_pkl(mask_pth)


# </editor-fold>

# <editor-fold desc='ds inst标签转化'>
INSTRD_MAPPER = {}
INSTWT_MAPPER = {}


def instwt_registor(*extends):
    def wrapper(_instwt):
        for extend in extends:
            INSTWT_MAPPER[extend] = _instwt
        return _instwt

    return wrapper


def instwt(items, inst_pth, **kwargs):
    _instwt = INSTWT_MAPPER[inst_pth.split('.')[-1]]
    return _instwt(items, inst_pth, **kwargs)


def instrd_registor(*extends):
    def wrapper(_instrd):
        for extend in extends:
            INSTRD_MAPPER[extend] = _instrd
        return _instrd

    return wrapper


def instrd(inst_pth, boxes, **kwargs):
    _instrd = INSTRD_MAPPER[inst_pth.split('.')[-1]]
    return _instrd(inst_pth, boxes, **kwargs)


@instwt_registor('jpg', 'png')
def _instwt_pil(items, inst_pth, colors, **kwargs):
    maskN = np.zeros(shape=(items.img_size[1], items.img_size[0], 3))
    for i, item in enumerate(items):
        maskN[item.rgn.maskNb, :3] = colors[i]
    maskP = imgN2imgP(maskN)
    maskP.save(inst_pth)
    return True


@instrd_registor('jpg', 'png')
def _instrd_pil(inst_pth, boxes, colors, rgn_type=AbsBoolRegion, **kwargs):
    maskN = np.array(Image.open(inst_pth).convert("RGB"))
    size = (maskN.shape[1], maskN.shape[0])
    insts = InstsLabel(img_size=size)
    for i, box in enumerate(boxes):
        if rgn_type == RefValRegion:
            xyxy = XYXYBorder.convert(box.border).xyxyN.astype(np.int32)
            patchN = maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            patchNb = np.all(patchN == colors[i], axis=2)
            rgn = RefValRegion(xyN=xyxy[:2], maskN_ref=patchNb.astype(np.float32), size=size)
        elif rgn_type == AbsBoolRegion or rgn_type is None:
            rgn = AbsBoolRegion(maskNb_abs=np.all(maskN == colors[i], axis=2))
        else:
            raise Exception('err rgn type')
        inst = InstItem(border=box.border, category=box.category, rgn=rgn, **box)
        insts.append(inst)
    return insts


@instwt_registor('npy')
def _instwt_npy(items, inst_pth, **kwargs):
    maskN = [np.zeros(shape=(0, items.img_size[1], items.img_size[0]))]
    for item in items:
        rgn = AbsValRegion.convert(item.rgn)
        maskN.append(rgn.maskN)
    maskN = np.concatenate(maskN, axis=0)
    np.save(inst_pth, maskN)
    return True


@instrd_registor('npy')
def _instrd_npy(inst_pth, boxes, **kwargs):
    maskN = np.array(Image.open(inst_pth).convert("RGB"))
    size = (maskN.shape[1], maskN.shape[0])
    insts = InstsLabel(img_size=size)
    for i, box in enumerate(boxes):
        rgn = AbsValRegion(maskN[i])
        inst = InstItem(border=box.border, category=box.category, rgn=rgn, **box)
        insts.append(inst)
    return insts


@instwt_registor('pkl')
def _maskwt_pkl(items, inst_pth, **kwargs):
    rgns = [item.rgn for item in items]
    save_pkl(inst_pth, rgns)
    return True


@instrd_registor('pkl')
def _maskrd_pkl(inst_pth, boxes, **kwargs):
    rgns = load_pkl(inst_pth)
    insts = InstsLabel(img_size=boxes.img_size)
    for rgn, box in zip(rgns, boxes):
        inst = InstItem(border=box.border, category=box.category, rgn=rgn, **box)
        insts.append(inst)
    return insts


# </editor-fold>


def _msgfmtr_init_apply(root, set_name, folders_src, folders_dst, prefix='Apply'):
    folders_dst = [f for f in folders_dst if f is not None]
    folders_src = [f for f in folders_src if f is not None]
    msg_dst = ' [ ' + ' | '.join(folders_dst) + ' ] '
    msg_src = ' [ ' + ' | '.join(folders_src) + ' ] '
    return prefix + ' ' + root + ' ( ' + str(set_name) + ' ) ' + msg_src + '->' + msg_dst


def _msgfmtr_init_create(root, set_name, folders, prefix='Create'):
    folders = [f for f in folders if f is not None]
    msg = ' [ ' + ' | '.join(folders) + ' ] '
    return prefix + ' ' + root + ' ( ' + str(set_name) + ' ) ' + msg


def _msgfmtr_end(prefix='Apply'):
    return prefix + ' completed'


# <editor-fold desc='数据集转化'>

# 确保文件夹存在
def _ensure_folders(root, folders):
    dirs = []
    for folder in folders:
        if folder is None:
            root_dir = None
        else:
            root_dir = os.path.join(root, folder)
            ensure_folder_pth(root_dir)
        dirs.append(root_dir)
    return dirs


# 逐图像地生成分割数据集
def vocD2vocS(dataset, colors, root, mask_folder='SegmentationClass', mask_extend='png',
              prefix='Convert Segmentation', broadcast=BROADCAST, ):
    (mask_dir,) = _ensure_folders(root, [mask_folder])
    broadcast(_msgfmtr_init_create(root, dataset.set_name, [mask_folder], prefix=prefix))
    for i, (img, boxes) in MEnumerate(dataset, broadcast=broadcast):
        mask_pth = os.path.join(mask_dir, ensure_extend(boxes.meta, mask_extend))
        segs = SegsLabel.convert(boxes)
        VocSegmentationDataset.create_mask(mask_pth=mask_pth, segs=segs, colors=colors)
    broadcast(_msgfmtr_end(prefix=prefix))
    return True


# 逐图像地生成分割数据集
def vocD2vocI(dataset, colors, root, inst_folder='SegmentationClass', inst_extend='png',
              prefix='Convert Instance', broadcast=BROADCAST, ):
    (inst_dir,) = _ensure_folders(root, [inst_folder])
    broadcast(_msgfmtr_init_create(root, dataset.set_name, [inst_folder], prefix=prefix))
    for i, (img, boxes) in MEnumerate(dataset, broadcast=broadcast):
        inst_pth = os.path.join(inst_dir, ensure_extend(boxes.meta, inst_extend))
        insts = InstsLabel.convert(boxes)
        VocInstanceDataset.create_inst(inst_pth=inst_pth, insts=insts, colors=colors)
    broadcast(_msgfmtr_end(prefix=prefix))
    return True


# 逐图像地生成分割数据集
def datasetI2vocI(dataset, set_name, colors, root, set_folder='ImageSets/Main', fltr=None,
                  inst_folder='SegmentationObject', inst_extend='png', ignore_empty=True,
                  anno_folder='Annotations', anno_extend='xml', img_folder='JPEGImages', img_extend='jpg',
                  prefix='Convert Instance', broadcast=BROADCAST, ):
    folders = [inst_folder, anno_folder, img_folder, set_folder]
    inst_dir, anno_dir, img_dir, set_dir = _ensure_folders(root, folders)
    metas = []
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2img_filt(img, items, fltr=fltr, ignore_empty=ignore_empty)
        for piece, plabel in zip(pieces, plabels):
            piece_pth = os.path.join(img_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            inst_pth = os.path.join(inst_dir, ensure_extend(plabel.meta, inst_extend))
            piece.save(piece_pth)
            VocInstanceDataset.create_anno_inst(anno_pth=anno_pth, inst_pth=inst_pth, colors=colors, insts=plabel, )
            metas.append(plabel.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return True


def datasetD2vocD(dataset, set_name, root, set_folder='ImageSets/Main', fltr=None,
                  ignore_empty=True, anno_folder='Annotations', anno_extend='xml',
                  img_folder='JPEGImages', img_extend='jpg', prefix='Convert Detection', broadcast=BROADCAST, ):
    folders = [anno_folder, img_folder, set_folder]
    anno_dir, img_dir, set_dir = _ensure_folders(root, folders)
    metas = []
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2img_filt(img, items, fltr=fltr, ignore_empty=ignore_empty)
        for piece, plabel in zip(pieces, plabels):
            piece_pth = os.path.join(img_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            piece.save(piece_pth)
            VocDetectionDataset.create_anno(anno_pth=anno_pth, boxes=plabel)
            metas.append(plabel.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return True


def datasetI2vocI_persize(dataset, set_name, colors, root, set_folder='ImageSets/Main', patch_folder='JPEGImages',
                          inst_folder='SegmentationClass', anno_folder='Annotations', ignore_empty=True,
                          with_clip=False, piece_size=(640, 640), over_lap=(100, 100),
                          fltr=None, img_extend='jpg', anno_extend='xml', inst_extend='png',
                          prefix='Crop Instance', broadcast=BROADCAST, ):
    folders = [inst_folder, anno_folder, patch_folder, set_folder]
    inst_dir, anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2piece_persize(img, items, piece_size=piece_size, fltr=fltr, over_lap=over_lap,
                                            ignore_empty=ignore_empty, with_clip=with_clip)
        for piece, plabel in zip(pieces, plabels):
            piece_pth = os.path.join(patch_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            inst_pth = os.path.join(inst_dir, ensure_extend(plabel.meta, inst_extend))
            piece.save(piece_pth)
            VocInstanceDataset.create_anno_inst(anno_pth=anno_pth, inst_pth=inst_pth, colors=colors, insts=plabel, )
            metas.append(plabel.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


def datasetD2vocD_persize(dataset, set_name, root, set_folder='ImageSets/Main', patch_folder='JPEGImages',
                          anno_folder='Annotations', ignore_empty=True, with_clip=False, piece_size=(640, 640),
                          over_lap=(100, 100), fltr=None, img_extend='jpg', anno_extend='xml',
                          prefix='Crop Box', broadcast=BROADCAST, ):
    folders = [anno_folder, patch_folder, set_folder]
    anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    metas = []
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2piece_persize(img, items, piece_size=piece_size, fltr=fltr, over_lap=over_lap,
                                            ignore_empty=ignore_empty, with_clip=with_clip)
        for piece, plabel in zip(pieces, plabels):
            piece_pth = os.path.join(patch_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            piece.save(piece_pth)
            VocDetectionDataset.create_anno(anno_pth, boxes=plabel, )
            metas.append(plabel.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


# 逐检测框地生成分割数据集
def datasetI2vocI_perbox(dataset, set_name, colors, root, set_folder='ImageSets/Main', patch_folder='JPEGImages',
                         inst_folder='SegmentationClass', anno_folder='Annotations', expend_ratio=1.2, with_clip=False,
                         as_square=False, fltr=None, extractor=None, ignore_empty=True,
                         img_extend='jpg', anno_extend='xml', inst_extend='png', meta_encoder=None,
                         prefix='Crop Instance', broadcast=BROADCAST, ):
    folders = [inst_folder, anno_folder, patch_folder, set_folder]
    inst_dir, anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    metas = set()
    for i, (img, insts) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2piece_perbox(
            img, insts, expend_ratio=expend_ratio, with_clip=with_clip, as_square=as_square,
            extractor=extractor, fltr=fltr, ignore_empty=ignore_empty,
            meta_encoder=meta_encoder)
        for piece, pinsts in zip(pieces, plabels):
            patch_pth = os.path.join(patch_dir, ensure_extend(pinsts.meta, img_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(pinsts.meta, anno_extend))
            inst_pth = os.path.join(inst_dir, ensure_extend(pinsts.meta, inst_extend))
            piece.save(patch_pth)
            VocInstanceDataset.create_anno_inst(anno_pth=anno_pth_new, inst_pth=inst_pth, colors=colors, insts=pinsts, )
            metas.add(pinsts.meta)
    metas = list(metas)
    save_txt(file_pth=os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


# 根据标注文件随机裁剪背景
def datasetD2vocI_background(dataset, set_name, root, set_folder='ImageSets/Main', patch_folder='JPEGImages',
                             inst_folder='SegmentationClass', anno_folder='Annotations', min_size=0, max_size=16,
                             repeat_num=1.0, fltr=None, img_extend='jpg', anno_extend='xml',
                             inst_extend='png', prefix='Crop Background', broadcast=BROADCAST, ):
    folders = [inst_folder, anno_folder, patch_folder, set_folder]
    inst_dir, anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2background(
            img, items, min_size=min_size, max_size=max_size, repeat_num=repeat_num, fltr=fltr)
        for piece, pinsts in zip(pieces, plabels):
            patch_pth = os.path.join(patch_dir, ensure_extend(pinsts.meta, img_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(pinsts.meta, anno_extend))
            inst_pth = os.path.join(inst_dir, ensure_extend(pinsts.meta, inst_extend))
            piece.save(patch_pth)
            VocInstanceDataset.create_anno_inst(anno_pth=anno_pth_new, inst_pth=inst_pth, colors=[], insts=pinsts, )
            metas.append(pinsts.meta)
    save_txt(file_pth=os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


def datasetD2vocD_background(dataset, set_name, root, set_folder='ImageSets/Main', patch_folder='JPEGImages',
                             anno_folder='Annotations', min_size=0, max_size=16,
                             repeat_num=1.0, fltr=None, img_extend='jpg', anno_extend='xml',
                             prefix='Crop Background', broadcast=BROADCAST, func=None):
    folders = [anno_folder, patch_folder, set_folder]
    anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, (img, items) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2background(
            img, items, min_size=min_size, max_size=max_size, repeat_num=repeat_num, fltr=fltr)
        for piece, plabel in zip(pieces, plabels):
            patch_pth = os.path.join(patch_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            if func is not None:
                piece, plabel = func(piece, plabel)
            img2imgP(piece).save(patch_pth)
            VocDetectionDataset.create_anno(anno_pth=anno_pth_new, boxes=plabel)
            metas.append(plabel.meta)
    save_txt(file_pth=os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


def datasetD2vocD_perbox(dataset, set_name, root, anno_folder='Annotations', patch_folder='JPEGImages',
                         set_folder='ImageSets/Main', expend_ratio=1.2,
                         with_clip=False, as_square=False, fltr=None, extractor=None,
                         ignore_empty=True, img_extend='jpg', anno_extend='xml',
                         prefix='Crop Box', broadcast=BROADCAST, func=None):
    folders = [anno_folder, patch_folder, set_folder]
    anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, (img, boxes) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2piece_perbox(
            img, boxes, expend_ratio=expend_ratio, with_clip=with_clip, as_square=as_square,
            fltr=fltr, extractor=extractor, ignore_empty=ignore_empty, )
        for piece, plabel in zip(pieces, plabels):
            patch_pth = os.path.join(patch_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            if func is not None:
                piece, plabel = func(piece, plabel)
            img2imgP(piece).save(patch_pth)
            VocDetectionDataset.create_anno(anno_pth_new, boxes=plabel)
            metas.append(plabel.meta)
    save_txt(file_pth=os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


def datasetD2vocD_pyramid(dataset, set_name, root, anno_folder='Annotations', patch_folder='JPEGImages',
                          set_folder='ImageSets/Main', with_clip=False, piece_sizes=((640, 640), (320, 320)),
                          over_laps=((100, 100), (50, 50)), fltr=None, extractor=None,
                          ignore_empty=True, img_extend='jpg', anno_extend='xml',
                          prefix='Crop Pyramid', broadcast=BROADCAST, func=None):
    folders = [anno_folder, patch_folder, set_folder]
    anno_dir, patch_dir, set_dir = _ensure_folders(root, folders)
    broadcast(_msgfmtr_init_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, (img, boxes) in MEnumerate(dataset, broadcast=broadcast):
        pieces, plabels = img2piece_pyramid(
            img, boxes, piece_sizes=piece_sizes, with_clip=with_clip, over_laps=over_laps,
            fltr=fltr, extractor=extractor, ignore_empty=ignore_empty, )
        for piece, plabel in zip(pieces, plabels):
            patch_pth = os.path.join(patch_dir, ensure_extend(plabel.meta, img_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
            if func is not None:
                piece, plabel = func(piece, plabel)
            img2imgP(piece).save(patch_pth)
            VocDetectionDataset.create_anno(anno_pth_new, boxes=plabel)
            metas.append(plabel.meta)
    save_txt(file_pth=os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(_msgfmtr_end(prefix=prefix))
    return metas


# </editor-fold>

# <editor-fold desc='检测'>

def _expand_pths(root, folder, metas, extend='xml'):
    return [os.path.join(root, folder, meta + '.' + extend) for meta in metas]


def _load_metas(root, set_name, set_folder='ImageSets/Main', img_folder='JPEGImages', set_extend='txt'):
    if set_name is None or set_name == 'all':
        img_dir = os.path.join(root, img_folder)
        img_names = sorted(listdir_extend(file_dir=img_dir, extends=VocDetectionDataset.IMG_EXTEND))
        metas = [img_name.split('.')[0] for img_name in img_names]
        return metas
    set_pth = os.path.join(root, set_folder, set_name)
    metas = load_txt(set_pth, extend=set_extend)
    metas = [m for m in metas if len(m) > 0]
    return metas


class VocDataset(MDataSet):
    IMG_EXTEND = 'jpg'
    SET_EXTEND = 'txt'

    def __init__(self, root, set_name, set_folder='ImageSets/Main', img_folder='JPEGImages', img_extend=IMG_EXTEND):
        self._root = root
        self._set_name = set_name if isinstance(set_name, str) else 'all'
        # 加载标签
        self.set_folder = set_folder
        self.img_extend = img_extend
        self.img_folder = img_folder
        self._metas = _load_metas(root, set_name, set_folder=set_folder, img_folder=img_folder)

    @property
    def folders(self):
        return [self._img_folder, self._set_folder]

    def ensure_folders(self):
        _ensure_folders(self.root, self.folders)
        return self

    @property
    def metas(self):
        return self._metas

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def set_folder(self):
        return self._set_folder

    @property
    def set_dir(self):
        return os.path.join(self._root, self._set_folder)

    @set_folder.setter
    def set_folder(self, set_folder):
        self._set_folder = set_folder

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def img_dir(self):
        return os.path.join(self.root, self._img_folder)

    @property
    def img_pths(self):
        return _expand_pths(self.root, folder=self.img_folder, metas=self.metas, extend=self.img_extend)

    @img_folder.setter
    def img_folder(self, img_folder):
        self._img_folder = img_folder

    @property
    def imgs(self):
        imgs = [imgwt_pil(img_pth) for img_pth in self.img_pths]
        return imgs

    @staticmethod
    def collect_img_sizes(img_dir, img_extend='jpg', metas=None):
        img_sizes = []
        img_names = listdir_extend(img_dir, extends=img_extend) if metas is None else \
            [ensure_extend(meta, img_extend) for meta in metas]
        for img_name in img_names:
            img_pth = os.path.join(img_dir, img_name)
            img_size = Image.open(img_pth).size
            img_sizes.append(img_size)
        return np.array(img_sizes)

    @staticmethod
    def partition_set(root, metas, split_dict, set_folder='ImageSets/Main'):
        set_names = list(split_dict.keys())
        ratios = list(split_dict.values())
        ratios_cum = np.cumsum(ratios)
        ensure_folder_pth(os.path.join(root, set_folder))
        np.random.shuffle(metas)
        last_ptr = 0
        for set_name, ratio, ratio_cum in zip(set_names, ratios, ratios_cum):
            cur_ptr = int(len(metas) * ratio_cum)
            metas_set = sorted(metas[last_ptr:cur_ptr])
            last_ptr = cur_ptr
            set_pth = os.path.join(root, set_folder, set_name + '.txt')
            save_txt(set_pth, lines=metas_set)
            BROADCAST('Split completed ' + set_name + ' : %d data' % len(metas_set))
        return True

    @staticmethod
    def merge_set(root, set_names, new_name, set_folder='ImageSets/Main'):
        metas = []
        for set_name in set_names:
            set_pth = os.path.join(root, set_folder, set_name + '.txt')
            metas += load_txt(set_pth, extend='txt')
        new_set_pth = os.path.join(root, set_folder, new_name + '.txt')
        save_txt(new_set_pth, lines=metas)
        BROADCAST('Merge completed [ ' + ' '.join(set_names) + ' ] -> [ ' + new_name + ' ] with %d data' % len(metas))
        return True

    @property
    def img_sizes(self):
        return VocDataset.collect_img_sizes(img_dir=self.img_dir, img_extend=self.img_extend, metas=self._metas)

    def partition(self, split_dict, set_folder='ImageSets/Main'):
        VocDataset.partition_set(root=self.root, metas=self._metas, split_dict=split_dict, set_folder=set_folder)
        return self

    def __repr__(self):
        msg = _msgfmtr_init_create(
            root=self.root, folders=self.folders, prefix=self.__class__.__name__, set_name=self.set_name) + '\n'
        img_size_aver = tuple(np.mean(self.img_sizes, axis=0).astype(np.int32))
        msg += 'Length %d ' % len(self) + 'ImgSize ' + str(img_size_aver)
        return msg


class VocDetectionDataset(NameMapper, MDataBuffer, VocDataset, ):
    ANNO_EXTEND = 'xml'
    IMG_EXTEND = VocDataset.IMG_EXTEND
    SET_EXTEND = VocDataset.SET_EXTEND

    def __init__(self, root, set_name, cls_names=None, set_folder='ImageSets/Main', img_folder='JPEGImages',
                 anno_folder='Annotations', anno_extend=ANNO_EXTEND, img_extend=IMG_EXTEND,
                 border_type=None, **kwargs):
        if cls_names is None:
            names = VocDetectionDataset.collect_names(anno_dir=os.path.join(root, anno_folder))
            cls_names = sorted(Counter(names).keys())
        NameMapper.__init__(self, cls_names)

        VocDataset.__init__(self, root=root, set_name=set_name, set_folder=set_folder,
                            img_folder=img_folder, img_extend=img_extend)
        self.border_type = border_type
        self.anno_extend = anno_extend
        self.anno_folder = anno_folder
        # 预加载
        MDataBuffer.__init__(self, **kwargs)

    @property
    def anno_folder(self):
        return self._anno_folder

    @property
    def anno_dir(self):
        return os.path.join(self.root, self._anno_folder)

    @anno_folder.setter
    def anno_folder(self, anno_folder):
        self._anno_folder = anno_folder

    @property
    def anno_pths(self):
        return _expand_pths(self.root, folder=self.anno_folder, metas=self.metas, extend=self.anno_extend)

    # <editor-fold desc='VOC工具'>
    # 获取所有图片大小
    @staticmethod
    def collect_img_sizes(anno_dir, anno_extend='xml', metas=None):
        img_sizes = []
        anno_names = listdir_extend(anno_dir, extends=anno_extend) if metas is None else \
            [ensure_extend(meta, anno_extend) for meta in metas]
        for anno_name in anno_names:
            anno_pth = os.path.join(anno_dir, anno_name)
            root = ET.parse(anno_pth).getroot()
            img_size = _xmlrd_img_size(root)
            img_sizes.append(img_size)
        return np.array(img_sizes)

    # 获取所有标注类名称
    @staticmethod
    def collect_names(anno_dir, anno_extend='xml', metas=None):
        names = []
        anno_names = listdir_extend(anno_dir, extends=anno_extend) if metas is None else \
            [ensure_extend(meta, anno_extend) for meta in metas]
        for anno_name in anno_names:
            anno_pth = os.path.join(anno_dir, anno_name)
            root = ET.parse(anno_pth).getroot()
            for i, obj in enumerate(root):
                if obj.tag not in TAGS_BOXITEM:
                    continue
                names.append(obj.find(TAG_NAME).text)
        return np.array(names)

    # 获取所有图片大小
    @staticmethod
    def collect_sizes(anno_dir, anno_extend='xml', metas=None):
        sizes = []
        anno_names = listdir_extend(anno_dir, extends=anno_extend) if metas is None else \
            [ensure_extend(meta, anno_extend) for meta in metas]
        for anno_name in anno_names:
            anno_pth = os.path.join(anno_dir, anno_name)
            root = ET.parse(anno_pth).getroot()
            boxes_list = xmlrd_items(root, size=(0, 0), name2cind=None, num_cls=1)
            for box in boxes_list:
                xyxy = XYXYBorder.convert(box.border).xyxyN
                sizes.append(xyxy[2:4] - xyxy[:2])
        return np.array(sizes)

    # 解析xml标注
    @staticmethod
    def prase_anno(anno_pth, name2cind=None, num_cls=1, img_size=(256, 256), border_type=None):
        if not os.path.exists(anno_pth):
            meta = os.path.basename(anno_pth).split('.')[0]
            return BoxesLabel(meta=meta, img_size=img_size)
        else:
            root = ET.parse(anno_pth).getroot()
            boxes = _xmlrd_boxes(root, name2cind=name2cind, num_cls=num_cls)

            boxes.meta = os.path.basename(anno_pth).split('.')[0]
            if border_type is not None: boxes.as_border_type(border_type)
        return boxes

    # 创建
    @staticmethod
    def create_anno(anno_pth, boxes):
        root = xmlwt_item(ET.Element(''), boxes)
        pretty_node(root, indent='\t', newline='\n', level=0)
        root = ET.ElementTree(root)
        root.write(anno_pth, encoding='utf-8')
        return root

    # 重命名
    @staticmethod
    def rename_anno_obj(anno_pth, anno_pth_new, rename_dict):
        if not os.path.exists(anno_pth):
            BROADCAST('Anno not exist ' + anno_pth)
            return True
        root = ET.parse(anno_pth).getroot()
        stat_dict = dict([(key, 0) for key in rename_dict.keys()])
        for i, obj in enumerate(root):
            if obj.tag not in TAGS_BOXITEM:
                continue
            name = obj.find(TAG_NAME).text
            if name not in rename_dict.keys():
                continue
            elif rename_dict[name] is None:
                root.remove(obj)
            else:
                name_new = rename_dict[name]
                obj.find(TAG_NAME).text = name_new
            stat_dict[name] += 1
        root_new = ET.ElementTree(root)
        root_new.write(anno_pth_new, encoding='utf-8')
        return stat_dict

    @staticmethod
    def raname_annos_obj(anno_pths, anno_dir, rename_dict):
        BROADCAST('Rename annos to ' + anno_dir)
        stat_dict = dict([(key, 0) for key in rename_dict.keys()])
        ensure_folder_pth(anno_dir)
        for anno_pth in anno_pths:
            anno_pth_new = os.path.join(anno_dir, os.path.basename(anno_pth))
            stat_dict_i = VocDetectionDataset.rename_anno_obj(anno_pth, anno_pth_new, rename_dict)
            for key in stat_dict.keys():
                stat_dict[key] += stat_dict_i[key]
        for key in stat_dict.keys():
            new_name = rename_dict[key]
            if new_name is not None:
                BROADCAST('%30s' % key + ' -> %-30s' % new_name + ' : ' + '%-10d' % stat_dict[key])
            else:
                BROADCAST('%30s' % key + ' remove : ' + '%-10d' % stat_dict[key])
        return stat_dict

    # </editor-fold>

    def __len__(self):
        return len(self._metas)

    def _meta2data(self, meta):
        if self.img_folder is not None:
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
            img = imgwt_pil(img_pth)
            img_size = img.size
        else:
            img, img_size = None, None
        if self.anno_folder is not None:
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            label = VocDetectionDataset.prase_anno(
                anno_pth, name2cind=self.name2cind, num_cls=self.num_cls, img_size=img_size,
                border_type=self.border_type)
        else:
            label = None
        return img, label

    def _index2data(self, index):
        return self._meta2data(self.metas[index])

    @property
    def folders(self):
        return [self._img_folder, self._anno_folder, self._set_folder]

    def __repr__(self):
        msg = super(VocDetectionDataset, self).__repr__() + '\n'
        num_dict = Counter(self.names)
        width = max([len(name) for name in num_dict.keys()])
        msg += '\n'.join([str(i) + ' | ' + name.ljust(width) + ' | %-6d' % num
                          for i, (name, num) in enumerate(num_dict.items())])
        return msg

    # 得到物体尺寸
    @property
    def sizes(self):
        return VocDetectionDataset.collect_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def names(self):
        return VocDetectionDataset.collect_names(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def img_sizes(self):
        return VocDetectionDataset.collect_img_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def labels(self):
        labels = [VocDetectionDataset.prase_anno(
            anno_pth, img_size=None, name2cind=self.name2cind, num_cls=self.num_cls, border_type=self.border_type)
            for anno_pth in self.anno_pths]
        return labels

    @property
    def objnums(self):
        num_dict = Counter(self.names)
        nums = np.zeros(len(self.cls_names))
        for cls_name, num in num_dict.items():
            nums[self.cls_names.index(cls_name)] = num
        return nums

    # 重命名物体
    def rename(self, rename_dict, anno_folder='Annotations2'):
        stat_dict = VocDetectionDataset.raname_annos_obj(
            self.anno_pths, anno_dir=os.path.join(self.root, anno_folder), rename_dict=rename_dict)
        return stat_dict

    def dump(self, labels, anno_folder='Annotations2', with_recover=True, anno_extend='xml',
             prefix='Create', broadcast=BROADCAST):
        folders = [anno_folder]
        anno_dir, = _ensure_folders(self.root, folders)
        broadcast(_msgfmtr_init_create(self.root, self.set_name, folders, prefix=prefix))
        for i, boxes in MEnumerate(labels, broadcast=broadcast):
            anno_pth = os.path.join(anno_dir, ensure_extend(boxes.meta, anno_extend))
            if with_recover:
                boxes.recover()
            VocDetectionDataset.create_anno(anno_pth, boxes=boxes)
        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    def append(self, imgs, labels, anno_folder='Annotations2', img_folder='JPEGImages2', set_name='append',
               with_recover=True, prefix='Append', broadcast=BROADCAST):
        folders = [img_folder, anno_folder, self.set_folder]
        img_dir, anno_dir, set_dir = _ensure_folders(self.root, folders)
        broadcast(_msgfmtr_init_create(self.root, self.set_name, folders, prefix=prefix))
        metas = []
        for i, boxes in MEnumerate(labels, broadcast=broadcast):
            meta = boxes.meta
            anno_pth = os.path.join(anno_dir, meta + '.' + self.anno_extend)
            img_pth = os.path.join(img_dir, meta + '.' + self.img_extend)
            if with_recover:
                boxes.recover()
            imgs[i].save(img_pth)
            VocDetectionDataset.create_anno(anno_pth, boxes=boxes)
            metas.append(meta)
        save_txt(os.path.join(set_dir, set_name + '.txt'), metas)
        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    def update(self, labels_md, anno_folder='Annotations', with_recover=True, anno_extend='xml',
               prefix='Update', broadcast=BROADCAST):
        folders = [anno_folder]
        anno_dir, = _ensure_folders(self.root, folders)
        broadcast(_msgfmtr_init_create(self.root, self.set_name, [anno_folder], prefix=prefix))
        for i, boxes in MEnumerate(labels_md, broadcast=broadcast):
            anno_pth = os.path.join(anno_dir, ensure_extend(boxes.meta, anno_extend))
            if with_recover:
                boxes.recover()
            boxes_ori = VocDetectionDataset.prase_anno(anno_pth)
            for box in boxes:
                ind = box['ind']
                box_ori = boxes_ori[ind]
                box_ori.border = box.border
            VocDetectionDataset.create_anno(anno_pth, boxes=boxes_ori)
        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    # 执行增广生成新标签
    def apply(self, func, img_folder='JPEGImages2', anno_folder='Annotations2', img_extend='jpg', anno_extend='xml'
              , prefix='Apply', broadcast=BROADCAST, ):
        folders_src = [self.img_folder, self.anno_folder]
        folders_dst = [img_folder, anno_folder]
        img_dir, anno_dir = _ensure_folders(self.root, folders_dst)
        broadcast(_msgfmtr_init_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
            img = imgwt_pil(img_pth)
            boxes = VocDetectionDataset.prase_anno(anno_pth, name2cind=self.name2cind, num_cls=self.num_cls,
                                                   img_size=img.size, border_type=self.border_type)
            img_cvt, boxes_cvt = func(img, boxes)
            if img_dir is not None and img_cvt is not None:
                img_pth_new = os.path.join(img_dir, ensure_extend(meta, img_extend))
                img2imgP(img_cvt).save(img_pth_new)
            if anno_dir is not None and boxes_cvt is not None:
                anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
                VocDetectionDataset.create_anno(anno_pth_new, boxes_cvt)
        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    # 执行增广生成新标签
    def copy(self, func, img_folder='JPEGImages2', anno_folder='Annotations2', img_extend='jpg',
             anno_extend='xml', prefix='Copy', broadcast=BROADCAST, fltr=None, ignore_empty=True):
        folders_src = [self.img_folder, self.anno_folder]
        folders_dst = [img_folder, anno_folder]
        img_dir, anno_dir = _ensure_folders(self.root, folders_dst)
        broadcast(_msgfmtr_init_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))

            img = imgwt_pil(img_pth)
            label = VocDetectionDataset.prase_anno(anno_pth, name2cind=self.name2cind, num_cls=self.num_cls,
                                                   img_size=img.size, border_type=self.border_type)
            imgs, labels = img2img_filt(img, label, fltr=fltr, ignore_empty=ignore_empty)
            for img, label in zip(imgs, labels):
                if not func(img, label):
                    continue
                if img_dir is not None:
                    img_pth_new = os.path.join(img_dir, ensure_extend(meta, img_extend))
                    img2imgP(img).save(img_pth_new)
                if anno_dir is not None:
                    anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
                    VocDetectionDataset.create_anno(anno_pth_new, label)
        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    def create_set(self, set_name, fltr=None):
        if fltr is None:
            metas = self.metas
        else:
            metas = []
            for anno_pth in self.anno_pths:
                label = VocDetectionDataset.prase_anno(anno_pth)
                if fltr(label):
                    metas.append(label.meta)
        BROADCAST('Create set [ ' + set_name + ' ] with %d data' % len(metas))
        save_txt(os.path.join(self.set_dir, set_name), metas)
        return metas

    # 使用函数对标签形式进行变换
    def label_apply(self, func, anno_folder='Annotations2', anno_extend='xml', prefix='Apply', broadcast=BROADCAST):
        folders_dst = [anno_folder]
        anno_dir, = _ensure_folders(self.root, folders_dst)
        broadcast(_msgfmtr_init_apply(self.root, self.set_name, [self.anno_folder], folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            boxes = VocDetectionDataset.prase_anno(
                anno_pth, name2cind=self.name2cind, num_cls=self.num_cls, border_type=self.border_type)
            boxes_cvt = func(boxes)
            VocDetectionDataset.create_anno(anno_pth_new, boxes=boxes_cvt)
        broadcast(_msgfmtr_end(prefix=prefix))
        return True


# </editor-fold>


# <editor-fold desc='语义分割'>
class VocSegmentationDataset(ColorMapper, MDataBuffer, VocDataset):
    MASK_EXTEND = 'png'
    IMG_EXTEND = VocDataset.IMG_EXTEND
    SET_EXTEND = VocDataset.SET_EXTEND

    def __init__(self, root, set_name, cls_names=None, colors=None, rgn_type=None,
                 set_folder='ImageSets/Segmentation', img_folder='JPEGImages', mask_folder='SegmentationClass',
                 mask_extend=MASK_EXTEND, img_extend=IMG_EXTEND, **kwargs):
        ColorMapper.__init__(self, cls_names=cls_names, colors=colors)
        VocDataset.__init__(self, root=root, set_name=set_name, set_folder=set_folder,
                            img_folder=img_folder, img_extend=img_extend)
        self.mask_extend = mask_extend
        self.mask_folder = mask_folder
        self.rgn_type = rgn_type
        MDataBuffer.__init__(self, **kwargs)

    @property
    def mask_folder(self):
        return self._mask_folder

    @property
    def mask_dir(self):
        return os.path.join(self.root, self._mask_folder)

    @property
    def mask_pths(self):
        return _expand_pths(self.root, folder=self.mask_folder, metas=self.metas, extend=self.mask_extend)

    @mask_folder.setter
    def mask_folder(self, mask_folder):
        self._mask_folder = mask_folder

    # <editor-fold desc='VOC工具'>
    @staticmethod
    def prase_mask(mask_pth, colors, cind2name=None, num_cls=1, img_size=(256, 256)):
        meta = os.path.basename(mask_pth).split('.')[0]
        if not os.path.exists(mask_pth):
            return SegsLabel(img_size=img_size, meta=meta)
        segs = maskrd(mask_pth, colors=colors, num_cls=num_cls)
        segs.meta = meta
        for seg in segs:
            cind = IndexCategory.convert(seg.category).cindN
            if cind2name is not None:
                seg['name'] = cind2name(cind)
        return segs

    @staticmethod
    def create_mask(mask_pth, segs, colors):
        assert isinstance(segs, SegsLabel) or isinstance(segs, InstsLabel), \
            'fmt err ' + segs.__class__.__name__
        maskwt(mask_pth=mask_pth, items=segs, colors=colors)
        return True

    # </editor-fold>

    def __len__(self):
        return len(self.img_pths)

    def _meta2data(self, meta):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
        mask_pth = os.path.join(self.root, self.mask_folder, ensure_extend(meta, self.mask_extend))
        img = imgwt_pil(img_pth)
        segs = VocSegmentationDataset.prase_mask(
            mask_pth, colors=self.colors, cind2name=self.cind2name, num_cls=self.num_cls, img_size=img.size)
        if self.rgn_type is not None:
            for seg in segs: seg.rgn = self.rgn_type.convert(seg.rgn)
        return img, segs

    def _index2data(self, index):
        return self._meta2data(self.metas[index])

    def dump(self, labels, mask_folder='SegmentationClass'):
        mask_dir = os.path.join(self.root, mask_folder)
        ensure_folder_pth(mask_dir)
        for segs in labels:
            mask_pth = os.path.join(mask_dir, segs.meta + '.' + self.mask_extend)
            VocSegmentationDataset.create_mask(mask_pth=mask_pth, segs=segs, colors=self.colors)
        return True

    @property
    def folders(self):
        return [self._img_folder, self._mask_folder, self._set_folder]

    # 统计物体个数
    def __repr__(self):
        msg = VocDataset.__repr__(self) + '\n'
        num_dict = Counter(self.names)
        width = max([len(name) for name in num_dict.keys()])
        msg += '\n'.join([str(i) + ' | ' + name.ljust(width) + ' | %-6d' % num
                          for i, (name, num) in enumerate(num_dict.items())])
        return msg


# </editor-fold>

# <editor-fold desc='实例分割'>

class VocInstanceDataset(ColorMapper, MDataBuffer, VocDataset):
    INST_EXTEND = 'png'
    ANNO_EXTEND = 'xml'
    IMG_EXTEND = VocDataset.IMG_EXTEND
    SET_EXTEND = VocDataset.SET_EXTEND

    def __init__(self, root, set_name, cls_names=None, colors=None,
                 set_folder='ImageSets/Segmentation', img_folder='JPEGImages', inst_folder='SegmentationClass',
                 anno_folder='Annotations', img_extend=IMG_EXTEND, inst_extend=INST_EXTEND,
                 anno_extend=ANNO_EXTEND, rgn_type=None, border_type=None, **kwargs):
        ColorMapper.__init__(self, cls_names=cls_names, colors=colors)
        VocDataset.__init__(self, root=root, set_name=set_name, set_folder=set_folder, img_folder=img_folder,
                            img_extend=img_extend)
        self.inst_extend = inst_extend
        self.anno_extend = anno_extend
        self.inst_folder = inst_folder
        self.anno_folder = anno_folder
        self.rgn_type = rgn_type
        self.border_type = border_type
        # 其它属性
        self.rgn_type = rgn_type
        MDataBuffer.__init__(self, **kwargs)

    @property
    def anno_folder(self):
        return self._anno_folder

    @property
    def anno_dir(self):
        return os.path.join(self.root, self._anno_folder)

    @anno_folder.setter
    def anno_folder(self, anno_folder):
        self._anno_folder = anno_folder

    @property
    def anno_pths(self):
        return _expand_pths(self.root, folder=self.anno_folder, metas=self.metas, extend=self.anno_extend)

    @property
    def inst_folder(self):
        return self._inst_folder

    @inst_folder.setter
    def inst_folder(self, inst_folder):
        self._inst_folder = inst_folder

    @property
    def inst_pths(self):
        return _expand_pths(self.root, folder=self.inst_folder, metas=self.metas, extend=self.inst_extend)

    @property
    def inst_dir(self):
        return os.path.join(self.root, self._inst_folder)

    @property
    def sizes(self):
        return VocDetectionDataset.collect_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def names(self):
        return VocDetectionDataset.collect_names(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def img_sizes(self):
        return VocDetectionDataset.collect_img_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def labels(self):
        labels = []
        for anno_pth, inst_pth in zip(self.anno_pths, self.inst_pths):
            label = VocInstanceDataset.prase_anno_inst(
                anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=None,
                num_cls=self.num_cls, rgn_type=self.rgn_type, border_type=self.border_type)
            labels.append(label)
        return labels

    @property
    def objnums(self):
        cls_names = np.array(self.cls_names)
        vec = np.sum(cls_names == np.array(self.names)[..., None], axis=0)
        return vec

    # <editor-fold desc='VOC工具'>
    @staticmethod
    def prase_anno_inst(anno_pth, inst_pth, colors, name2cind=None, num_cls=1, img_size=(256, 256), border_type=None,
                        rgn_type=None):
        meta = os.path.basename(inst_pth).split('.')[0]
        boxes = VocDetectionDataset.prase_anno(anno_pth=anno_pth, name2cind=name2cind, num_cls=num_cls,
                                               img_size=img_size, border_type=border_type)
        if not os.path.exists(inst_pth):
            BROADCAST('inst not exist ' + inst_pth)
            return boxes
        insts = instrd(inst_pth, boxes=boxes, colors=colors, rgn_type=rgn_type)
        insts.meta = meta
        insts.kwargs = boxes.kwargs
        for i, box in enumerate(boxes):
            if isinstance(box, BoxRefItem):
                inst = InstRefItem.convert(insts[i])
                inst.border_ref = box.border_ref
                insts[i] = inst
        return insts

    @staticmethod
    def create_inst(inst_pth, colors, insts):
        instwt(insts, inst_pth, colors=colors)
        return True

    @staticmethod
    def create_anno_inst(anno_pth, inst_pth, colors, insts):
        assert isinstance(insts, InstsLabel), 'fmt err ' + insts.__class__.__name__
        VocDetectionDataset.create_anno(anno_pth, insts)
        VocInstanceDataset.create_inst(inst_pth, colors=colors, insts=insts)
        return True

    # </editor-fold>

    def __len__(self):
        return len(self.img_pths)

    @property
    def folders(self):
        return [self._img_folder, self._anno_folder, self._inst_folder, self._set_folder]

    # 统计物体个数
    def __repr__(self):
        msg = VocDataset.__repr__(self) + '\n'
        num_dict = Counter(self.names)
        width = max([len(name) for name in num_dict.keys()])
        msg += '\n'.join([str(i) + ' | ' + name.ljust(width) + ' | %-6d' % num
                          for i, (name, num) in enumerate(num_dict.items())])
        return msg

    def _meta2data(self, meta):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
        anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
        inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
        img = imgwt_pil(img_pth)
        label = VocInstanceDataset.prase_anno_inst(
            anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=img.size, num_cls=self.num_cls,
            rgn_type=self.rgn_type, border_type=self.border_type)
        return img, label

    def _index2data(self, index):
        return self._meta2data(self.metas[index])

    def update(self, labels_md, anno_folder='Annotations', inst_folder='SegmentationClass', with_recover=True,
               anno_extend='xml', inst_extend='png', prefix='Update ', broadcast=BROADCAST):
        folders_dst = [anno_folder, inst_folder]
        anno_dir, inst_dir = _ensure_folders(self.root, folders_dst)
        broadcast(_msgfmtr_init_create(self.root, self.set_name, folders_dst, prefix=prefix))
        for i, label in MEnumerate(labels_md, broadcast=broadcast):
            meta = label.meta
            assert meta is not None
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))

            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            inst_pth_new = os.path.join(inst_dir, ensure_extend(meta, inst_extend))
            assert isinstance(label, InstsLabel), 'fmt err ' + label.__class__.__name__
            if with_recover:
                label.recover()
            insts_ori = VocInstanceDataset.prase_anno_inst(anno_pth, inst_pth, self.colors)
            for inst, inst_ori in zip(label, insts_ori):
                inst_ori.rgn = inst.rgn
                inst_ori.update(inst)

            VocInstanceDataset.create_anno_inst(anno_pth_new, inst_pth_new, self.colors, insts_ori)

        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    def label_apply(self, func, anno_folder='Annotations2', inst_folder='SegmentationClass2', anno_extend='xml',
                    inst_extend='png', prefix='Apply', broadcast=BROADCAST):
        folders_src = [self.anno_folder, self.inst_folder]
        folders_dst = [anno_folder, inst_folder]
        anno_dir, inst_dir = _ensure_folders(self.root, folders_dst)
        broadcast(_msgfmtr_init_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
            insts = VocInstanceDataset.prase_anno_inst(
                anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=None, num_cls=self.num_cls,
                rgn_type=self.rgn_type, border_type=self.border_type)
            insts_cvt = func(insts)

            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            inst_pth_new = os.path.join(inst_dir, ensure_extend(meta, inst_extend))
            VocInstanceDataset.create_anno_inst(anno_pth_new, inst_pth_new, colors=self.colors, insts=insts_cvt, )
        broadcast(_msgfmtr_end(prefix=prefix))
        return True

    def apply(self, func, img_folder='JPEGImages2', anno_folder='Annotations2', inst_folder='SegmentationClass2',
              anno_extend='xml', inst_extend='png', img_extend='jpg', prefix='Apply', broadcast=BROADCAST):
        folders_src = [self.img_folder, self.anno_folder, self.inst_folder]
        folders_dst = [img_folder, anno_folder, inst_folder]
        img_dir, anno_dir, inst_dir = _ensure_folders(self.root, folders_dst)
        broadcast(_msgfmtr_init_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
            inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
            img = self.load_img(img_pth)
            insts = VocInstanceDataset.prase_anno_inst(anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind,
                                                       img_size=None, num_cls=self.num_cls, rgn_type=self.rgn_type,
                                                       border_type=self.border_type)
            img_cvt, insts_cvt = func(img, insts)
            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            inst_pth_new = os.path.join(inst_dir, ensure_extend(meta, inst_extend))
            img_pth_new = os.path.join(img_dir, ensure_extend(meta, img_extend))
            img_cvt.save(img_pth_new)
            VocInstanceDataset.create_anno_inst(anno_pth_new, inst_pth_new, colors=self.colors, insts=insts_cvt, )
        broadcast(_msgfmtr_end(prefix=prefix))
        return True


# </editor-fold>

class ColorGenerator():
    def __init__(self, low=30, high=200):
        self.low = low
        self.high = high

    def __getitem__(self, index):
        return random_color(low=self.low, high=self.high, index=index, unit=False)


class VocCommon(MDataSource):
    SET_FOLDER = 'ImageSets/Main'
    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = 'SegmentationClass'
    INST_FOLDER = 'SegmentationObject'
    COLORS = ColorGenerator(low=30, high=200)

    CLS_NAMES = ('object',)

    BUILDER_MAPPER = {
        TASK_TYPE.DETECTION: VocDetectionDataset,
        TASK_TYPE.SEGMENTATION: VocSegmentationDataset,
        TASK_TYPE.INSTANCE: VocInstanceDataset,
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER,
                 img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER, with_border=True, set_names=None, **kwargs):
        root = self.__class__.get_root() if root is None else root
        if set_names is None:
            set_dir = os.path.join(root, set_folder)
            if os.path.exists(set_dir):
                set_names = [file_name.split('.')[0] for file_name in listdir_extend(set_dir, 'txt')]
            else:
                set_names = tuple()
        MDataSource.__init__(self, root=root, set_names=set_names)
        self.set_folder = set_folder
        self.img_folder = img_folder
        self.anno_folder = anno_folder
        self.mask_folder = mask_folder
        self.inst_folder = inst_folder
        self.cls_names = cls_names
        self.colors = colors
        self.task_type = task_type
        self.with_border = with_border
        self.kwargs = kwargs

    def dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = self.__class__.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names, root=self.root,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = builder(**kwargs_update)
        return dataset


class Voc(VocCommon):
    SET_FOLDER_DET = 'ImageSets/Main'
    SET_FOLDER_SEG = 'ImageSets/Segmentation'
    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = 'SegmentationClass'
    INST_FOLDER = 'SegmentationObject'
    CLS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    COLORS = ((128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 128), (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
              (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128), (64, 192, 128),
              (192, 192, 128), (0, 0, 64), (128, 0, 64), (0, 128, 0), (128, 128, 64),
              (0, 0, 192), (128, 0, 192), (0, 128, 192))
    BORDER_COLOR = (224, 224, 192)
    RAND_COLORS = ColorGenerator(low=30, high=200)

    SUB_2007 = '2007'
    SUB_2012 = '2012'
    SUB_0712 = '0712'
    SUB_NONE = None

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//VOC//',
        PLATFORM_SEV3090: '//home//data-storage//VOC',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/exspace/dataset//VOC2007',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER_DET,
                 img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val'), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)

    def dataset(self, set_name='train', task_type=None, sub=SUB_2007, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = self.__class__.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors,
                             with_border=self.with_border)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        if sub == Voc.SUB_0712:
            ds07 = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2007'),
                           fmt=set_name + '_07_%d', **kwargs_update)
            ds12 = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2012'),
                           fmt=set_name + '_12_%d', **kwargs_update)
            dataset = ConcatDataset([ds07, ds12])
            dataset.num_cls = ds07.num_cls
            dataset.name2cind = ds07.name2cind
            dataset.cind2name = ds07.cind2name
        elif sub == Voc.SUB_2007:
            dataset = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2007'),
                              **kwargs_update)
        elif sub == Voc.SUB_2012:
            dataset = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2012'),
                              **kwargs_update)
        elif sub == Voc.SUB_NONE:
            dataset = builder(root=self.root, **kwargs_update)
        else:
            raise Exception('err sub ' + sub.__class__.__name__)
        return dataset

# if __name__ == '__main__':
#     voc = Voc.SEV_NEW()
#     loader = voc.loader(set_name='test', batch_size=4, num_workers=0, aug_seq=None)
#     imgs, labels = next(iter(loader))
# a = np.array(labels[0])

# if __name__ == '__main__':
#     ds = Voc.SEV_NEW(task_type=TASK_TYPE.INSTANCE, set_folder=Voc.SET_FOLDER_SEG)
#     dataset = ds.dataset('train', sub=Voc.SUB_2012)
#     for img, label in dataset:
#         print(label.meta)
#         pass
