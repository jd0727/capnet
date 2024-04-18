from data.voc import *


def create_set(root, img_folder='AllImages', inst_folder='Segmentations', set_folder='ImageSets/Main'):
    img_dir = os.path.join(root, img_folder)
    inst_dir = os.path.join(root, inst_folder)
    set_dir = os.path.join(root, set_folder)
    ensure_folder_pth(set_dir)
    metas_all = [os.path.splitext(name)[0] for name in os.listdir(img_dir)]
    metas_test = [os.path.splitext(name)[0] for name in os.listdir(inst_dir)]
    metas_train = [name for name in metas_all if name not in metas_test]
    save_txt(os.path.join(set_dir, 'train.txt'), metas_train)
    save_txt(os.path.join(set_dir, 'test.txt'), metas_test)
    return None


def cvt_anno(root, anno_folder_src='AnnotationsOri', anno_folder_dst='Annotations'):
    anno_dir_src = os.path.join(root, anno_folder_src)
    anno_dir_dst = os.path.join(root, anno_folder_dst)
    ensure_folder_pth(anno_dir_dst)
    for anno_name in os.listdir(anno_dir_src):
        print(anno_name)
        anno_pth_src = os.path.join(anno_dir_src, anno_name)
        anno_pth_dst = os.path.join(anno_dir_dst, anno_name)
        root = ET.parse(anno_pth_src).getroot()
        W = int(root.find('Img_SizeWidth').text)
        H = int(root.find('Img_SizeHeight').text)
        img_size = (W, H)
        objs = root.find('HRSC_Objects').findall('HRSC_Object')
        boxes = BoxesLabel(meta=os.path.splitext(anno_name)[0], img_size=img_size)
        for obj in objs:
            truncated = int(obj.find('truncated').text) > 0
            difficult = int(obj.find('difficult').text) > 0
            xmin = float(obj.find('box_xmin').text)
            ymin = float(obj.find('box_ymin').text)
            xmax = float(obj.find('box_xmax').text)
            ymax = float(obj.find('box_ymax').text)
            name = 'ship'
            border = XYXYBorder((xmin, ymin, xmax, ymax), size=img_size)
            box = BoxItem(border=border, category=0, name=name, truncated=truncated, difficult=difficult)
            boxes.append(box)
        VocDetectionDataset.create_anno(anno_pth_dst, boxes)
    return None


def remove(set_pth):
    lines = load_txt(set_pth)
    lines = [line.split('>')[1].split('<')[0] for line in lines]
    save_txt(set_pth, lines)


# if __name__ == '__main__':
#     set_pth = 'D:\Datasets\HRSC\ImageSets\Main//test.txt'
#     inst_dir='D:\Datasets\HRSC//Segmentations'
#     lines = read_txt(set_pth)
#     lines = [line for line in lines if os.path.exists(os.path.join(inst_dir,line+'.png'))]
#     write_txt(set_pth, lines)


# if __name__ == '__main__':
#     root = 'D:\Datasets\HRSC'
#     # create_set(root, img_folder='AllImages', inst_folder='Segmentations', set_folder='ImageSets/Main')
#     cvt_anno(root, anno_folder_src='AnnotationsOri', anno_folder_dst='Annotations')


class HRSC(VocCommon):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//HRSC//',
        PLATFORM_SEV3090: '/home/data-storage/HRSC',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    CLS_NAMES = ('ship',)
    INST_FOLDER = 'SegmentationObject'
    IMG_EXTEND = 'bmp'
    COLORS = Voc.COLORS

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, img_extend=IMG_EXTEND,
                 task_type=TASK_TYPE.DETECTION, img_folder=Voc.IMG_FOLDER, anno_folder=Voc.ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=Voc.SET_FOLDER_DET, **kwargs):
        VocCommon.__init__(self, root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                           img_folder, anno_folder, img_extend=img_extend, **kwargs)


class HRSCObj(VocCommon):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '//ses-data//JD//HRSCObj',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }
    COLORS = Voc.COLORS
    CLS_NAMES = ('ship',)
    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    ANNO_FOLDER = 'PatchAnnotations'
    INST_FOLDER = 'PatchInstance'
    IMG_EXTEND = 'bmp'

    def __init__(self, root, cls_names=CLS_NAMES, colors=COLORS,
                 task_type=TASK_TYPE.DETECTION, img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, **kwargs):
        VocCommon.__init__(self, root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                           img_folder, anno_folder, **kwargs)

# if __name__ == '__main__':
#     ds = HRSC.DES(task_type=TASK_TYPE.INSTANCE)
#     dataset = ds.dataset('test')
#     inst_dir='D:\Datasets\HRSC\Seg2'
#     for img,label in dataset:
#         print(label.meta)
#         VocInstanceDataset.create_inst(os.path.join(inst_dir,label.meta+'.png'),colors=Voc.COLORS,insts=label)
