from data.voc import *


# <editor-fold desc='SFID标注格式转化'>
def create_set(img_dir_src, root, set_name, set_folder='ImageSets/Main'):
    set_dir = os.path.join(root, set_folder)
    ensure_folder_pth(set_dir)
    metas_all = [os.path.splitext(name)[0] for name in os.listdir(img_dir_src)]
    save_txt(os.path.join(set_dir, set_name + '.txt'), metas_all)
    return None


def cvt_data(anno_dir_src, img_dir_src,
             root_dst, anno_folder_dst='Annotations', img_folder_dst='JPEGImages'):
    anno_dir_dst = os.path.join(root_dst, anno_folder_dst)
    img_dir_dst = os.path.join(root_dst, img_folder_dst)
    ensure_folder_pth(anno_dir_dst)
    ensure_folder_pth(img_dir_dst)
    for anno_name in os.listdir(anno_dir_src):
        print(anno_name)
        anno_pth_src = os.path.join(anno_dir_src, anno_name)
        anno_pth_dst = os.path.join(anno_dir_dst, anno_name.replace('.txt', '.xml'))
        img_name = anno_name.replace('.txt', '.jpg')
        img_pth_src = os.path.join(img_dir_src, img_name)
        img_pth_dst = os.path.join(img_dir_dst, img_name)
        lines = load_txt(anno_pth_src)
        img = Image.open(img_pth_src)
        img_size = img.size
        img.save(img_pth_dst)
        boxes = BoxesLabel(meta=os.path.splitext(anno_name)[0], img_size=img_size)
        for line in lines:
            pieces = line.split(' ')
            cind = int(pieces[0])
            name = ('insulator', 'broken_cap')[cind]
            xyxy = xywhN2xyxyN(np.array([float(v) for v in pieces[1:5]]))
            scaler = np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
            border = XYXYBorder(xyxy * scaler, size=img_size)
            box = BoxItem(border=border, category=0, name=name, truncated=False, difficult=False)
            boxes.append(box)
        VocDetectionDataset.create_anno(anno_pth_dst, boxes)
    return None


# if __name__ == '__main__':
#     root = 'D:\Datasets\SFID'
#     for set_name in ['train', 'test']:
#         anno_dir_src = os.path.join(root, 'labels', set_name)
#         img_dir_src = os.path.join(root, 'images', set_name)
#         create_set(img_dir_src, root, set_name, set_folder='ImageSets/Main')
#         cvt_data(anno_dir_src, img_dir_src,
#                  root_dst=root, anno_folder_dst='Annotations', img_folder_dst='JPEGImages')


# </editor-fold>

# <editor-fold desc='CPLID标注格式转化'>

def merge_labels(anno_dir_insu, anno_dir_dft, anno_dir_mrged):
    file_names = os.listdir(anno_dir_insu)
    ensure_folder_pth(anno_dir_mrged)
    for file_name in file_names:
        anno_pth_insu = os.path.join(anno_dir_insu, file_name)
        anno_pth_dft = os.path.join(anno_dir_dft, file_name)
        anno_pth_mrged = os.path.join(anno_dir_mrged, file_name)
        lb_insu = VocDetectionDataset.prase_anno(anno_pth_insu)
        lb_dft = VocDetectionDataset.prase_anno(anno_pth_dft)
        lb_mrged = lb_insu + lb_dft
        VocDetectionDataset.create_anno(anno_pth_mrged, lb_mrged)


# if __name__ == '__main__':
#     root = '/home/data-storage/JD/CPLID'
#     anno_dir = os.path.join(root, 'Annotations')
#     img_dir = os.path.join(root, 'JPEGImages')
# merge_labels(anno_dir_insu=os.path.join(root, 'Defective_Insulators/labels/insulator'),
#              anno_dir_dft=os.path.join(root, 'Defective_Insulators/labels/defect'),
#              anno_dir_mrged=os.path.join(root, 'Defective_Insulators/labels/merged'))

# shutil.copytree(src=os.path.join(root, 'Normal_Insulators/images'), dst=img_dir,dirs_exist_ok=True)
# shutil.copytree(src=os.path.join(root, 'Defective_Insulators/images'), dst=img_dir,dirs_exist_ok=True)
#
# shutil.copytree(src=os.path.join(root, 'Normal_Insulators/labels'), dst=anno_dir,dirs_exist_ok=True)
# shutil.copytree(src=os.path.join(root, 'Defective_Insulators/labels/merged'), dst=anno_dir,dirs_exist_ok=True)
# metas = [fn.split('.')[0] for fn in os.listdir(img_dir)]
# VocDetectionDataset.partition_set(root=root, split_dict={'train': 0.7, 'test': 0.3}, metas=metas)


# </editor-fold>


class SFID(VocCommon):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//SFID//',
        PLATFORM_SEV3090: '//home//data-storage//SFID',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    CLS_NAMES = ('insulator', 'broken_cap')

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=Voc.COLORS,
                 task_type=TASK_TYPE.DETECTION, img_folder=Voc.IMG_FOLDER, anno_folder=Voc.ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=Voc.INST_FOLDER, set_folder=Voc.SET_FOLDER_DET, **kwargs):
        VocCommon.__init__(self, root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                           img_folder, anno_folder, **kwargs)


class CPLID(VocCommon):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//CPLID//',
        PLATFORM_SEV3090: '/home/data-storage/JD/CPLID',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    CLS_NAMES = ('insulator', 'defect')

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=Voc.COLORS,
                 task_type=TASK_TYPE.DETECTION, img_folder=Voc.IMG_FOLDER, anno_folder=Voc.ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=Voc.INST_FOLDER, set_folder=Voc.SET_FOLDER_DET, **kwargs):
        VocCommon.__init__(self, root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                           img_folder, anno_folder, **kwargs)

# if __name__ == '__main__':
#     ds = CPLID()
#     dataset = ds.dataset('all')
#     dataset.partition_set()
