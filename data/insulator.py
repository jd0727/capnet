from data.folder import FolderClassificationDataset
from data.voc import *
from utils.frame import *


class InsulatorC(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorC//',
        PLATFORM_SEV3090: '/ses-data/JD//InsulatorC//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//InsulatorC//',
        PLATFORM_BOARD: ''
    }

    ROOT_RAWC = '//home//data-storage//JD//RawC//unknown//'
    ROOT_BKGD = '//ses-img//JD//Bkgd//'

    def __init__(self, root=None, resample=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=('train', 'test'))
        self.resample = resample
        self.kwargs = kwargs

    def dataset(self, set_name, **kwargs):
        dataset = FolderClassificationDataset(root=os.path.join(self.root, set_name), resample=self.resample,
                                              cls_names=('abnormal', 'normal'), **self.kwargs)
        return dataset


class InsulatorD(VocCommon):
    CLS_NAMES = ('insulator_normal', 'insulator_blast', 'insulator_comp', 'cap', 'cap_missing',)
    CLS_NAMES_BORDER = ('insulator',)
    CLS_NAMES_MERGE = ('insulator_glass', 'insulator_comp',)
    COLORS = Voc.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VocCommon.MASK_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorD//',
        PLATFORM_SEV3090: '//home//data-storage//JD//InsulatorD',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/InsulatorD',
        PLATFORM_BOARD: '/home/jd/data/DataSets/InsulatorD'
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class InsulatorD2(VocCommon):
    CLS_NAMES = ('insulator_normal', 'insulator_blast', 'insulator_comp', 'metal_normal', 'metal_rust',
                 'clamp_normal', 'clamp_rust')
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '//home//data-storage//JD//InsulatorD2',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VocCommon.MASK_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER
    COLORS = VocCommon.COLORS

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class InsulatorDI(VocCommon):
    CLS_NAMES = ('insulator_normal', 'insulator_blast')

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VocCommon.MASK_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER
    COLORS = VocCommon.COLORS

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D:\Datasets\InsulatorDI',
        PLATFORM_DESTOPLAB: 'D:\Datasets\InsulatorDI',
        PLATFORM_SEV3090: '//home//data-storage//JD//InsulatorDI',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class InsulatorObj(VocCommon):
    COLORS = Voc.COLORS
    CLS_NAMES = ('insulator_glass',)
    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    ANNO_FOLDER = 'PatchAnnotations'
    INST_FOLDER = 'PatchInstance'
    MASK_FOLDER = 'PatchMask'

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorObj/',
        PLATFORM_SEV3090: '//ses-data//JD//InsulatorObj',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class DistriNetworkDevice(VocCommon):
    CLS_NAMES = (
        'bgxj', 'bl_jyz', 'bxxs_fhjyz', 'byq', 'dls_rdq',
        'fh_jyz', 'flydjdh', 'fzc', 'gyglkg', 'gyzkdlq',
        'gz_zsq', 'hd_jyz', 'lj_jyz', 'nz_xj', 'qnq',
        'qth', 'qxxj', 'uxgh', 'wtgb', 'xc_xj',
        'xslb_jyz', 'xsp_jyz', 'zhenshi_jyz', 'zhushi_jyz', 'zjgb')
    COLORS = VocCommon.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VocCommon.MASK_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval',), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class DistriNetworkDefect(VocCommon):
    CLS_NAMES = (
        'bxxztl', 'bzxbgf', 'cpps', 'cpqx', 'dxwbz',
        'flgflxs', 'jyxjycps_jyhtsh', 'jyzqs', 'jyztl', 'ljjjqtxs',
        'tdsh', 'tk', 'wh_fdhj', 'yw_nc', 'zx_hxlw')
    CLS_NAMES_ZN = (
        '保险销子脱落', '绑扎线不规范', '瓷瓶破损', '瓷瓶倾斜', '导线未绑扎',
        '法兰杆法兰锈蚀', '绝缘层破损', '绝缘罩缺失', '绝缘罩脱落', '连接金具球头锈蚀',
        '塔顶损坏', '脱扣', '污秽或放电痕迹', '异物或鸟巢', '纵横向裂纹',
    )
    COLORS = Voc.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VocCommon.MASK_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval',), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class DistriNetworkMix(VocCommon):
    CLS_NAMES = (
        'bgxj', 'bl_jyz', 'bxxs_fhjyz', 'byq', 'dls_rdq',
        'fh_jyz', 'flydjdh', 'fzc', 'gyglkg', 'gyzkdlq',
        'gz_zsq', 'hd_jyz', 'lj_jyz', 'nz_xj', 'qnq',
        'qth', 'qxxj', 'uxgh', 'wtgb', 'xc_xj',
        'xslb_jyz', 'xsp_jyz', 'zhenshi_jyz', 'zhushi_jyz', 'zjgb',
        'bxxztl', 'bzxbgf', 'cpps', 'cpqx', 'dxwbz',
        'flgflxs', 'jyxjycps_jyhtsh', 'jyzqs', 'jyztl', 'ljjjqtxs',
        'tdsh', 'tk', 'wh_fdhj', 'yw_nc', 'zx_hxlw')
    COLORS = Voc.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VocCommon.MASK_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval',), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


# if __name__ == '__main__':
#     ds = InsulatorD2()
#     print(ds.dataset(set_name='train'))
#     print(ds.dataset(set_name='test'))
# loader = ds.loader(set_name='test', batch_size=4, num_workers=0, aug_seq=None)
# imgs, labels = next(iter(loader))

if __name__ == '__main__':
    ds = InsulatorD(anno_folder='AnnotationsCap')
    print(ds.dataset('val'))

# 拷贝val数据集所有图像
# if __name__ == '__main__':
#     ds = InsulatorD()
#     dataset = ds.dataset('val')
#     save_dir = os.path.join(ds.root, 'buff')
#     for img_pth in dataset.img_pths:
#         shutil.copy(img_pth, os.path.join(save_dir, os.path.basename(img_pth)))

# 拷贝blast数据集所有图像和标注
# if __name__ == '__main__':
#     ds = InsulatorD()
#     save_dir = ensure_folder_pth(os.path.join(ds.root, 'buff'))
#
#     for set_name in ['train', 'test', 'val']:
#         dataset = ds.dataset(set_name)
#         for anno_pth, img_pth in zip(dataset.anno_pths, dataset.img_pths):
#             label = VocDetectionDataset.prase_anno(anno_pth)
#             if any([item['name'] == 'insulator_blast' for item in label]):
#                 print(img_pth)
#                 shutil.copy(img_pth, os.path.join(save_dir, os.path.basename(img_pth)))
#                 shutil.copy(anno_pth, os.path.join(save_dir, os.path.basename(anno_pth)))
