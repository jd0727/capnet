import os

from data.coco import CoCo
from data.voc import Voc, VocCommon

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils import *


class ISAID(CoCo):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAID//',
        PLATFORM_SEV3090: '//home//data-storage//ISAID',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER_FMT = CoCo.IMG_FOLDER_FMT
    JSON_NAME_FMT = CoCo.JSON_NAME_FMT
    JSON_FOLDER = CoCo.JSON_FOLDER

    CLS_NAMES = ('storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'plane', 'ship',
                 'Swimming_pool', 'Harbor', 'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field',
                 'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')

    CIND2NAME_REMAPPER = None

    def __init__(self, root=None, json_name_fmt=JSON_NAME_FMT, img_folder_fmt=IMG_FOLDER_FMT, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=None,
                 set_names=None, **kwargs):
        CoCo.__init__(self, root=root, json_name_fmt=json_name_fmt, img_folder_fmt=img_folder_fmt,
                      json_folder=json_folder, cind2name_remapper=cind2name_remapper, task_type=task_type,
                      cls_names=cls_names, set_names=set_names, **kwargs)


class ISAIDPatch(CoCo):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAID//',
        PLATFORM_SEV3090: '//home//data-storage//ISAID',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER_FMT = 'patches_%s'
    JSON_NAME_FMT = 'instances_%s'
    JSON_FOLDER = 'annotation_ptch'

    CLS_NAMES = ISAID.CLS_NAMES

    CIND2NAME_REMAPPER_DICT = {
        9: 'Small_Vehicle', 8: 'Large_Vehicle', 14: 'plane', 2: 'storage_tank', 1: 'ship',
        11: 'Swimming_pool', 15: 'Harbor', 4: 'tennis_court', 6: 'Ground_Track_Field', 13: 'Soccer_ball_field',
        3: 'baseball_diamond', 7: 'Bridge', 5: 'basketball_court', 12: 'Roundabout', 10: 'Helicopter'}
    NAME2CIND_REMAPPER_DICT = dict([(name, cind) for cind, name in CIND2NAME_REMAPPER_DICT.items()])
    CIND2NAME_REMAPPER = lambda cind: ISAIDPatch.CIND2NAME_REMAPPER_DICT[cind]
    NAME2CIND_REMAPPER = lambda cind: ISAIDPatch.NAME2CIND_REMAPPER_DICT[cind]

    def __init__(self, root=None, json_name_fmt=JSON_NAME_FMT, img_folder_fmt=IMG_FOLDER_FMT, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=CIND2NAME_REMAPPER,
                 set_names=None, **kwargs):
        CoCo.__init__(self, root=root, json_name_fmt=json_name_fmt, img_folder_fmt=img_folder_fmt,
                      json_folder=json_folder, cind2name_remapper=cind2name_remapper, task_type=task_type,
                      cls_names=cls_names, set_names=set_names, **kwargs)


class ISAIDObj(VocCommon):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAIDObj//',
        PLATFORM_SEV3090: '//ses-data//JD//ISAIDObj',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }
    ROOT_SEV_NEW1 = '//ses-data//JD//ISAIDObj1'
    ROOT_SEV_NEW2 = '//ses-data//JD//ISAIDObj2'
    ROOT_SEV_NEW3 = '//ses-data//JD//ISAIDObj3'

    CLS_NAMES = ISAID.CLS_NAMES
    CLS_NAMES1 = ('Small_Vehicle',)
    CLS_NAMES2 = ('Large_Vehicle', 'ship')
    CLS_NAMES3 = ('storage_tank', 'plane', 'Swimming_pool', 'Harbor', 'tennis_court',
                  'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond',
                  'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')
    DYAM_MASK = [1, 0, 1, 0, 1,
                 1, 1, 1,
                 0, 1, 1, 0]

    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    COLORS = Voc.COLORS

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS,
                 task_type=TASK_TYPE.INSTANCE, img_folder=IMG_FOLDER, anno_folder=Voc.ANNO_FOLDER,
                 mask_folder=Voc.MASK_FOLDER, inst_folder=Voc.INST_FOLDER, set_folder=SET_FOLDER, **kwargs):
        VocCommon.__init__(self, root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                           img_folder, anno_folder, **kwargs)


class ISAIDPart(CoCo):
    CLS_NAMES = ISAID.CLS_NAMES
    CIND2NAME_REMAPPER = ISAID.CIND2NAME_REMAPPER

    IMG_FOLDER_FMT = ISAID.IMG_FOLDER_FMT
    JSON_NAME_FMT = ISAID.JSON_NAME_FMT
    JSON_FOLDER = ISAID.JSON_FOLDER

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAIDPart//',
        PLATFORM_SEV3090: '//home//data-storage//ISAIDPart',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, json_name_fmt=JSON_NAME_FMT, img_folder_fmt=IMG_FOLDER_FMT, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=CIND2NAME_REMAPPER,
                 set_names=None, **kwargs):
        CoCo.__init__(self, root=root, json_name_fmt=json_name_fmt, img_folder_fmt=img_folder_fmt,
                      json_folder=json_folder, cind2name_remapper=None, task_type=task_type, cls_names=cls_names,
                      set_names=set_names, **kwargs)


if __name__ == '__main__':
    ds_voc = ISAIDObj()
