from data.voc import Voc, VocCommon, VocInstanceDataset, VocDetectionDataset, VocSegmentationDataset
from utils.frame import *


def rand_sect(img, ptch_size=(80, 80), prori=0.1, sxy=20, srgb=40, num_infer=10, kernel_size=10):
    h, w = img.shape[:2]
    xyxy = xyxyN_samp_size(np.array([0, 0, w, h]), ptch_size)
    maskNb = xyxyN2maskNb(np.array(xyxy), size=(w, h))
    maskN = maskNb.astype(np.float32)[..., None] * (1 - prori * 2) + prori
    maskN_ext = np.concatenate([maskN, 1 - maskN], axis=2)
    maskN_ext = masksN_crf(imgN=img, masksN=maskN_ext, sxy=sxy, srgb=srgb, num_infer=num_infer)
    maskN = maskN_ext[..., 0]
    kernel = np.ones(shape=(kernel_size, kernel_size))
    maskN = cv2.dilate(maskN, kernel)
    maskN = cv2.erode(maskN, kernel)
    maskNb = (maskN > np.max(maskN) / 2)
    return maskNb


def rand_paste(img, maskNb):
    xyxy_dst = RefValRegion._maskNb2xyxyN(maskNb)
    h, w = img.shape[:2]
    xyxy_src = xyxyN_samp_size(np.array([0, 0, w, h]), xyxy_dst[2:4] - xyxy_dst[:2])
    mskr_ptch = maskNb[xyxy_dst[1]:xyxy_dst[3], xyxy_dst[0]:xyxy_dst[2]]
    ptch = img[xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
    img[xyxy_dst[1]:xyxy_dst[3], xyxy_dst[0]:xyxy_dst[2]] = \
        np.where(mskr_ptch[..., None], ptch, img[xyxy_dst[1]:xyxy_dst[3], xyxy_dst[0]:xyxy_dst[2]])
    return img


class VocUpsvDataset(VocInstanceDataset):

    def _meta2data(self, meta):
        img, label = super(VocUpsvDataset, self)._meta2data(meta)
        img = np.array(img)
        maskNb = rand_sect(img, ptch_size=(0.3, 0.3), prori=0.1, sxy=20, srgb=40, num_infer=10, kernel_size=10)
        if np.sum(maskNb) > 100:
            xyxy_dst = RefValRegion._maskNb2xyxyN(maskNb)
            img = rand_paste(img, maskNb)
            rgn = AbsBoolRegion(maskNb_abs=maskNb)
            inst = InstItem(category=1, border=XYXYBorder(xyxy_dst, size=label.img_size), rgn=rgn, name='cut')
            label.append(inst)
        return img, label


class InsulatorUpsv(VocCommon):
    COLORS = Voc.COLORS
    CLS_NAMES = ('insulator_normal', 'insulator_blast', 'cap', 'insulator_comp', 'cap_missing')
    IMG_FOLDER = VocCommon.IMG_FOLDER
    SET_FOLDER = VocCommon.SET_FOLDER
    ANNO_FOLDER = VocCommon.ANNO_FOLDER
    INST_FOLDER = VocCommon.INST_FOLDER
    MASK_FOLDER = VocCommon.MASK_FOLDER

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorUpsv',
        PLATFORM_SEV3090: '//home//data-storage//JD//InsulatorUpsv',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    BUILDER_MAPPER = {
        TASK_TYPE.DETECTION: VocDetectionDataset,
        TASK_TYPE.SEGMENTATION: VocSegmentationDataset,
        TASK_TYPE.INSTANCE: VocUpsvDataset,
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VocCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)
