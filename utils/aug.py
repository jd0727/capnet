# import imgaug as ia
# import imgaug.augmenters as iaa

import albumentations as A

from .file import _pair
from .ropr import *
from .visual import *


# random.seed(10)

# <editor-fold desc='基础标签变换'>

# 混合类别标签
def blend_cates(cate0, cate1, mix_rate=0.5):
    oc0 = OneHotCategory.convert(cate0.category)
    oc1 = OneHotCategory.convert(cate1.category)
    chot = (1 - mix_rate) * oc0.chotN + mix_rate * oc1.chotN
    cate = copy.deepcopy(cate0)
    cate.category = OneHotCategory(chotN=chot)
    return cate


# 混合检测类标签
def blend_cate_contss(cate_conts0, cate_conts1, mix_rate=0.5):
    cate_conts0 = copy.deepcopy(cate_conts0)
    cate_conts1 = copy.deepcopy(cate_conts1)
    for cate_cont in cate_conts0:
        cate_cont.category.conf_scale(1 - mix_rate)
    for cate_cont in cate_conts1:
        cate_cont.category.conf_scale(mix_rate)
        cate_conts0.append(cate_cont)
    return cate_conts0


def imgP_affine(imgP, scale=1.0, angle=0.0, shear=0.0, resample=Image.BICUBIC):
    img_size = np.array(imgP.size)
    img_size_scled = (img_size * scale).astype(np.int32)
    A = np.array([[np.cos(angle + shear), np.sin(angle + shear)],
                  [-np.sin(angle - shear), np.cos(angle - shear)]]) * scale
    Ai = np.linalg.inv(A)
    bi = img_size / 2 - Ai @ img_size_scled / 2
    data = [Ai[0, 0], Ai[0, 1], bi[0], Ai[1, 0], Ai[1, 1], bi[1]]
    imgP = imgP.transform(size=tuple(img_size_scled), data=data,
                          method=Image.AFFINE, resample=resample, )
    return imgP


# </editor-fold>


# <editor-fold desc='基本功能'>


class DataTransform(DCTExtractable):

    @abstractmethod
    def trans_datas(self, imgs, labels):
        pass

    @abstractmethod
    def trans_data(self, img, label):
        pass


class GeneralTransform(DataTransform):

    def trans_data(self, img, label):
        imgs, labels = self.trans_datas([img], [label])
        img, label = imgs[0], labels[0]
        return img, label


class IndependentTransform(DataTransform):

    def trans_datas(self, imgs, labels):
        imgs_aug, labels_aug = [], []
        for img, label in zip(imgs, labels):
            img_aug, label_aug = self.trans_data(img, label)
            imgs_aug.append(img_aug)
            labels_aug.append(label_aug)
        return imgs_aug, labels_aug


class SizedTransform(GeneralTransform):

    @property
    @abstractmethod
    def img_size(self):
        pass

    @img_size.setter
    @abstractmethod
    def img_size(self, img_size):
        pass


# 组合增广
class Compose(list, GeneralTransform, DCTBuildable):

    @classmethod
    def buildfrom_dct(cls, dct):
        dct = [dct2obj(val) for val in dct]
        return dct

    def __init__(self, *item):
        super().__init__(item)

    def extract_dct(self):
        dct = []
        for seq in self:
            name = seq.__class__.__name__
            dct.append((name, seq.extract_dct()))
        return dct

    def trans_datas(self, imgs, labels):
        for seq in self:
            imgs, labels = seq.trans_datas(imgs, labels)
        return imgs, labels


class SizedCompose(Compose, SizedTransform):

    @property
    def img_size(self):
        for seq in self:
            if isinstance(seq, SizedTransform):
                return seq.img_size
        return None

    @img_size.setter
    def img_size(self, img_size):
        for seq in self:
            if isinstance(seq, SizedTransform):
                seq.img_size = img_size


class ReInitSizedTransform(SizedTransform):

    def __init__(self, img_size, **kwargs):
        self.kwargs = kwargs
        self._img_size = img_size
        self.transform = self._build_transform(img_size, **self.kwargs)

    def extract_dct(self):
        return dict(kwargs=self.kwargs, img_size=self.img_size, transform=self.transform.extract_dct())

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        if not self._img_size == img_size:
            self._img_size = img_size
            self.transform = self._build_transform(img_size, **self.kwargs)

    def trans_datas(self, imgs, labels):
        imgs, labels = self.transform(imgs, labels)
        return imgs, labels

    @abstractmethod
    def _build_transform(self, img_size, **kwargs):
        pass


# 标准tnsor输出
class ToTensor(IndependentTransform, InitDCTExtractable):

    def __init__(self, concat=True):
        self.concat = concat

    def trans_data(self, img, label):
        return img2imgT(img), label

    def trans_datas(self, imgs, labels):
        imgs, labels = super(ToTensor, self).trans_datas(imgs, labels)
        if self.concat:
            imgs = torch.cat(imgs, dim=0)
        return imgs, labels


class ItemsFiltMeasure(IndependentTransform, InitDCTExtractable):

    def __init__(self, thres=1, with_clip=True):
        self.thres = thres
        self.with_clip = with_clip

    def trans_data(self, img, label):
        if isinstance(label, ImageItemsLabel):
            if self.with_clip:
                label.clip_(xyxyN_rgn=np.array([0, 0, label.img_size[0], label.img_size[1]]))
            label.filt_measure_(thres=self.thres)
        return img, label


class ItemsFiltCategory(IndependentTransform, InitDCTExtractable):

    def __init__(self, cinds=None):
        self.cinds = cinds

    def trans_data(self, img, label):
        if isinstance(label, ImageItemsLabel):
            label.filt_(lambda item: item.category in self.cinds)
        return img, label


class ItemsFilt(IndependentTransform, InitDCTExtractable):

    def __init__(self, fltr=None):
        self.fltr = fltr

    def trans_data(self, img, label):
        if isinstance(label, ImageItemsLabel):
            label.filt_(fltr=self.fltr)
        return img, label


#  <editor-fold desc='Albumentations偏移修改'>


def _trans_data_with_albu(albu_trans, img, label):
    kwargs = dict(image=img2imgN(img))
    if label.num_pnt > 0:
        xlyl = label.extract_xlylN()
        xlyl = np.concatenate([xlyl, np.ones_like(xlyl)], axis=1)  # 补全angle和scale
        kwargs['keypoints'] = xlyl
    masksN = []
    if label.num_bool_chan > 0:
        maskN_enc = label.extract_maskNb_enc(index=1)
        masksN.append(maskN_enc)
    if label.num_chan > 0:
        maskN_val = label.extract_maskN()
        masksN.append(maskN_val)
    if len(masksN) > 0:
        masksN = np.concatenate(masksN, axis=2)
        kwargs['mask'] = masksN

    transformed = albu_trans(**kwargs)
    img_aug = transformed['image']
    img_size = (img_aug.shape[1], img_aug.shape[0])
    if label.num_pnt > 0:
        xlyl_aug = np.array(transformed['keypoints'])[:, :2]
        label.refrom_xlylN(xlyl_aug, img_size)
    offset = 0
    if label.num_bool_chan > 0:
        maskN_enc_aug = transformed['mask'][..., 0:1]
        label.refrom_maskNb_enc(maskN_enc_aug, index=1)
        offset = 1
    if label.num_chan > 0:
        maskN_val_aug = transformed['mask'][..., offset:]
        label.refrom_maskN(maskN_val_aug)
    return img_aug, label

    # def extract_dct(self):
    #     if isinstance(self.transform, A.Compose):
    #         trans_dct = []
    #         for sub_trans in self.transform:
    #             name = sub_trans.__class__.__name__
    #             trans_dct.append((name, InitDCTExtractable.extract_dct_from_init(sub_trans)))
    #     else:
    #         name = self.transform.__class__.__name__
    #         paras = InitDCTExtractable.extract_dct_from_init(self.transform)
    #         trans_dct = (name, paras)
    #     dct = dict(thres=self.thres, transform=trans_dct)
    #     return dct


class AlbuInterface(IndependentTransform):

    def trans_data(self, img, label):
        return _trans_data_with_albu(self, img, label)

    def extract_dct(self):
        dct = InitDCTExtractable.extract_dct_from_init(self, super())
        return dct


class AlbuKeyPointPatch():
    def apply_to_keypoint(self, keypoint, **params):
        keypoint_ofst = (keypoint[0] - 0.5, keypoint[1] - 0.5, keypoint[2], keypoint[3])
        keypoint_trd = super(AlbuKeyPointPatch, self).apply_to_keypoint(
            keypoint_ofst, **params)
        return (keypoint_trd[0] + 0.5, keypoint_trd[1] + 0.5, keypoint_trd[2], keypoint_trd[3])


class A_Compose(A.Compose, AlbuInterface, AlbuKeyPointPatch, SizedTransform):
    def extract_dct(self):
        trans_dct = []
        for sub_trans in self:
            name = sub_trans.__class__.__name__
            trans_dct.append((name, InitDCTExtractable.extract_dct_from_init(sub_trans)))
        return trans_dct

    @property
    def img_size(self):
        for seq in self:
            if isinstance(seq, SizedTransform):
                return seq.img_size
        return None

    @img_size.setter
    def img_size(self, img_size):
        for seq in self:
            if isinstance(seq, SizedTransform):
                seq.img_size = img_size


class A_Flip(A.Flip, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_HorizontalFlip(A.HorizontalFlip, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_VerticalFlip(A.VerticalFlip, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_Affine(A.Affine, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_RandomRotate90(A.RandomRotate90, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_Resize(A.Resize, AlbuInterface, SizedTransform):
    @property
    def img_size(self):
        return (self.width, self.height)

    @img_size.setter
    def img_size(self, img_size):
        self.width, self.height = img_size


class A_PadIfNeeded(A.PadIfNeeded, AlbuInterface, SizedTransform):
    @property
    def img_size(self):
        return (self.min_width, self.min_height)

    @img_size.setter
    def img_size(self, img_size):
        self.min_width, self.min_height = img_size


# </editor-fold>


# 缩放最大边
class LargestMaxSize(SizedTransform, IndependentTransform, InitDCTExtractable):
    def __init__(self, max_size=(256, 256), resample=cv2.INTER_LANCZOS4, thres=10,
                 only_smaller=False, only_larger=False):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres
        self.only_smaller = only_smaller
        self.only_larger = only_larger

    @property
    def img_size(self):
        return self.max_size

    @img_size.setter
    def img_size(self, img_size):
        self.max_size = img_size

    def trans_data(self, img, label):
        imgN = img2imgN(img)
        imgN_scld, ratio = imgN_lmtsize(imgN, max_size=self.max_size, resample=self.resample,
                                        only_smaller=self.only_smaller, only_larger=self.only_larger)
        if not ratio == 1.0:
            label.linear_(scale=np.array([ratio, ratio]), size=img2size(imgN_scld))
        if isinstance(label, ImageItemsLabel):
            label.filt_measure_(thres=self.thres)
        return imgN_scld, label


class LargestMaxSizeWithPadding(IndependentTransform, InitDCTExtractable):
    def __init__(self, max_size=(256, 256), resample=cv2.INTER_LANCZOS4, thres=10):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres

    def trans_data(self, img, label):
        imgN, label = img2imgN(img), label
        imgN_scld, ratio = imgN_lmtsize_pad(imgN, max_size=self.max_size, pad_val=PAD_CVAL, resample=self.resample)
        if not ratio == 1.0:
            label.linear_(scale=np.array([ratio, ratio]), size=img2size(imgN_scld))
            if isinstance(label, ImageItemsLabel):
                label.filt_measure_(thres=self.thres)
        return imgN_scld, label


# 缩放最大边
class CenterRescale(IndependentTransform, InitDCTExtractable):

    def __init__(self, size=(256, 256), expand_ratio=1.0, resample=cv2.INTER_LANCZOS4, thres=10):
        self.size = size
        self.resample = resample
        self.thres = thres
        self.expand_ratio = expand_ratio

    def trans_data(self, img, label):
        imgN = img2imgN(img)
        img_size = np.array((imgN.shape[1], imgN.shape[0]))
        size = np.array(self.size)
        ratio = min(size / img_size) * self.expand_ratio
        bias = size[0] / 2 - ratio * img_size / 2
        A = np.array([[ratio, 0, bias[0]], [0, ratio, bias[1]]]).astype(np.float32)
        imgN = cv2.warpAffine(imgN.astype(np.float32), A, size, flags=self.resample)
        imgN = np.clip(imgN, a_min=0, a_max=255).astype(np.uint8)
        label.linear_(scale=[ratio, ratio], bias=bias, size=tuple(size))
        if isinstance(label, ImageItemsLabel):
            label.filt_measure_(thres=self.thres)

        return imgN, label


class ADD_TYPE:
    APPEND = 'append'
    REPLACE = 'replace'
    COVER = 'cover'
    COVER_SRC = 'cover_src'
    COVER_ORD = 'cover_ord'


from itertools import chain


# 图像混合基类
class MutiMixTransform(GeneralTransform):
    def __init__(self, num_input, num_output, repeat=3.0, add_type=ADD_TYPE.APPEND, diff_input=True, diff_output=True):
        self.num_input = num_input
        self.num_output = num_output
        self.repeat = repeat
        self.add_type = add_type
        self.diff_input = diff_input
        self.diff_output = diff_output

    @abstractmethod
    def mix(self, imgs, labels):
        pass

    @staticmethod
    def samp_inds(num_sample, repeat, num_batch, diff=True):
        num_require = num_batch * repeat
        if diff and num_require <= num_sample:
            inds = np.random.choice(a=num_sample, replace=False, size=num_require)
            indss = np.reshape(inds, (repeat, num_batch))
        elif diff:
            indss = np.stack(
                [np.random.choice(a=num_sample, replace=False, size=num_batch) for _ in range(repeat)], axis=0)
        else:
            indss = np.random.choice(a=num_sample, replace=True, size=(repeat, num_batch))
        return indss

    def trans_datas(self, imgs, labels):
        num_sample = len(imgs)
        if len(imgs) < self.num_input:
            return imgs, labels
        repeat = int(self.repeat * num_sample) if isinstance(self.repeat, float) else self.repeat
        indss_src = MutiMixTransform.samp_inds(num_sample, repeat=repeat, num_batch=self.num_input,
                                               diff=self.diff_input)
        imgss_p = []
        labelss_p = []
        for n in range(repeat):
            imgs_c = [copy.deepcopy(imgs[int(ind)]) for ind in indss_src[n]]
            labels_c = [copy.deepcopy(labels[int(ind)]) for ind in indss_src[n]]
            imgs_p, labels_p = self.mix(imgs_c, labels_c)
            imgss_p.append(imgs_p)
            labelss_p.append(labels_p)
        if self.add_type == ADD_TYPE.REPLACE:
            return list(chain(*imgss_p)), list(chain(*labelss_p))
        elif self.add_type == ADD_TYPE.APPEND:
            imgs += list(chain(*imgss_p))
            labels += list(chain(*labelss_p))
            return imgs, labels
        elif self.add_type == ADD_TYPE.COVER:
            indss_tar = MutiMixTransform.samp_inds(num_sample, repeat=repeat, num_batch=self.num_output,
                                                   diff=self.diff_output)
            for n in range(repeat):
                for k in range(self.num_output):
                    ind = indss_tar[n, k]
                    imgs[ind] = imgss_p[n][k]
                    labels[ind] = labelss_p[n][k]
            return imgs, labels
        elif self.add_type == ADD_TYPE.COVER_ORD:
            for n in range(repeat):
                for k in range(min(self.num_output, self.num_input)):
                    ind = indss_src[n, k]
                    imgs[ind] = imgss_p[n][k]
                    labels[ind] = labelss_p[n][k]
            return imgs, labels
        elif self.add_type == ADD_TYPE.COVER_SRC:
            for n in range(repeat):
                inds_tar = np.random.choice(
                    a=indss_src[n], replace=self.num_output > self.num_input, size=self.num_output)
                for k in range(self.num_output):
                    ind = inds_tar[k]
                    imgs[ind] = imgss_p[n][k]
                    labels[ind] = labelss_p[n][k]
            return imgs, labels
        else:
            raise Exception('err add type')


# 按透明度混合
class MixAlpha(MutiMixTransform, InitDCTExtractable):
    def __init__(self, repeat=0.2, mix_rate=0.5, add_type=ADD_TYPE.COVER, diff_input=True, diff_output=True):
        MutiMixTransform.__init__(self, num_input=2, num_output=1, repeat=repeat, add_type=add_type,
                                  diff_input=diff_input, diff_output=diff_output)
        self.mix_rate = mix_rate

    def mix(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        img = (1 - self.mix_rate) * imgs[0] + self.mix_rate * imgs[1]
        if isinstance(labels[0], CategoryLabel):
            label = blend_cates(labels[0], labels[1], mix_rate=self.mix_rate)
        elif isinstance(labels[0], ImageItemsLabel):
            label = blend_cate_contss(labels[0], labels[1], mix_rate=self.mix_rate)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return [img], [label]


def _rand_uniform_log(low=0.0, high=1.0, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))


# 马赛克增广
class Mosaic(MutiMixTransform, InitDCTExtractable):
    def __init__(self, repeat=0.5, img_size=(416, 416), add_type=ADD_TYPE.COVER, pad_val=(127, 127, 127),
                 diff_input=True, diff_output=True, scale_aspect=(0.7, 1.3), scale_imgs=None, scale_base=(0.25, 4),
                 resample=cv2.INTER_CUBIC):
        MutiMixTransform.__init__(self, num_input=4, num_output=1, repeat=repeat, add_type=add_type,
                                  diff_input=diff_input, diff_output=diff_output)
        self.img_size = img_size
        self.pad_val = pad_val
        self.scale_aspect = scale_aspect
        self.scale_base = scale_base
        self.resample = resample
        self.scale_imgs = scale_imgs

    def mix(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        w, h = self.img_size
        # 图像缩放
        whs = np.array([[img.shape[1], img.shape[0]] for img in imgs], dtype=np.float32)
        scales = np.ones_like(whs)
        if self.scale_base is not None:
            scales = scales * _rand_uniform_log(self.scale_base[0], self.scale_base[1], size=1)
        if self.scale_aspect is not None:
            scales = scales * _rand_uniform_log(self.scale_aspect[0], self.scale_aspect[1], size=whs.shape[1])[None, :]
        if self.scale_imgs is not None:
            scales = scales * _rand_uniform_log(self.scale_imgs[0], self.scale_imgs[1], size=whs.shape[0])[:, None]
        whs = (whs * scales).astype(np.int32)
        # 中心点确定
        l, r = np.max(whs[[0, 1], 0]), w - np.max(whs[[2, 3], 0])
        t, d = np.max(whs[[0, 2], 1]), h - np.max(whs[[1, 3], 1])
        wp = int(np.random.uniform(low=min(l, r), high=max(l, r)))
        hp = int(np.random.uniform(low=min(t, d), high=max(t, d)))
        # 定义偏移量
        xyxys_rgn = np.array([
            [wp - whs[0, 0], hp - whs[0, 1], wp, hp],
            [wp - whs[1, 0], hp, wp, hp + whs[1, 1]],
            [wp, hp - whs[2, 1], wp + whs[2, 0], hp],
            [wp, hp, wp + whs[3, 0], hp + whs[3, 1]]]).astype(np.int32)
        xyxys_rgn = xyxysN_clip(xyxys_rgn, np.array([0, 0, w, h]))
        whs_r = xyxys_rgn[:, 2:4] - xyxys_rgn[:, :2]
        xyxys_src = np.array([
            [max(whs_r[0, 0] - wp, 0), max(whs_r[0, 1] - hp, 0), whs_r[0, 0], whs_r[0, 1]],
            [max(whs_r[1, 0] - wp, 0), 0, whs_r[1, 0], min(h - hp, whs_r[1, 1])],
            [0, max(whs_r[2, 1] - hp, 0), min(w - wp, whs_r[2, 0]), whs_r[2, 1]],
            [0, 0, min(w - wp, whs_r[3, 0]), min(h - hp, whs_r[3, 1])]]).astype(np.int32)
        # 整合
        img_sum = np.zeros(shape=(self.img_size[1], self.img_size[0], 3)) + np.array(self.pad_val)
        label_sum = labels[0].empty()
        for i, (img, label, scale, xyxy_src, xyxy_rgn, wh) in enumerate(
                zip(imgs, labels, scales, xyxys_src, xyxys_rgn, whs)):
            if np.any(xyxy_rgn[2:4] - xyxy_rgn[:2] <= 0):
                continue
            label.linear_(scale=scale, bias=xyxy_rgn[2:4] - xyxy_src[2:4], size=(w, h))
            label.clip_(xyxyN_rgn=xyxy_rgn)
            label.filt_measure_(thres=1)
            label_sum.extend(label)
            if not np.all(scale == 1):
                img = cv2.resize(img, dsize=wh.astype(np.int32), interpolation=self.resample)
            img_sum[xyxy_rgn[1]:xyxy_rgn[3], xyxy_rgn[0]:xyxy_rgn[2]] = \
                img[xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
        label_sum.ctx_size = self.img_size
        return [img_sum.astype(np.uint8)], [label_sum]


def _samp_pair_scale(scale=None, keep_aspect=False, ):
    if scale is None:
        return np.ones(shape=2)
    scale = _pair(scale)
    if keep_aspect:
        scale_smpd = np.random.uniform(low=scale[0], high=scale[1], size=1)
        scale_smpd = np.repeat(scale_smpd, repeats=2)
    else:
        scale_smpd = np.random.uniform(low=scale[0], high=scale[1], size=2)
    return scale_smpd


def xyxyN_samp_area(xyxyN: np.ndarray, aspect=None, area_range=(0.5, 1)) -> np.ndarray:
    xyxyN = np.array(xyxyN)
    wh = xyxyN[2:4] - xyxyN[0:2]
    if aspect is not None:
        area_samp = np.random.uniform(low=area_range[0], high=area_range[1]) * np.prod(wh)
        wh_patch = np.sqrt([area_samp * aspect, area_samp / aspect])
    else:
        len_range = np.sqrt(area_range)
        wh_patch = np.random.uniform(low=len_range[0], high=len_range[1], size=2) * wh
    wh_patch = wh_patch.astype(np.int32)
    x1 = np.random.uniform(low=xyxyN[0], high=xyxyN[2] - wh_patch[0] + 1)
    y1 = np.random.uniform(low=xyxyN[1], high=xyxyN[3] - wh_patch[1] + 1)
    xyxy_patch = np.array([x1, y1, x1 + wh_patch[0], y1 + wh_patch[1]]).astype(np.int32)
    return xyxy_patch


def xyxyN_samp_size(xyxyN: np.ndarray, patch_size=(0.5, 1.0)) -> np.ndarray:
    xyxyN = np.array(xyxyN)
    w, h = xyxyN[2:4] - xyxyN[0:2]
    pw, ph = patch_size
    pw = int(pw * w) if isinstance(pw, float) else pw
    ph = int(ph * h) if isinstance(ph, float) else ph
    pw = min(pw, w)
    ph = min(ph, h)
    x1 = np.random.randint(low=xyxyN[0], high=xyxyN[2] - pw + 1)
    y1 = np.random.randint(low=xyxyN[1], high=xyxyN[3] - ph + 1)
    xyxy_patch = np.array([x1, y1, x1 + pw, y1 + ph]).astype(np.int32)
    return xyxy_patch


def maskNb_samp_ptchNb(maskNb: np.ndarray, ptchNb: np.ndarray) -> np.ndarray:
    h, w = maskNb.shape
    ph, pw = ptchNb.shape
    pw = min(pw, w)
    ph = min(ph, h)
    maskNb_part = maskNb[ph // 2:h - ph // 2 - ph % 2, pw // 2:w - pw // 2 - pw % 2]
    if not np.any(maskNb_part):
        return np.zeros(shape=4)
    maskNb_valid = cv2.erode(maskNb_part.astype(np.uint8), kernel=ptchNb.astype(np.uint8))
    ys, xs = np.nonzero(maskNb_valid)
    if len(ys) == 0:
        return np.zeros(shape=4)
    index = np.random.choice(a=len(ys))
    xc, yc = xs[index], ys[index]
    return np.array([xc, yc])


def maskNb_samp_size(maskNb: np.ndarray, patch_size=(0.2, 0.2)) -> np.ndarray:
    h, w = maskNb.shape
    pw, ph = patch_size
    pw = int(pw * w) if isinstance(pw, float) else pw
    ph = int(ph * h) if isinstance(ph, float) else ph
    xc, yc = maskNb_samp_ptchNb(maskNb, ptchNb=np.ones((ph, pw)))
    return np.array([xc, yc, xc + pw, yc + ph])


# 目标区域的裁剪混合
class CutMix(MutiMixTransform, InitDCTExtractable):
    def __init__(self, repeat=0.5, num_patch=2.0, scale=(0.5, 1.5), keep_aspect=False, with_frgd=True,
                 thres_irate=0.2, add_type=ADD_TYPE.COVER, diff_input=True, diff_output=True,
                 resample=cv2.INTER_CUBIC):
        MutiMixTransform.__init__(self, num_input=2, num_output=1, repeat=repeat, add_type=add_type,
                                  diff_input=diff_input,
                                  diff_output=diff_output)
        self.num_patch = num_patch
        self.thres_irate = thres_irate
        self.scale = scale
        self.keep_aspect = keep_aspect
        self.resample = resample
        self.with_frgd = with_frgd

    def cutmix_cates(self, imgs, cates):
        imgs = [img2imgN(img) for img in imgs]
        xyxy_src = xyxyN_samp_area(np.array((0, 0, imgs[1].shape[1], imgs[1].shape[0])), aspect=None,
                                   area_range=(0.5, 1))
        patch = imgs[1][xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
        aspect = patch.shape[1] / patch.shape[0] if self.keep_aspect else None
        xyxy_dst = xyxyN_samp_area(np.array((0, 0, imgs[1].shape[1], imgs[1].shape[0])), aspect=aspect,
                                   area_range=(0.5, 1))
        patch = cv2.resize(patch, dsize=xyxy_dst[2:4] - xyxy_dst[:2], interpolation=self.resample)
        imgs[0][xyxy_dst[1]:xyxy_dst[3], xyxy_dst[0]:xyxy_dst[2]] = patch
        mix_rate = np.prod(patch.shape) / np.prod(imgs[0].shape)
        cate = blend_cates(cates[0], cates[1], mix_rate=mix_rate)
        return imgs[0], cate

    def cutmix_items(self, imgs, labels):
        num_src = len(labels[1])
        num_patch = int(np.ceil(self.num_patch * num_src)) \
            if isinstance(self.num_patch, float) else self.num_patch
        if num_patch == 0:
            return imgs[0], labels[0]
        img_size = labels[0].img_size
        imgs = [img2imgN(img) for img in imgs]
        patches = []
        masks = []
        xyxys_patch = []
        items_patch = []
        inds = np.random.choice(size=num_patch, a=min(num_src, num_patch), replace=True)
        for ind in inds:
            item = copy.deepcopy(labels[1][ind])
            if isinstance(item, BoxItem):
                border = copy.deepcopy(item.border)
                xyxy_src = XYXYBorder.convert(border).xyxyN.astype(np.int32)
                wh_src = xyxy_src[2:4] - xyxy_src[:2]
                if not self.with_frgd:
                    mask = np.full(shape=wh_src, fill_value=1.0)
                else:
                    border.linear_(bias=-xyxy_src[:2], size=xyxy_src[2:4] - xyxy_src[:2])
                    mask = border.maskNb.astype(np.float32)
            elif isinstance(item, InstItem):
                rgn = RefValRegion.convert(item.rgn)
                xyxy_src = rgn.xyxyN.astype(np.int32)
                wh_src = xyxy_src[2:4] - xyxy_src[:2]
                if not self.with_frgd:
                    mask = np.full(shape=wh_src, fill_value=1.0)
                else:
                    mask = rgn.maskNb_ref.astype(np.float32)
            else:
                raise Exception('err item')
            patch = imgs[1][xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
            scale_smpd = _samp_pair_scale(self.scale, self.keep_aspect)
            wh_dst = (wh_src * scale_smpd).astype(np.int32)
            xy_dst = (np.random.rand(2) * (np.array(img_size) - wh_dst)).astype(np.int32)
            bias = xy_dst - xyxy_src[:2] * scale_smpd
            item.linear_(bias=bias, scale=scale_smpd, size=img_size)
            xyxy_dst = np.concatenate([xy_dst, xy_dst + wh_dst], axis=0)
            if not np.all(scale_smpd == 1):
                mask = cv2.resize(mask, wh_dst, interpolation=self.resample)
                patch = cv2.resize(patch, wh_dst, interpolation=self.resample)
            xyxys_patch.append(xyxy_dst)
            masks.append(mask)
            patches.append(patch)
            items_patch.append(item)

        xyxys_dist = labels[0].export_xyxysN()
        # 放置patch
        for i in range(num_patch):
            xyxy_patch = xyxys_patch[i]
            irate = ropr_arr_xyxysN(xyxy_patch[None], xyxys_dist, opr_type=OPR_TYPE.IRATE2)
            if np.max(irate) > self.thres_irate:  # 防止新粘贴的图像影响原有目标
                continue
            imgs[0][xyxy_patch[1]:xyxy_patch[3], xyxy_patch[0]:xyxy_patch[2]] = \
                np.where(masks[i][..., None] > 0, patches[i],
                         imgs[0][xyxy_patch[1]:xyxy_patch[3], xyxy_patch[0]:xyxy_patch[2]])
            labels[0].append(items_patch[i])
            xyxys_dist = np.concatenate([xyxys_dist, xyxy_patch[None]], axis=0)
        return imgs[0], labels[0]

    def mix(self, imgs, labels):
        if isinstance(labels[0], CategoryLabel):
            img, label = self.cutmix_cates(imgs, labels)
        elif isinstance(labels[0], ImageItemsLabel):
            img, label = self.cutmix_items(imgs, labels)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return [img], [label]


@dct_registor
class ExchangeMix(MutiMixTransform, InitDCTExtractable):

    def __init__(self, repeat=0.3, add_type=ADD_TYPE.COVER_SRC, diff_input=True, diff_output=True, ):
        MutiMixTransform.__init__(self, num_input=2, num_output=2, repeat=repeat, add_type=add_type,
                                  diff_input=diff_input,
                                  diff_output=diff_output)

    def mix(self, imgs, labels):
        if isinstance(labels[0], ImageItemsLabel):
            imgs, labels = self.exchangemix_items(imgs, labels)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return imgs, labels

    def exchangemix_items(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        masks = []
        for label in labels:
            if isinstance(label, BoxesLabel):
                mask = label.export_border_masksN_enc(img_size=label.img_size, num_cls=-1)
            elif isinstance(label, InstsLabel):
                mask = label.export_masksN_enc(img_size=label.img_size, num_cls=-1)
            else:
                raise Exception('err fmt ' + label.__class__.__name__)
            masks.append(mask)

        min_shape = np.minimum(masks[0].shape, masks[1].shape)
        mask0 = (masks[0][:min_shape[0], :min_shape[1]] != -1)
        mask1 = (masks[1][:min_shape[0], :min_shape[1]] != -1)
        mask_join = (mask0 + mask1)[..., None]

        img0 = imgs[0][:min_shape[0], :min_shape[1]]
        img1 = imgs[1][:min_shape[0], :min_shape[1]]

        imgs[0][:min_shape[0], :min_shape[1]] = np.where(mask_join, img0, img1)
        imgs[1][:min_shape[0], :min_shape[1]] = np.where(mask_join, img1, img0)
        return imgs, labels


# </editor-fold>


# <editor-fold desc='lean扩展'>
class ConvertBorder(GeneralTransform):
    def extract_dct(self):
        return dict(border_type=self.border_type.__name__)

    def __init__(self, border_type=XYWHABorder):
        self.border_type = border_type

    def __call__(self, imgs, labels):
        for img, label in zip(imgs, labels):
            assert isinstance(label, ImageItemsLabel), 'fmt err ' + label.__class__.__name__
            for j, item in enumerate(label):
                if isinstance(item, BoxItem) or isinstance(item, InstItem):
                    item.border = self.border_type.convert(item.border)
        return imgs, labels


@dct_registor
class FlipBox(GeneralTransform, InitDCTExtractable):
    def __init__(self, wflip=0.5, hflip=0.5):
        super(FlipBox, self).__init__()
        self.wflip = wflip
        self.hflip = hflip

    def _flipbox_pnt(self, xy0, a0, xy, a, wflip=False, hflip=False):
        if not wflip and not hflip:
            return xy, a
        elif wflip and hflip:
            return 2 * xy0 - xy, a + math.pi
        ar0 = a0 if wflip else a0 + math.pi / 2
        v = np.array([-np.cos(ar0), np.sin(ar0)])
        scale = np.dot(xy - xy0, v) * 2
        return xy - scale * v, 2 * ar0 - a

    def _flipbox_boxes(self, imgP, boxes):
        xywhas = boxes.export_xywhasN(aname_bdr='border_ref')
        for j, box in enumerate(boxes):
            assert isinstance(box, InstRefItem), 'fmt err ' + box.__class__.__name__
            wflip = np.random.rand() < self.wflip
            hflip = np.random.rand() < self.hflip
            if not wflip and not hflip:
                continue
            xywha = box.border_ref.xywhaN
            xywhas_other = np.concatenate([xywhas[:j], xywhas[(j + 1):]], axis=0)
            iareas = ropr_arr_xywhasN(
                np.broadcast_to(xywha, shape=xywhas_other.shape), xywhas_other, opr_type=OPR_TYPE.IAREA)
            if np.any(iareas > 0):
                continue
            # imgP = imgP_rflip_paste_xywhaN(imgP, xywha, vflip=wflip, flip=hflip)
            box.border.xywhaN[:2], box.border.xywhaN[4] = self._flipbox_pnt(
                xy0=xywha[:2], a0=xywha[4], xy=box.border.xywhaN[:2], a=box.border.xywhaN[4], wflip=wflip, hflip=hflip)

        return imgP

    def _flipbox_insts(self, imgP, insts):
        xywhas = insts.export_xywhasN(aname_bdr='border_ref')
        for j, inst in enumerate(insts):
            assert isinstance(inst, InstRefItem), 'fmt err ' + inst.__class__.__name__
            wflip = np.random.rand() < self.wflip
            hflip = np.random.rand() < self.hflip
            if not wflip and not hflip:
                continue
            xywha = inst.border_ref.xywhaN
            xywhas_other = np.concatenate([xywhas[:j], xywhas[(j + 1):]], axis=0)
            iareas = ropr_arr_xywhasN(
                np.broadcast_to(xywha, shape=xywhas_other.shape), xywhas_other, opr_type=OPR_TYPE.IAREA)
            if np.any(iareas > 0):
                continue
            # imgP = imgP_rflip_paste_xywhaN(imgP, xywha, vflip=wflip, flip=hflip)
            maskP = inst.rgn.maskP
            # maskP = imgP_rflip_paste_xywhaN(maskP, xywha, vflip=wflip, flip=hflip)
            inst.rgn = AbsBoolRegion(maskP)
            inst.border.xywhaN[:2], inst.border.xywhaN[4] = self._flipbox_pnt(
                xy0=xywha[:2], a0=xywha[4], xy=inst.border.xywhaN[:2], a=inst.border.xywhaN[4], wflip=wflip,
                hflip=hflip)
        return imgP

    def __call__(self, imgs, labels):
        for i, (img, label) in enumerate(zip(imgs, labels)):
            imgP = img2imgP(img)
            if isinstance(label, BoxesLabel):
                imgP = self._flipbox_boxes(imgP, label)
            elif isinstance(label, InstsLabel):
                imgP = self._flipbox_insts(imgP, label)
            else:
                raise Exception('fmt err ' + label.__class__.__name__)
            imgs[i] = imgP
        return imgs, labels


# </editor-fold>


#  <editor-fold desc='cap扩展'>
def _cutsN2intervalsN(cutsN: np.ndarray, low=0.0, high=np.inf) -> np.ndarray:
    cuts_min = np.concatenate([[low], cutsN], axis=0)
    cuts_max = np.concatenate([cutsN, [high]], axis=0)
    return np.stack([cuts_min, cuts_max], axis=1)


def _xyxyN_wsplit(xyxyN: np.ndarray, cutsN: np.ndarray) -> np.ndarray:
    ints = _cutsN2intervalsN(cutsN)
    xyxysN_cliped = np.repeat(xyxyN[None], axis=0, repeats=ints.shape[0])
    xyxysN_cliped[:, 0] = np.maximum(xyxysN_cliped[:, 0], ints[:, 0])
    xyxysN_cliped[:, 2] = np.minimum(xyxysN_cliped[:, 2], ints[:, 1])
    return xyxysN_cliped


def _xyxyN_hsplit(xyxyN: np.ndarray, cutsN: np.ndarray) -> np.ndarray:
    ints = _cutsN2intervalsN(cutsN)
    xyxysN_cliped = np.repeat(xyxyN[None], axis=0, repeats=ints.shape[0])
    xyxysN_cliped[:, 1] = np.maximum(xyxysN_cliped[:, 1], ints[:, 0])
    xyxysN_cliped[:, 3] = np.minimum(xyxysN_cliped[:, 3], ints[:, 1])
    return xyxysN_cliped


def _make_gray_steel(piece, axis=0):
    gray_piece = np.mean(piece, axis=(axis, 2), keepdims=True)
    texture = np.random.randint(low=-30, high=30, size=gray_piece.shape)
    noise = np.random.randint(low=-10, high=10, size=(piece.shape[0], piece.shape[1], 1))
    return np.clip(noise + gray_piece + texture, a_min=0, a_max=255)


def _rand_interval(low, high, wid_low, wid_high):
    wid = np.random.randint(low=wid_low, high=wid_high)
    start = np.random.randint(low=low, high=max(high - wid, low))
    end = min(start + wid, high)
    return start, end


class BarMask(GeneralTransform, InitDCTExtractable):

    def __init__(self, p=0.2, wid_low=10, wid_high=20, num_max=5):
        self.p = p
        self.wid_low = wid_low
        self.wid_high = wid_high
        self.num_max = num_max

    def trans_datas(self, imgs, labels):
        imgs = imgsP2imgsN(imgs)
        for i, (imgN, label) in enumerate(zip(imgs, labels)):
            if np.random.rand() > self.p:
                continue
            xyxy_img = np.array([0, 0, label.img_size[0], label.img_size[1]])
            num_samp = np.random.randint(low=1, high=self.num_max)
            xyxys = label.export_xyxysN()
            # 横切或者竖切
            if np.random.rand() > 0.5:
                axis, v1s, v2s, spliter = 0, xyxys[:, 1], xyxys[:, 3], _xyxyN_hsplit
            else:
                axis, v1s, v2s, spliter = 1, xyxys[:, 0], xyxys[:, 2], _xyxyN_wsplit
            for j in range(num_samp):
                r1, r2 = _rand_interval(low=0, high=imgN.shape[axis], wid_low=self.wid_low, wid_high=self.wid_high)
                xyxy1, xyxy_fill, xyxy2 = spliter(xyxy_img, cutsN=np.array([r1, r2]))
                for k, (item, v1, v2) in enumerate(zip(label, v1s, v2s)):
                    if v1 >= r2 or v2 <= r1:
                        pass
                    elif v1 >= r1 and v2 <= r2:
                        xyxys[k] = 0
                        item.border = XYXYBorder(np.zeros(4), label.img_size)
                    elif v1 < r1 and v2 > r2:
                        pass
                    else:
                        xyxy_rgn = xyxy1 if v1 < r1 else xyxy2
                        item.clip_(xyxy_rgn)
                        xyxys[k] = xyxyN_clip(xyxys[k], xyxy_rgn)
                colors = imgN[xyxy_fill[1]:xyxy_fill[3], xyxy_fill[0]:xyxy_fill[2]]
                imgN[xyxy_fill[1]:xyxy_fill[3], xyxy_fill[0]:xyxy_fill[2]] = _make_gray_steel(colors, axis=1 - axis)
            label.filt_measure_(thres=1)
        return imgs, labels


class RmoveCap(GeneralTransform, InitDCTExtractable):

    def __init__(self, p=0.2, num_thres=7, rmv_item=True, force_normal=False, num_rmv=1,
                 ignore_noise=False):
        self.p = p
        self.num_thres = num_thres
        self.rmv_item = rmv_item
        self.force_normal = force_normal
        self.num_rmv = num_rmv
        self.ignore_noise = ignore_noise

    def trans_datas(self, imgs, labels):
        imgs = imgsP2imgsN(imgs)
        for i, (img, label) in enumerate(zip(imgs, labels)):
            insus, caps = label.split(lambda item: item.category.cindN < 2)
            imgN = img2imgN(img)

            for j, item in enumerate(insus):
                if self.force_normal:
                    item['name'] = 'insulator_normal'
                    item.category.cindN = 0
                if self.ignore_noise and item.get('noise', False):
                    continue
                if item.border.measure < 64:
                    continue
                if np.random.rand() > self.p:
                    continue
                cluster = item['cluster']
                caps_j = [cap for cap in caps if cap['cluster'] == cluster]
                if len(caps_j) < self.num_thres:
                    continue
                areas = np.array([cap.border.area for cap in caps_j])
                areas[areas < 10] = 0
                if np.sum(areas) == 0:
                    continue
                for k in range(self.num_rmv):
                    index = np.random.choice(a=len(areas), p=areas / np.sum(areas), replace=False)
                    areas[index] = 0
                    cap = caps_j[index]
                    xlyl = cap.border.xlylN
                    imgN = imgN_polr_fill_mirr_std(imgN=imgN, xlyl=xlyl, rand_cen=True, expand=1.5)
                    item['name'] = 'insulator_blast'
                    item['aug'] = True
                    item.category.cindN = 1
                    if self.rmv_item:
                        caps.remove(cap)
                    else:
                        cap['name'] = 'cap_missing'
                        cap['aug'] = True
                        cap.category.cindN = 4
            insus += caps
            labels[i] = insus
            imgs[i] = imgN

        return imgs, labels


# </editor-fold>


class ConvertItemsToCategoryByMain(IndependentTransform, InitDCTExtractable):

    def trans_data(self, img, label):
        cate = IndexCategory(num_cls=1, cindN=0)
        for item in label:
            if item.get('main', False):
                cate = item.category
                break
        label_c = CategoryLabel(category=cate, img_size=label.img_size, meta=label.meta, **label.kwargs)
        label_c.ctx_from(label)
        return img, label_c


#  <editor-fold desc='增广序列'>
PAD_CVAL = 127
PAD_CVALS = (PAD_CVAL, PAD_CVAL, PAD_CVAL)


@dct_registor
class AugNorm(SizedCompose):
    def __init__(self, img_size, thres=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [LargestMaxSize(max_size=img_size),
                 A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS,
                               border_mode=pad_mode, always_apply=True),
                 ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV1(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, scale=(0.9, 1.1),
                 **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(scale=scale, p=p, cval=PAD_CVALS, mode=pad_mode),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV2(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(scale=(0.8, 1.2), p=p, cval=PAD_CVALS, mode=pad_mode),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV3(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(shear=(-5, 5), scale=(0.8, 1.2), p=p,
                         translate_percent=(-0.2, 0.2), cval=PAD_CVALS, mode=pad_mode),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV3R(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, scale=(0.5, 2),
                 rotate=(-180, 180), shear=(-5, 5), translate_percent=(-0.2, 0.2), **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(shear=shear, scale=scale, p=p, rotate=rotate,
                         translate_percent=translate_percent, cval=PAD_CVALS, mode=pad_mode),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_Flip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV4(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, repeat=0.2, cen_range=0.8, scale=(0.5, 1.5), **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            Mosaic(repeat=repeat, img_size=img_size, add_type=ADD_TYPE.REPLACE, pad_val=PAD_CVALS, ),
            A_Compose([
                A.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV5(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, repeat=1.0, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            Mosaic(repeat=repeat, img_size=img_size, add_type=ADD_TYPE.REPLACE, pad_val=PAD_CVALS, ),
            A_Compose([
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV5Cap(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, repeat=1.0, scale=(0.5, 2.0),
                 remove_label=False, rotate=(-180, 180), **kwargs):
        trans = [
            ItemsFilt(fltr=lambda item: not (item.category.cindN == 4 and item.get('repeat', 1) <= 1)),
            LargestMaxSize(max_size=img_size),
            Mosaic(repeat=repeat, img_size=img_size, add_type=ADD_TYPE.REPLACE, pad_val=PAD_CVALS, ),
            RmoveCap(p=0.1, num_thres=5, rmv_item=remove_label, num_rmv=1),
            BarMask(p=0.5, wid_low=3, wid_high=30, num_max=5),
            A_Compose([
                A_Affine(rotate=rotate, cval=PAD_CVALS, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, p=p),
                A_Flip(p=p),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV5Apply(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, repeat=1.0, remove_label=False, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            Mosaic(repeat=repeat, img_size=img_size, add_type=ADD_TYPE.REPLACE, pad_val=PAD_CVALS, ),
            RmoveCap(p=0.1, num_thres=5, rmv_item=remove_label, num_rmv=1),
            ItemsFiltCategory(cinds=(0, 1,)),
            A_Compose([
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV5R(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True,
                 repeat=0.8, cen_range=0.8, rotate=(-180, 180), scale=(0.5, 2), pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),

            Mosaic(repeat=repeat, img_size=img_size, add_type=ADD_TYPE.COVER, pad_val=PAD_CVALS,
                   cen_range=cen_range),
            RmoveCap(p=0.7, num_thres=2, rmv_item=False),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A_Affine(rotate=rotate, cval=PAD_CVALS, mode=cv2.BORDER_CONSTANT,
                         interpolation=cv2.INTER_CUBIC, p=p),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_Flip(p=0.5),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01), ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


@dct_registor
class AugV3Cap(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, scale=(0.8, 1.2),
                 rotate=(-180, 180), shear=(-5, 5), translate_percent=(-0.2, 0.2), **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            RmoveCap(p=0.7, num_thres=5, rmv_item=False, num_rmv=1),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A_Affine(shear=shear, scale=scale, p=p, rotate=rotate, keep_ratio=True,
                         translate_percent=translate_percent, cval=PAD_CVALS, mode=pad_mode),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_RandomRotate90(p=p),
                A_Flip(p=p), ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


class AugAffine(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True,
                 rotate=(-0, 0), scale=(0.5, 2), pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A_Affine(rotate=rotate, scale=scale, cval=PAD_CVALS, mode=cv2.BORDER_CONSTANT,
                         interpolation=cv2.INTER_CUBIC, p=p),
                A_HorizontalFlip(p=0.5),
            ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


class AugRigid(SizedCompose):
    def __init__(self, img_size, thres=1, p=0.5, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A_RandomRotate90(p=p),
                A_Flip(p=p),
            ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)


class AugTest(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True,
                 repeat=1.0, cen_range=0.8, rotate=(-90, 90), scale=(0.5, 1.5), **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            # CutMix(repeat=0.5, add_type=ADD_TYPE.REPLACE, diff_input=True, diff_output=True),
            RmoveCap(p=1),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=False))
        SizedCompose.__init__(self, *trans)



class AugV3D2C(SizedCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            ConvertItemsToCategoryByMain(),
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(shear=(-5, 5), scale=(0.8, 1.2), p=p,
                         translate_percent=(-0.2, 0.2), cval=PAD_CVALS, mode=pad_mode),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)



class AugNormD2C(SizedCompose):
    def __init__(self, img_size, thres=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            ConvertItemsToCategoryByMain(),
            LargestMaxSize(max_size=img_size),
            A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS,
                          border_mode=pad_mode, always_apply=True),
        ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)

# </editor-fold>
