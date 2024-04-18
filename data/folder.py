import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils.frame import NameMapper, MDataBuffer, MDataSet, MixedDataBuffer
from collections import Counter
from utils.label import IndexCategory, CategoryLabel, img2imgP
from utils.file import *
from external import imagesize

# <editor-fold desc='folder编辑'>
IMAGE_APPENDEIX = ['jpg', 'JPEG', 'png']


def resample_by_names(cls_names, resample):
    presv_inds = []
    for i, cls_name in enumerate(cls_names):
        if not cls_name in resample.keys():
            presv_inds.append(i)
            continue
        resamp_num = resample[cls_name]
        low = np.floor(resamp_num)
        high = np.ceil(resamp_num)
        resamp_num_rand = np.random.uniform(low=low, high=high)
        resamp_num = int(low if resamp_num_rand > resamp_num else high)
        for j in range(resamp_num):
            presv_inds.append(i)
    return presv_inds


def get_pths(root, name_remapper=None):
    img_pths = []
    names = []
    for dir_name in os.listdir(root):
        cls_dir = os.path.join(root, dir_name)
        if not os.path.isdir(cls_dir):
            continue
        if name_remapper is not None:
            dir_name = name_remapper[dir_name]
        for img_dir, _, img_names in os.walk(cls_dir):
            for img_name in img_names:
                if img_name.split('.')[1] not in IMAGE_APPENDEIX:
                    continue
                img_pths.append(os.path.join(cls_dir, img_dir, img_name))
                names.append(dir_name)
    return img_pths, names


def get_cls_names(root, name_remapper=None):
    cls_names = []
    for dir_name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dir_name)):
            continue
        if name_remapper is not None:
            dir_name = name_remapper[dir_name]
        cls_names.append(dir_name)
    return cls_names


# </editor-fold>


def imgs_labels2dataset_pkl(imgs, labels, root, img_folder='images', pkl_name='label', img_extend='jpg'):
    print('Create dataset at ' + root + ' < ' + img_folder + ' , ' + pkl_name + ' > ')
    img_dir = os.path.join(root, img_folder)
    ensure_folder_pth(img_dir)
    pkl_pth = os.path.join(root, ensure_extend(pkl_name, SinglePKLDataset.EXTEND))
    ensure_file_dir(pkl_pth)
    for i, img in MEnumerate(imgs):
        meta = labels[i].meta
        img_pth = os.path.join(img_dir, ensure_extend(meta, img_extend))
        imgP = img2imgP(img)
        imgP.save(img_pth)
    labels.sort(key=lambda x: x.meta)
    save_pkl(pkl_pth, labels)
    print('Create complete')
    return True


class SinglePKLDataset(NameMapper, MDataBuffer, MDataSet):
    EXTEND = 'pkl'

    def __len__(self):
        return len(self._metas)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def set_name(self):
        return self._set_name

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def img_dir(self):
        return os.path.join(self._root, self._img_folder)

    @property
    def img_pths(self):
        img_dir = self.img_dir
        return [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]

    @property
    def pkl_pth(self):
        return os.path.join(self._root, ensure_extend(self._pkl_name, 'pkl'))

    @property
    def pkl_name(self):
        return self._pkl_name

    @property
    def labels(self):
        return self._labels

    def _meta2data(self, meta):
        return

    def _index2data(self, index):
        img_pth = self.img_pths[index]
        img = self.load_img(img_pth)
        label = self.labels[index] if self.labels is not None else None
        return img, label

    def __init__(self, root, set_name, cls_names, img_folder='images', pkl_name='label', **kwargs):
        self._root = root
        self._set_name = set_name
        self._img_folder = img_folder
        self._pkl_name = pkl_name

        if os.path.exists(self.pkl_pth):
            self._labels = load_pkl(self.pkl_pth)
            self._metas = [label.meta for label in self._labels]

        NameMapper.__init__(self, cls_names)
        MDataBuffer.__init__(self, **kwargs)


class FolderClassificationDataset(NameMapper, MixedDataBuffer):
    def __init__(self, root, set_name, cls_names=None, resample=None, name_remapper=None, **kwargs):
        root_set = os.path.join(root, set_name)
        cls_names = get_cls_names(root_set, name_remapper=name_remapper) if cls_names is None else cls_names
        NameMapper.__init__(self, cls_names)
        self._root = root
        self._set_name = set_name
        self._img_pths, self._names = get_pths(root_set, name_remapper=name_remapper)
        if resample is not None:
            presv_inds = resample_by_names(self.names, resample=resample)
            self._img_pths = [self._img_pths[ind] for ind in presv_inds]
            self._names = [self._names[ind] for ind in presv_inds]
        self._metas = [os.path.split(os.path.basename(img_pth))[0] for img_pth in self._img_pths]
        MixedDataBuffer.__init__(self, **kwargs)

    def __len__(self):
        return len(self._metas)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def set_name(self):
        return self._set_name

    @property
    def metas(self):
        return self._metas

    @property
    def labels(self):
        lbs = []
        for img_pth, name, meta in zip(self._img_pths, self._names, self._metas):
            label = CategoryLabel(
                category=IndexCategory(cindN=int(self.name2cind(name)), conf=1, num_cls=self.num_cls),
                img_size=imagesize.get(img_pth), meta=meta, name=name)
            lbs.append(label)
        return lbs

    def _meta2data(self, meta):
        return self._index2data(self._metas.index(meta))

    def _index2data(self, index):
        img_pth, name, meta = self._img_pths[index], self._names[index], self._metas[index]
        img = Image.open(img_pth).convert('RGB')
        label = CategoryLabel(
            category=IndexCategory(cindN=int(self.name2cind(name)), conf=1, num_cls=self.num_cls),
            img_size=img.size, meta=meta, name=name)
        return img, label

    def __repr__(self):
        num_dict = Counter(self._names)
        msg = '\n'.join(['%10s ' % name + ' %5d' % num for name, num in num_dict.items()])
        return msg


class FolderUnlabeledDataset(NameMapper, MixedDataBuffer):
    def __init__(self, root, set_name='unlabel', cls_names=('obj',), **kwargs):
        NameMapper.__init__(self, cls_names)
        self._root = root
        self._set_name = set_name
        file_names = os.listdir(root)
        self._img_pths = [os.path.join(root, file_name) for file_name in file_names]
        self._metas = [os.path.splitext(file_name)[0] for file_name in file_names]
        MixedDataBuffer.__init__(self, **kwargs)

    def __len__(self):
        return len(self._metas)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def set_name(self):
        return self._set_name

    @property
    def metas(self):
        return self._metas

    @property
    def labels(self):
        lbs = []
        for img_pth, meta in zip(self._img_pths, self._metas):
            label = CategoryLabel(
                category=IndexCategory(cindN=0, conf=1, num_cls=self.num_cls),
                img_size=imagesize.get(img_pth), meta=meta, )
            lbs.append(label)
        return lbs

    def _meta2data(self, meta):
        return self._index2data(self._metas.index(meta))

    def _index2data(self, index):
        img_pth, meta = self._img_pths[index], self._metas[index]
        img = Image.open(img_pth).convert('RGB')
        label = CategoryLabel(
            category=IndexCategory(cindN=0, conf=1, num_cls=self.num_cls),
            img_size=img.size, meta=meta)
        return img, label

    def __repr__(self):
        msg = 'FolderUnlabeledDataset ' + self._root
        return msg
