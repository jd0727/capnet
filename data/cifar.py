from .folder import FolderClassificationDataset
from .processing import *





class CIFAR10DataSet(NameMapper, torchvision.datasets.CIFAR10, MDataSet):
    IMG_SIZE = (32, 32)

    def __init__(self, root, set_name, cls_names):
        NameMapper.__init__(self, cls_names=cls_names)
        self._set_name = set_name
        self._root = root
        torchvision.datasets.CIFAR10.__init__(self, root=root, train=set_name == 'train', download=False)

    def __getitem__(self, index):
        return MDataSet.__getitem__(self, index)

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
    def labels(self):
        lbs = []
        for index, cind in enumerate(self.targets):
            lbs.append(CategoryLabel(
                category=IndexCategory(cindN=cind, num_cls=self.num_cls, conf=1),
                img_size=CIFAR10DataSet.IMG_SIZE, meta=str(index), name=self.cind2name(cind)))
        return lbs

    def _index2data(self, index):
        img, cind = self.data[index], self.targets[index]
        cate = CategoryLabel(
            category=IndexCategory(cindN=cind, num_cls=self.num_cls, conf=1),
            img_size=CIFAR10DataSet.IMG_SIZE, meta=str(index), name=self.cind2name(cind))
        return img, cate

    def _meta2data(self, meta):
        return self._index2data(int(meta))


class CIFAR10(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D://Datasets//CIFAR//',
        PLATFORM_DESTOPLAB: 'D://Datasets//CIFAR//',
        PLATFORM_SEV3090: '//home//data-storage//CIFAR',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/CIFAR',
        PLATFORM_BOARD: '/home/jd/img/DataSets/CIFAR'
    }
    CLS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    SET_NAMES = ('train', 'test')

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=CIFAR10.SET_NAMES)
        self.kwargs = kwargs

    def dataset(self, set_name, **kwargs):
        assert set_name in CIFAR10.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=CIFAR10.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = CIFAR10DataSet(**kwargs_update)
        return dataset


class CIFAR100DataSet(NameMapper, torchvision.datasets.CIFAR100, MDataSet):
    def __init__(self, root, set_name, cls_names):
        NameMapper.__init__(self, cls_names=cls_names)
        self._set_name = set_name
        self._root = root
        torchvision.datasets.CIFAR100.__init__(self, root=root, train=set_name == 'train', download=False)

    def __getitem__(self, index):
        return MDataSet.__getitem__(self, index)

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
    def labels(self):
        lbs = []
        for index, cind in enumerate(self.targets):
            lbs.append(CategoryLabel(
                category=IndexCategory(cindN=cind, num_cls=self.num_cls, conf=1),
                img_size=CIFAR10DataSet.IMG_SIZE, meta=str(index), name=self.cind2name(cind)))
        return lbs

    def _index2data(self, index):
        img, cind = self.data[index], self.targets[index]
        cate = CategoryLabel(
            category=IndexCategory(cindN=cind, num_cls=self.num_cls, conf=1),
            img_size=CIFAR10DataSet.IMG_SIZE, meta=str(index), name=self.cind2name(cind))
        return img, cate

    def _meta2data(self, meta):
        return self._index2data(int(meta))


class CIFAR100(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D://Datasets//CIFAR//',
        PLATFORM_DESTOPLAB: 'D://Datasets//CIFAR//',
        PLATFORM_SEV3090: '//home//data-storage//CIFAR',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/CIFAR',
        PLATFORM_BOARD: '/home/jd/img/DataSets/CIFAR'
    }
    CLS_NAMES = (
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
        'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
        'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm')
    SET_NAMES = ('train', 'test')

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=CIFAR100.SET_NAMES)
        self.kwargs = kwargs

    def dataset(self, set_name, **kwargs):
        assert set_name in CIFAR100.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=CIFAR100.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = CIFAR100DataSet(**kwargs_update)
        return dataset


class CINIC10(MDataSource):
    CLS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//CINIC-10//',
        PLATFORM_SEV3090: '//home//data-storage//CINIC-10',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/CINIC-10',
        PLATFORM_BOARD: ''
    }
    SET_NAMES = ('train', 'test', 'val')

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=CINIC10.SET_NAMES)
        self.kwargs = kwargs

    def dataset(self, set_name, **kwargs):
        assert set_name in CINIC10.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=CINIC10.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = FolderClassificationDataset(**kwargs_update)
        return dataset


if __name__ == '__main__':
    ds = CIFAR100()
    loader = ds.loader(set_name='train', batch_size=4, pin_memory=False, num_workers=0, aug_seqTp=None)
    imgs, labs = next(iter(loader))
