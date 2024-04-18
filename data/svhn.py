from torchvision.datasets.mnist import read_image_file, read_label_file

from data.processing import *
import torchvision

# if __name__ == '__main__':
#     d = torchvision.datasets.MNIST(root='D://Datasets//MNIST//', download=True)
import codecs


class MNISTDataSet(NameMapper, MDataSet):
    IMG_SIZE = (28, 28)

    def __len__(self):
        return len(self.imgs)

    def __init__(self, root, set_name, cls_names):
        super(MNISTDataSet, self).__init__(cls_names)
        self._root = root
        self._set_name = set_name
        prefix = 'train' if set_name == 'train' else 't10k'
        self.imgs = read_image_file(os.path.join(self._root, prefix + '-images-idx3-ubyte')).numpy()
        self.targets = read_label_file(os.path.join(self._root, prefix + '-labels-idx1-ubyte')).numpy()

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def labels(self):
        labels = []
        for index, target in enumerate(self.targets):
            labels.append(CategoryLabel(category=IndexCategory(target, num_cls=self.num_cls),
                                        img_size=MNISTDataSet.IMG_SIZE, meta=str(index)))
        return labels

    def _index2data(self, index):
        img = self.imgs[index][..., None]
        label = CategoryLabel(category=IndexCategory(self.targets[index], num_cls=self.num_cls),
                              img_size=MNISTDataSet.IMG_SIZE, meta=str(index))
        return img, label

    def _meta2data(self, meta):
        return self._index2data(int(meta))


class SVHNDataSet(NameMapper, MDataSet):
    def __len__(self):
        return self._tragets.shape[0]

    IMG_SIZE = (32, 32)

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
        for index, cind in enumerate(self._tragets):
            category = IndexCategory(cindN=cind, num_cls=self.num_cls, conf=1)
            lbs.append(CategoryLabel(category=category, img_size=SVHNDataSet.IMG_SIZE, meta=str(index)))
        return lbs

    def _index2data(self, index):
        img, cind = self._imgs[index], int(self._tragets[index])
        # img = np.transpose(img, (1, 2, 0))
        category = IndexCategory(cindN=cind, num_cls=self.num_cls, conf=1)
        cate = CategoryLabel(category=category, img_size=SVHNDataSet.IMG_SIZE, meta=str(index))
        return img, cate

    def _meta2data(self, meta):
        return self._index2data(int(meta))

    def __init__(self, root, cls_names, set_name, **kwargs):
        NameMapper.__init__(self, cls_names=cls_names)
        self._root = root
        self._set_name = set_name
        import scipy.io as sio
        datas = sio.loadmat(os.path.join(self.root, ensure_extend(set_name + '_32x32', 'mat')))
        self._imgs = datas['X'].transpose(3, 0, 1, 2)
        self._tragets = datas['y'].astype(np.int64).squeeze()


class MNIST(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D://Datasets//MNIST//',
        PLATFORM_DESTOPLAB: 'D://Datasets//MNIST//',
        PLATFORM_SEV3090: '//home//data-storage//MNIST',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/MNIST',
        PLATFORM_BOARD: '/home/jd/data/DataSets/MNIST'
    }
    CLS_NAMES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    SET_NAMES = ('train', 'test',)

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=SVHN.SET_NAMES)
        self.kwargs = kwargs

    def dataset(self, set_name, **kwargs):
        assert set_name in MNIST.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=SVHN.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = MNISTDataSet(**kwargs_update)
        return dataset


class SVHN(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D://Datasets//SVHN//',
        PLATFORM_DESTOPLAB: 'D://Datasets//SVHN//',
        PLATFORM_SEV3090: '//home//data-storage//SVHN',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/SVHN',
        PLATFORM_BOARD: '/home/jd/data/DataSets/SVHN'
    }
    CLS_NAMES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    SET_NAMES = ('train', 'test', 'extra')

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=SVHN.SET_NAMES)
        self.kwargs = kwargs

    def dataset(self, set_name, **kwargs):
        assert set_name in SVHN.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=SVHN.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = SVHNDataSet(**kwargs_update)
        return dataset


if __name__ == '__main__':
    ds = MNIST()
    dataset = ds.dataset('train')
    img, label = dataset[5]

# if __name__ == '__main__':
#     ds = SVHN()
#     loader = ds.loader(set_name='train', batch_size=4, pin_memory=False, num_workers=0, aug_seqTp=None)
#     imgs, labs = next(iter(loader))
