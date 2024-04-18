import scipy.io

from data.folder import FolderClassificationDataset
from data.voc import *
from utils.frame import *


def make_dataset(setid_pth, imgllb_pth, img_dir, root):
    setid = scipy.io.loadmat(setid_pth)
    imgllb = scipy.io.loadmat(imgllb_pth)
    labels = imgllb['labels'][0]
    img_pths = [os.path.join(img_dir, img_name) for img_name in sorted(os.listdir(img_dir))]
    for set_name, key in zip(['train', 'test', 'val'], ['trnid', 'tstid', 'valid']):
        set_dir = ensure_folder_pth(os.path.join(root, set_name))
        ids = np.array(setid[key][0]) - 1
        for id in ids:
            print(id)
            label = labels[id]
            lb_dir = ensure_folder_pth(os.path.join(set_dir, 'c' + str(label)))
            img_pth = img_pths[id]
            shutil.copy(img_pth, os.path.join(lb_dir, os.path.basename(img_pth)))
    return True


# if __name__ == '__main__':
#     root = 'D:/Download/Flowers/'
#     setid_pth = os.path.join(root, 'setid.mat')
#     imgllb_pth = os.path.join(root, 'imagelabels.mat')
#     img_dir = os.path.join(root, 'jpg')
#     make_dataset(setid_pth, imgllb_pth, img_dir, root)


class OXFlower(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D://Datasets//OXFlower//',
        PLATFORM_DESTOPLAB: 'D://Datasets//OXFlower//',
        PLATFORM_SEV3090: '//home//data-storage//OXFlower//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//OXFlower//',
        PLATFORM_BOARD: ''
    }
    CLS_NAMES = tuple(['c%d' % i for i in range(1, 103)])

    def __init__(self, root=None, resample=None, cls_names=CLS_NAMES, **kwargs):
        MDataSource.__init__(self, root=root, set_names=('train', 'test'))
        self.resample = resample
        self.kwargs = kwargs
        self.cls_names = cls_names

    def dataset(self, set_name, **kwargs):
        kwargs_update = dict(root=self.root, cls_names=self.cls_names, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)

        dataset = FolderClassificationDataset(**kwargs_update)
        return dataset


if __name__ == '__main__':
    ds = OXFlower()
    loader = ds.loader(set_name='test', batch_size=4, num_workers=0, aug_seq=None)
    imgs, labels = next(iter(loader))
