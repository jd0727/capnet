from .folder import *
from .processing import *


class ImageNet(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ImageNet Test//',
        PLATFORM_SEV3090: '//home//data-storage//ImageNet//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//InsulatorC//',
        PLATFORM_BOARD: ''
    }
    SET_NAMES = ('train', 'val')

    def __init__(self, root=None, resample=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=ImageNet.SET_NAMES)
        self.resample = resample
        self.kwargs = kwargs
        lines = load_txt(os.path.join(self.root, 'classes.txt'))
        name_remapper = {}
        for line in lines:
            pieces = line.split(' ')
            name_ori = pieces[1]
            name_map = ' '.join(pieces[2:]).split(',')[1]
            name_remapper[name_ori] = name_map
        self.name_remapper = name_remapper
        self.cls_names = list(self.name_remapper.values())

    def dataset(self, set_name, **kwargs):
        assert set_name in ImageNet.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=self.cls_names, set_name=set_name,
                             name_remapper=self.name_remapper)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = FolderClassificationDataset(**kwargs_update)
        return dataset


class TinyImageNet(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//Tiny-ImageNet//',
        PLATFORM_SEV3090: '//home//data-storage//Tiny-ImageNet//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//Tiny-ImageNet//',
        PLATFORM_BOARD: ''
    }
    SET_NAMES = ('train', 'val')
    def __init__(self, root=None, resample=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=('train', 'val'))
        self.resample = resample
        self.kwargs = kwargs
        lines = load_txt(os.path.join(self.root, 'classes.txt'))
        name_remapper = {}
        for line in lines:
            pieces = line.split(' ')
            name_ori = pieces[1]
            name_map = ' '.join(pieces[2:]).split(',')[1]
            name_remapper[name_ori] = name_map
        self.name_remapper = name_remapper
        self.cls_names = list(self.name_remapper.values())

    def dataset(self, set_name, **kwargs):
        assert set_name in ImageNet.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=self.cls_names, set_name=set_name,
                             name_remapper=self.name_remapper)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = FolderClassificationDataset(**kwargs_update)
        return dataset


if __name__ == '__main__':
    ds = TinyImageNet()
    loader = ds.loader(set_name='val', batch_size=4, num_workers=0, aug_seq=None)
    imgs, labels = next(iter(loader))
