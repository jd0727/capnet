from data.folder import FolderClassificationDataset
from data.processing import *


class Cub200(MDataSource):
    ROOT_MAPPER = {
        PLATFORM_LAPTOP: 'D://Datasets//CUB-200//',
        PLATFORM_DESTOPLAB: 'D://Datasets//CUB-200//',
        PLATFORM_SEV3090: '//home//data-storage//CUB-200//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//CUB-200//',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, resample=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=('train', 'test'))
        self.resample = resample
        self.kwargs = kwargs
        lines = load_txt(os.path.join(self.root, 'classes.txt'))
        self.name_remapper = dict([('c' + line.split(' ')[0], line.split('.')[1].replace('_', ' ')) for line in lines])
        self.cls_names = list(self.name_remapper.values())

    def dataset(self, set_name, **kwargs):
        kwargs_update = dict(root=self.root, cls_names=self.cls_names, set_name=set_name,
                             name_remapper=self.name_remapper, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)

        dataset = FolderClassificationDataset(**kwargs_update)
        return dataset


if __name__ == '__main__':
    ds = Cub200()
    loader = ds.loader(set_name='test', batch_size=4, num_workers=0, aug_seq=None)
    imgs, labels = next(iter(loader))
