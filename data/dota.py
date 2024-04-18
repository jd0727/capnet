from collections import Counter

from data.processing import *


# <editor-fold desc='标准DOTA'>
def _expand_pths(root, folder, metas, extend='xml'):
    return [os.path.join(root, folder, meta + '.' + extend) for meta in metas]


class DotaDDataset(NameMapper, MDataBuffer, MDataSet):
    ANNO_EXTEND = 'txt'
    IMG_EXTEND = 'png'

    def __init__(self, root, cls_names=None, anno_folder='labelTxt-v1.0', img_folder='images', **kwargs):
        if cls_names is None:
            names = DotaDDataset.collect_names(label_dir=os.path.join(root, anno_folder))
            cls_names = sorted(Counter(names).keys())
        NameMapper.__init__(self, cls_names)
        self.root = root
        # 加载标签
        img_names = sorted(listdir_extend(os.path.join(root, img_folder), extends='png'))
        self._metas = [img_name.split('.')[0] for img_name in img_names]
        self.img_folder = img_folder
        self._anno_folder = anno_folder
        # 预加载
        MDataBuffer.__init__(self, **kwargs)

    @property
    def metas(self):
        return self._metas

    @property
    def anno_folder(self):
        return self._anno_folder

    @anno_folder.setter
    def anno_folder(self, anno_folder):
        self._anno_folder = anno_folder

    @property
    def anno_pths(self):
        return _expand_pths(self.root, folder=self.anno_folder, metas=self.metas, extend=DotaDDataset.ANNO_EXTEND)

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def img_pths(self):
        return _expand_pths(self.root, folder=self.img_folder, metas=self.metas, extend=DotaDDataset.IMG_EXTEND)

    @img_folder.setter
    def img_folder(self, img_folder):
        self._img_folder = img_folder

    @staticmethod
    def collect_names(label_dir):
        label_names = os.listdir(label_dir)
        names = []
        for anno_name in label_names:
            if not str.endswith(anno_name, '.txt'):
                continue
            anno_pth = os.path.join(label_dir, anno_name)
            lines = load_txt(anno_pth)
            for line in lines[2:]:
                names.append(line.split(' ')[-2])
        return names

    @staticmethod
    def prase_anno(anno_pth, img_size, name2cind=None, num_cls=1, rotational=True):
        meta = os.path.basename(anno_pth).split('.')[0]
        boxes = BoxesLabel(img_size=img_size, meta=meta)
        if not os.path.exists(anno_pth):
            return boxes
        lines = load_txt(anno_pth)
        for line in lines[2:]:
            pieces = line.split(' ')
            name = pieces[-2]
            vs = np.array([float(v) for v in pieces[:8]])
            xlyl = vs.reshape(4, 2)
            border_xlyl = XLYLBorder(xlyl, size=img_size)
            if rotational:
                border = XYWHABorder.convert(border_xlyl)
            else:
                border = XYXYBorder.convert(border_xlyl)
            cind = name2cind(name) if name2cind is not None else 0
            category = IndexCategory(cindN=cind, num_cls=num_cls)
            difficult = (pieces[8] == '1')
            box = BoxItem(border=border, category=category, name=name, difficult=difficult)
            boxes.append(box)
        return boxes

    @staticmethod
    def create_anno(anno_pth, boxes, imagesource='GoogleEarth', gsd='0.146343590398'):
        lines = [imagesource, gsd]
        for box in boxes:
            xlyl = XLYLBorder.convert(box.border).xlylN
            xlyl_lst = [str(int(np.round(v))) for v in xlyl.reshape(-1)]
            name = box['name']
            difficult = '1' if 'difficult' in box.keys() and box['difficult'] else '0'
            line = ' '.join(xlyl_lst + [name, difficult])
            lines.append(line)
        save_txt(anno_pth, lines)
        return lines

    def __len__(self):
        return len(self.img_pths)

    def _index2data(self, index):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(self.metas[index], DotaDDataset.IMG_EXTEND))
        anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(self.metas[index], DotaDDataset.ANNO_EXTEND))
        img = self.load_img(img_pth)
        boxes = DotaDDataset.prase_anno(
            anno_pth=anno_pth, img_size=img.size, name2cind=self.name2cind, num_cls=self.num_cls)
        return img, boxes

    # 统计物体个数
    def __repr__(self):
        names = DotaDDataset.collect_names(label_dir=os.path.join(self.root, self.anno_folder))
        num_dict = Counter(names)
        msg = '\n'.join(['%20s ' % name + ' %5d' % num for name, num in num_dict.items()])
        return msg


class Dota(MDataSource):
    IMG_FOLDER = 'images'
    ANNO_FOLDER_ROT = 'labelTxt-v1.5'
    ANNO_FOLDER_RECT = 'labelTxt-v1.5-rect'

    CLS_NAMES = ('bridge', 'ground-track-field', 'harbor', 'helicopter', 'large-vehicle',
                 'roundabout', 'small-vehicle', 'soccer-ball-field', 'swimming-pool', 'baseball-diamond',
                 'basketball-court', 'plane', 'ship', 'storage-tank', 'tennis-court')

    ROOT_MAPPER = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//DOTA//',
        PLATFORM_SEV3090: '//home//data-storage//DOTA',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    BUILDER_MAPPER = {
        TASK_TYPE.DETECTION: DotaDDataset,
    }

    def __init__(self, root=None, pre_aug_seq=None, anno_folder=ANNO_FOLDER_ROT, img_folder=IMG_FOLDER,
                 task_type=TASK_TYPE.DETECTION, **kwargs):
        super().__init__(root=root, set_names=('train', 'val', 'trainval', 'test'), task_type=task_type)
        self.anno_folder = anno_folder
        self.img_folder = img_folder
        self.pre_aug_seq = pre_aug_seq
        self.kwargs = kwargs

    def dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = Dota.BUILDER_MAPPER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, pre_aug_seq=self.pre_aug_seq,
                             anno_folder=self.anno_folder, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = builder(fmt=set_name + '_%d', root=os.path.join(self.root, set_name), **kwargs_update)
        return dataset

# </editor-fold>


# if __name__ == '__main__':
#     ds = Dota.SEV_NEW(task_type=TASK_TYPE.INSTANCE)
#     dataset = ds.dataset('train')
#     cls_names = ['Bridge', 'Ground_Track_Field', 'Harbor', 'Helicopter', 'Large_Vehicle',
#                  'Roundabout', 'Small_Vehicle', 'Soccer_ball_field', 'Swimming_pool', 'baseball_diamond',
#                  'basketball_court', 'plane', 'ship', 'storage_tank', 'tennis_court']
#     rename_dict = dict([(name, name.replace('_', '-').lower()) for name in cls_names])
#     dataset.rename(rename_dict=rename_dict)


# if __name__ == '__main__':
#     ds_dota = Dota.SEV_NEW()
#     ds_voc = ISAIDObj.SEV_NEW()
#     ds_voc.export_dota(ds_dota, label_folder_dota='labelTxt-v1.0x')

# if __name__ == '__main__':
#     ds = Dota.SEV_NEW()
#     # data = ds.dataset('val')
#     loader = ds.loader(set_name='train', batch_size=4, num_workers=0, aug_seq=None)
#     imgs, labels = next(iter(loader))
