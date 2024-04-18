from .cifar import CIFAR100, CIFAR10, CINIC10
from .coco import CoCo, datasetI2cocoI_persize, datasetI2cocoI_perbox, CoCoDataSet, CoCoDetectionDataSet, \
    CoCoInstanceDataSet, CoCoSegmentationDataSet
from .cub200 import Cub200
from .dota import Dota
from .folder import SinglePKLDataset
from .hrsc import HRSC, HRSCObj
from .imgnet import ImageNet, TinyImageNet
from .insulator import InsulatorC, InsulatorD, InsulatorObj, InsulatorDI, DistriNetworkDefect, DistriNetworkDevice, \
    DistriNetworkMix
from .insupsv import InsulatorUpsv
from .isaid import ISAID, ISAIDPatch, ISAIDObj, ISAIDPart
from .sfid import SFID, CPLID
from .svhn import SVHN
from .voc import Voc, datasetD2vocD_perbox, datasetD2vocD_persize, datasetD2vocI_background, datasetI2vocI_perbox, \
    datasetI2vocI_persize, VocCommon, VocDetectionDataset, VocInstanceDataset, VocSegmentationDataset, \
    datasetD2vocD_background, VocDataset
from .oxflower import OXFlower