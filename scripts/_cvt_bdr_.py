import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from data import *
from models import *
from tools import *
from utils import *


def cvtor(label):
    for i in range(len(label)):
        box = BoxRefItem.convert(label[i])
        border = XLYLBorder.convert(box.border)
        border.samp(num_samp=5)
        box.border = border
        box.border_ref = XYWHABorder.convert(box.border_ref)
        label[i] = box
    return label


if __name__ == '__main__':
    ds = InsulatorDI.SEV_NEW()
    dataset = ds.dataset('trainval')
    dataset.label_apply(func=cvtor, anno_folder='AnnotationsLean', anno_extend='xml')
