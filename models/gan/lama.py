from models.base.resnet import np
from models.base.resunet import ResUNetEnc, ImageONNXExportable
from models.modules import *
from models.modules import _pair, _int2, _auto_pad
import cv2
from utils import *


class MASK_TYPE:
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def _cvt_int_normalize(val: Union[int, float], base: Union[int, float]) -> int:
    if isinstance(val, int):
        return val
    else:
        return int(val * base)


def _cvt_int_rand(val: Union[int, float]) -> int:
    if isinstance(val, int):
        return val
    else:
        return int(np.random.rand() * val)


def make_random_irregular_mask(size, max_angle=1.0, max_len=60, max_width=20, max_repeat=10, max_union=10,
                               mask_type=MASK_TYPE.LINE):
    w, h = size
    mask = np.zeros((h, w), np.float32)
    num_union = np.random.randint(max_union + 1)
    for i in range(num_union):
        num_repeat = 1 + np.random.randint(max_repeat)
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        angle = np.random.uniform(low=0, high=np.pi * 2)
        for j in range(num_repeat):
            length = np.random.randint(max_len + 1)
            brush_w = 1 + np.random.randint(max_width)
            angle = angle + np.random.uniform(-max_angle, max_angle)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, w)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, h)
            print(end_x, end_y)
            if mask_type == MASK_TYPE.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif mask_type == MASK_TYPE.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif mask_type == MASK_TYPE.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask


if __name__ == '__main__':
    size = (200, 200)
    mask = make_random_irregular_mask(size, max_angle=1.0, max_len=60, max_width=20, max_repeat=10,
                                      max_union=5,
                                      mask_type=MASK_TYPE.LINE)
    plt.imshow(mask)
    plt.pause(1e5)
