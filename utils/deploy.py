import collections
from functools import partial

import onnxruntime
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.onnx import OperatorExportTypes

from .file import *


# <editor-fold desc='模型统计'>
class FLOPHook():

    # 统计不为0的通道
    @staticmethod
    def count_nonzero(data, dim=0):
        if not dim == 0:
            data = data.transpose(dim, 0).contiguous()
        sum_val = torch.sum(torch.abs(data.reshape(data.shape[0], -1)), dim=1)
        num = torch.count_nonzero(sum_val).item()
        return num

    @staticmethod
    def Conv2d(module, data_input, data_output, ignore_zero, handler):
        _, _, fw, fh = list(data_output.size())
        co, ci, kw, kh = list(module.weight.size())
        groups = module.groups
        if not ignore_zero:
            ci = FLOPHook.count_nonzero(data_input[0], dim=1)
            co = FLOPHook.count_nonzero(module.weight, dim=0)
        flop = (kw * kh * ci) * co * fw * fh / groups
        if module.bias is not None:
            flop += co * fw * fh
        size = (co, ci, kw, kh)
        handler(size=size, flop=flop)

    @staticmethod
    def AvgPool(module, data_input, data_output, ignore_zero, handler):
        _, c, fwi, fhi = data_input[0].shape
        _, _, fwo, fho = data_output.shape
        if not ignore_zero:
            c = FLOPHook.count_nonzero(data_output, dim=1)
        flop = (1 + (fwi // fwo) * (fhi // fho)) * fwo * fho * c
        handler(size=[c], flop=flop)

    @staticmethod
    def Linear(module, data_input, data_output, ignore_zero, handler):
        co, ci = module.weight.shape
        if not ignore_zero:
            ci = FLOPHook.count_nonzero(data_input[0], dim=1)
            co = FLOPHook.count_nonzero(data_output, dim=1)
        flop = (ci) * co
        if not module.bias is None:
            flop += co
        handler(size=[co, ci], flop=flop)

    @staticmethod
    def BatchNorm2d(module, data_input, data_output, ignore_zero, handler):
        _, c, fw, fh = data_input[0].shape
        if not ignore_zero:
            c = FLOPHook.count_nonzero(module.weight.data)
        flop = c * fw * fh * 2
        handler(size=[c], flop=flop)

    MAPPER = {
        nn.SiLU: None,
        nn.Conv2d: Conv2d,
        # nn.AvgPool2d: hook_AvgPool,
        nn.AvgPool2d: None,
        # nn.AdaptiveAvgPool2d: hook_AvgPool,
        nn.AdaptiveAvgPool2d: None,
        nn.Linear: Linear,
        nn.BatchNorm2d: BatchNorm2d,
        nn.ReLU: None,
        nn.MaxPool2d: None,
        nn.Dropout: None,
        nn.Dropout2d: None,
        nn.CrossEntropyLoss: None,
        nn.UpsamplingNearest2d: None,
        nn.LeakyReLU: None,
        # nn.Mish: None,
        nn.UpsamplingBilinear2d: None,
        nn.Identity: None
    }

    @staticmethod
    def get_hook(model, handler, ignore_zero):
        type_m = type(model)
        if type_m in FLOPHook.MAPPER.keys():
            hook = FLOPHook.MAPPER[type_m]
            return partial(hook.__func__, ignore_zero=ignore_zero, handler=handler) if hook is not None else None
        else:
            print('Not support type ' + model.__class__.__name__)
            return None


def count_loader_delay(loader, iter_num=10):
    i = 0
    iterator = iter(loader)
    time_start = time.time()
    while i < iter_num:
        try:
            _ = next(iterator)
        except StopIteration:
            break
        i = i + 1
    time_end = time.time()
    delay = (time_end - time_start) / i
    return delay


def count_fn_delay(fn, args, iter_num=100):
    time_start = time.time()
    for i in range(iter_num):
        _ = fn(args)
    time_end = time.time()
    total_time = time_end - time_start
    delay = total_time / iter_num
    return delay


def count_model_delay(model, input, iter_num=1000):
    if isinstance(model, nn.Module):
        with torch.no_grad():
            delay = count_fn_delay(fn=model, iter_num=iter_num, args=input)
    else:
        delay = count_fn_delay(fn=model, iter_num=iter_num, args=input)
    return delay


def count_flop(model, input, ignore_zero=False):
    assert isinstance(model, nn.Module), 'mdoel err'
    msgs = collections.OrderedDict()
    handles = []

    def handler(module_name, cls_name, size, flop):
        if module_name not in msgs.keys():
            msgs[module_name] = {
                'Name': module_name,
                'Class': cls_name,
                'Size': str(size),
                'FLOP': flop
            }
        else:
            msgs[module_name]['FLOP'] += flop
        return None

    # 添加hook
    def add_hook(module, module_name=None):
        if len(list(module.children())) == 0:
            handler_m = partial(handler, module_name=module_name, cls_name=module.__class__.__name__)
            hook = FLOPHook.get_hook(module, handler_m, ignore_zero)
            if hook is not None:
                handle = module.register_forward_hook(hook)
                handles.append(handle)
        else:
            for name, sub_model in module.named_children():
                sub_model_name = name if module_name is None else module_name + '.' + name
                add_hook(sub_model, sub_model_name)

    # 规范输入
    model = model.eval()
    with torch.no_grad():
        add_hook(model, None)
        _ = model(*input)
    # 移除hook
    for handle in handles:
        handle.remove()
    order = ['Name', 'Class', 'Size', 'FLOP']
    data = pd.DataFrame(columns=order)
    for msg in msgs.values():
        data = pd.concat([data, pd.DataFrame(msg, index=[0])])
    return data


def count_para(model):
    assert isinstance(model, nn.Module), 'mdoel err ' + model.__class__.__name__
    count_para.data = pd.DataFrame(columns=['Name', 'Class', 'Para'])

    def stat_para(module, module_name=None):
        if len(list(module.children())) == 0:
            para_sum = 0
            for para in module.parameters():
                para_sum += para.numel()
            row = pd.DataFrame({
                'Name': module_name,
                'Class': module.__class__.__name__,
                'Para': para_sum
            }, index=[0])
            count_para.data = pd.concat([count_para.data, row])
        else:
            for name, sub_model in module.named_children():
                sub_model_name = name if module_name is None else module_name + '.' + name
                stat_para(sub_model, module_name=sub_model_name)

    stat_para(model, None)
    data = count_para.data
    return data


# </editor-fold>

# <editor-fold desc='数据分析'>
# 检查数据分布
def count_distribute(arr, quant_step=None, a_min=None, a_max=None, with_unit=True, with_terminal=True):
    arr = np.array(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return 0, 0
    quant_step = quant_step if quant_step is not None else np.std(arr) / 5
    assert quant_step > 0
    a_min = a_min if a_min is not None else min_val
    a_max = a_max if a_max is not None else max_val

    num_quant = int(np.ceil((a_max - a_min) / quant_step))

    quants = (np.arange(num_quant) + 0.5) * quant_step + a_min
    indexs = np.floor((arr - a_min) / quant_step)
    filter = (indexs >= 0) * (indexs < num_quant)
    indexs, nums_sub = np.unique(indexs[filter], return_counts=True)
    nums = np.zeros(shape=num_quant)
    nums[indexs.astype(np.int32)] = nums_sub
    # 添加端部节点
    if with_terminal:
        quants = np.concatenate([[a_min], quants, [a_max]])
        nums = np.concatenate([[0], nums, [0]])
    # 归一化
    if with_unit:
        nums = nums / np.sum(nums) / quant_step
    return quants, nums


def analyse_cens(whs, centers, whr_thres=4):
    n_clusters = centers.shape[0]
    ratios = whs[:, None, :] / centers[None, :, :]
    ratios = np.max(np.maximum(ratios, 1 / ratios), axis=2)
    markers = ratios < whr_thres
    matched = np.sum(np.any(markers, axis=1))
    aver_mtch = np.mean(np.sum(markers, axis=1))
    # 输出
    print('* Centers --------------')
    for i in range(n_clusters):
        width, height = centers[i, :]
        print('[ %5d' % int(width) + ' , %5d' % int(height) + ' ] --- ' + str(np.sum(markers[:, i])))
    print('* Boxes --------------')
    print('Matched ' + '%5d' % int(matched) + ' / %5d' % int(whs.shape[0]))
    print('Average ' + '%.2f' % aver_mtch + ' box per obj')
    print('* -----------------------')
    return True


def cluster_wh(whs, n_clusters=9, log_metric=True):
    if log_metric:
        whs_log = np.log(whs)
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(whs_log)
        centers_log = kmeans_model.cluster_centers_
        centers = np.exp(centers_log)
    else:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(whs)
        centers = kmeans_model.cluster_centers_
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    centers_sorted = centers[order]
    return centers_sorted


# </editor-fold>


# 得到device
def get_device(model):
    if hasattr(model, 'device'):
        return model.device
    else:
        if len(model.state_dict()) > 0:
            return next(iter(model.parameters())).device
        else:
            return torch.device('cpu')


# 规范4维输入
def _get_input_size(input_size=(32, 32), default_channel=3, default_batch=1):
    if isinstance(input_size, int):
        input_size = (default_batch, default_channel, input_size, input_size)
    elif len(input_size) == 1:
        input_size = (default_batch, default_channel, input_size[0], input_size[0])
    elif len(input_size) == 2:
        input_size = (default_batch, default_channel, input_size[0], input_size[1])
    elif len(input_size) == 3:
        input_size = (default_batch, input_size[0], input_size[1], input_size[2])
    elif len(input_size) == 4:
        pass
    else:
        raise Exception('err size ' + str(input_size))
    return input_size


def model2onnx(model, onnx_pth, input_size, **kwargs):
    onnx_dir = os.path.dirname(onnx_pth)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_pth = onnx_pth + '.onnx' if not str.endswith(onnx_pth, '.onnx') else onnx_pth
    input_size = _get_input_size(input_size, default_batch=1, default_channel=3)
    # 仅支持单输入单输出
    input_names = ['input']
    output_names = ['output']
    dynamic_batch = input_size[0] is None or input_size[0] < 0
    if dynamic_batch:
        input_size = list(input_size)
        input_size[0] = 1
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        print('Using dynamic batch size')
    else:
        dynamic_axes = None
        print('Exporting static batch size')
    test_input = (torch.rand(size=input_size) - 0.5) * 4
    test_input = test_input.to(get_device(model))
    model.eval()
    print('Exporting onnx to ' + onnx_pth)
    torch.onnx.export(model, test_input, onnx_pth, verbose=True, opset_version=11,
                      operator_export_type=OperatorExportTypes.ONNX, do_constant_folding=True,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    return True


# <editor-fold desc='计数'>

# </editor-fold>

# <editor-fold desc='规范化导出接口'>
class ONNXExportable(nn.Module):

    @property
    @abstractmethod
    def input_names(self):
        pass

    @property
    @abstractmethod
    def output_names(self):
        pass

    @property
    @abstractmethod
    def input_sizes(self):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def export_onnx(self, onnx_pth, dynamic_batch=False):
        onnx_pth = ensure_extend(onnx_pth, 'onnx')
        ensure_file_dir(onnx_pth)
        input_names = self.input_names
        output_names = self.output_names
        input_sizes = self.input_sizes
        device = self.device
        input_sizes = [[1] + list(input_size) for input_size in input_sizes]
        input_tens = tuple([torch.rand(input_size, device=device) for input_size in input_sizes])

        msg = 'Exporting onnx to ' + onnx_pth + ' |'
        if dynamic_batch:
            dynamic_axes = {}
            for input_name in input_names:
                dynamic_axes[input_name] = {0: 'batch_size'}

            for output_name in output_names:
                dynamic_axes[output_name] = {0: 'batch_size'}
            msg += ' <dynamic> batch |'
        else:
            dynamic_axes = None
            msg += ' <static> batch |'

        msg += ' ' + str(input_sizes) + ' |'
        print(msg)
        torch.onnx.export(self, input_tens, onnx_pth, verbose=False, opset_version=11,
                          operator_export_type=OperatorExportTypes.ONNX, do_constant_folding=True,
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
        return self

    def count_flop(self, ignore_zero=False):
        device = self.device
        input_sizes = [[1] + list(input_size) for input_size in self.input_sizes]
        input_tens = tuple([torch.rand(input_size, device=device) for input_size in input_sizes])
        data = count_flop(self, input_tens, ignore_zero=ignore_zero)
        return data

    def count_para(self):
        return count_para(self)


class SISOONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['input']

    @property
    def output_names(self):
        return ['output']


class SequenceONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['sequence']

    @property
    def output_names(self):
        return ['output']

    @property
    @abstractmethod
    def length(self):
        pass

    @property
    @abstractmethod
    def in_features(self):
        pass

    @property
    def input_sizes(self):
        return [(self.length,self.in_features )]


class ImageONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['image']

    @property
    def output_names(self):
        return ['output']

    @property
    @abstractmethod
    def img_size(self):
        pass

    @property
    @abstractmethod
    def in_channels(self):
        pass

    @property
    def input_sizes(self):
        return [(self.in_channels, self.img_size[1], self.img_size[0])]


class GeneratorONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['latvecs']

    @property
    def output_names(self):
        return ['fimages']

    @property
    def input_sizes(self):
        return [(self.in_features,)]

    @property
    @abstractmethod
    def in_features(self):
        pass


# </editor-fold>


# <editor-fold desc='onnx读取工具'>

class ONNXModule():
    def __init__(self, onnx_pth, device=None):
        onnx_pth = onnx_pth + '.onnx' if not str.endswith(onnx_pth, '.onnx') else onnx_pth
        device_ids = select_device(device)
        if device_ids[0] is None:
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_pth, providers=['CPUExecutionProvider'])
        else:
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_pth, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': str(device_ids[0])}])

    @property
    def output_names(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    @property
    def input_names(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name


class OneStageONNXModule(ONNXModule):
    def __init__(self, onnx_pth, device=None):
        super().__init__(onnx_pth=onnx_pth, device=device)
        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        assert len(inputs) == 1, 'fmt err'
        assert len(outputs) == 1, 'fmt err'
        self.input_size = inputs[0].shape
        self.output_size = outputs[0].shape
        print('ONNXModule from ' + onnx_pth + ' * input ' + str(inputs[0].shape) + ' * output ' + str(outputs[0].shape))

    def __call__(self, input, **kwargs):
        input_feed = {self.input_names[0]: input}
        outputs = self.onnx_session.run(self.output_names, input_feed=input_feed)
        output = outputs[0]
        return output
# </editor-fold>
