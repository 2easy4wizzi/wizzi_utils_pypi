import numpy as np
import os
import copy
from wizzi_utils import misc_tools as mt  # misc tools

# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
import torch.nn as nn
# noinspection PyPackageRequirements
import torchvision

# found that these are the most used transforms for mnist and cifar10
DEFAULT_MNIST_TR = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

DEFAULT_CIFAR10_TR_TRAIN = torchvision.transforms.Compose([
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

DEFAULT_CIFAR10_TR_TEST = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


def cuda_on() -> bool:
    """
     check if cuda available
     see cuda_on_test()
     """
    return torch.cuda.is_available()


def set_cuda_scope_and_seed(seed: int, dtype='FloatTensor', tabs: int = 1) -> None:
    """
    :param seed: setting torch seed and default torch if cuda on
    :param dtype:
        https://pytorch.org/docs/stable/tensors.html
        32-bit floating point: torch.cuda.FloatTensor
        64-bit floating point: torch.cuda.DoubleTensor
    :param tabs:
    see set_cuda_scope_and_seed_test()
    """
    mt.set_seed(seed)
    if cuda_on():
        def_dtype = 'torch.cuda.{}'.format(dtype)
        torch.set_default_tensor_type(def_dtype)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        print('{}working on CUDA. default dtype = {} <=> {}'.format(tabs * '\t', def_dtype, torch.get_default_dtype()))
    else:
        torch.manual_seed(seed)
        print('{}working on CPU'.format(tabs * '\t'))
    return


def add_cuda(var: torch.Tensor) -> torch.Tensor:
    """
    assigns the variables to GPU if available
    see add_cuda_test()
    """
    if cuda_on() and not is_cuda(var):
        var = var.cuda()
    return var


def is_trainable(var: torch.Tensor) -> bool:
    """
    :param var:
    :return: if the variable is trainable
    see is_trainable_test()
    """
    return var.requires_grad


def set_trainable(var: torch.Tensor, trainable: bool = True) -> None:
    """
    :param var:
    :param trainable: True/False
    sets the var to 'trainable'
    """
    var.requires_grad = trainable
    return


def is_cuda(var: torch.Tensor) -> bool:
    """
    :param var:
    :return: if the variable is Cuda/CPU variable
    see is_cuda_test()
    """
    # noinspection PyTypeChecker
    return var.is_cuda


def size_s(var: torch.Tensor) -> str:
    """
    :param var:
    :return: clean str of tensor size
    e.g. torch.Size([1, 3, 29]) -> [1, 3, 29]
    see size_s_test()
    """
    size_str = str(var.size())
    size_str = size_str[size_str.find("(") + 1:size_str.find(")")]
    return size_str


def total_size(t: torch.Tensor, ignore_first=True) -> int:
    """
    :param t:
    :param ignore_first: if the first dim is the rows and you want each row size
    calculates the total size of a tensor.
    see total_size_test()
    """
    total = 1
    my_shape = t.shape[1:] if ignore_first else t.shape
    for d in my_shape:
        total *= d
    return total


def torch_to_numpy(var_torch: torch.Tensor) -> np.array:
    """
    :param var_torch:
    :return: np array of var_torch
    convert torch to numpy
    see torch_to_numpy_test()
    """
    if is_trainable(var_torch):
        var_np = var_torch.detach().cpu().numpy()
    else:
        var_np = var_torch.cpu().numpy()
    return var_np


def numpy_to_torch(var_np: np.array, to_double=True, detach: bool = False) -> torch.Tensor:
    """
    :param var_np:
    :param to_double: create torch double - else float
    :param detach: if needed to be detached - don't remember when detached needed to be true happened
    convert numpy to torch also add to cude if cuda is on
    see numpy_to_torch_test()
    """
    if detach:
        if to_double:
            var_torch = add_cuda(torch.from_numpy(var_np).double()).detach()
        else:
            var_torch = add_cuda(torch.from_numpy(var_np).float()).detach()
    else:
        if to_double:
            var_torch = add_cuda(torch.from_numpy(var_np).double())
        else:
            var_torch = add_cuda(torch.from_numpy(var_np).float())
    return var_torch


def to_str(var: any,
           title: str = 'var',
           chars: int = 100,
           fp: int = 2,
           wm: bool = True,
           rec: bool = False
           ) -> str:
    """
    :param var: the variable
    :param title: str: the title (usually variable name)
    :param chars: int, None or str:
        chars>0: maximal number of chars
        chars==None: no chars
        chars=='all': all chars
    :param fp: float_precision: round number if possible(float, list or np array of floats...)
            fp>0 round
            fp==None: no rounding
    :param wm: with_meta: with metadata such as type, len/shape, dtype...
    :param rec: recursive: to keep printing if there are more items inside e.g. np.array(shape=(2,3,4)) -> 3 prints
    :return: informative string of the variable
    see to_str_test() on test_torch_tools.py
    """

    string = title
    type_s = str(type(var)).replace('<class \'', '').replace('\'>', '')  # clean type name

    if is_torch(var):
        if wm:
            string += '({}'.format(type_s)
            string += ',s={}'.format(var.shape)
            string += ',dtype={}'.format(var.dtype)
            string += ',trainable={}'.format(is_trainable(var))
            string += ',is_cuda={})'.format(is_cuda(var))

        if chars != 0:  # -1 for all, or x>0 for x chars
            # new_v = var.tolist()
            if fp >= 0 and len(var.size()) > 0:
                if all((mt.is_int(item) or mt.is_float(item)) for item in var):  # 1d list of int and floats
                    if all(mt.is_int(item) for item in var):  # if all ints - no rounding
                        f_format = '{:,}'
                    else:
                        f_format = '{:,.%xf}' % fp
                    new_v = [f_format.format(li_item) for li_item in var]
                else:  # >1d list or 1d with not just ints and floats
                    new_v = np.around(var.tolist(), fp).tolist()
            else:
                new_v = var
            data_str_raw = str(new_v).replace('\'', '')
            string += mt.get_data_str(data_str_raw, chars)

        if rec and len(var.size()) > 0:  # recursive call
            inner_str = to_str(var=var[0], title='{}[0]'.format(title), chars=chars, fp=fp, rec=rec)
            string += '\n\t{}'.format(inner_str)
    else:  # if it's not Tensor - use the default to_str() from misc_tools
        string = mt.to_str(var, title=title, chars=chars, wm=wm, fp=fp, rec=rec)
    return string


def save_tensor(t, path: str, ack_print: bool = True, tabs: int = 1):
    """
    :param t: dict or a tensor
    :param path:
    :param ack_print:
    :param tabs:

    see save_load_tensor_test()
    """
    torch.save(t, path)
    if ack_print:
        size_str = mt.file_or_folder_size(path)
        file_msg = '{}({})'.format(path, size_str)
        print('{}{}'.format(tabs * '\t', mt.SAVED.format(file_msg)))
    return


def load_tensor(path: str, ack_print: bool = True, tabs: int = 1):
    """
    :param path:
    :param ack_print:
    :param tabs:
    :return: t: dict or a tensor
    see save_load_tensor_test()
    """
    if os.path.exists(path):
        dst_allocation = None if cuda_on() else 'cpu'
        t = torch.load(path, map_location=dst_allocation)
        if ack_print:
            size_str = mt.file_or_folder_size(path)
            file_msg = '{}({})'.format(path, size_str)
            print('{}{}'.format(tabs * '\t', mt.LOADED.format(file_msg)))
    else:
        mt.exception_error(mt.NOT_FOUND.format(path), real_exception=False, tabs=tabs)
        t = None
    return t


def torch_uniform(shape: tuple, range_low: float, range_high: float) -> torch.Tensor:
    """
    :param shape:
    :param range_low:
    :param range_high:
    :return:
    see torch_uniform_test()
    """
    # noinspection PyArgumentList
    ret = torch.empty(shape).uniform_(range_low, range_high)
    return ret


def torch_normal(shape: tuple, miu: float, std: float) -> torch.Tensor:
    """
    :param shape:
    :param miu:
    :param std:
    :return:
    see torch_normal_test()
    """
    ret = torch.empty(shape).normal_(miu, std)
    return ret


def opt_to_str(optimizer: torch.optim) -> str:
    """
    :param optimizer:
    :return:
    see opt_to_str_test()
    """
    opt_s = str(optimizer).replace('\n', '').replace('    ', ' ')
    return opt_s


def get_lr(optimizer: [torch.optim, int]) -> float:
    """
    :param optimizer:
    :return:
    see get_lr_test()
    """
    lr = None
    if isinstance(optimizer, OptimizerHandler):
        lr = optimizer.lr()
    else:  # isinstance(optimizer, torch.Tensor):
        # noinspection PyUnresolvedReferences
        if "lr" in optimizer.param_groups[0]:
            # noinspection PyUnresolvedReferences
            lr = optimizer.param_groups[0]['lr']
    return lr


def set_lr(optimizer: torch.optim, new_lr: float) -> None:
    """
    :param optimizer:
    :param new_lr:
    :return:
    see set_lr_test()
    """
    optimizer.param_groups[0]['lr'] = new_lr
    return


def get_opt_by_name(opt_d: dict, params: list) -> torch.optim:
    """
    :param params: trainable params
    :param opt_d: has keys: name, lr, weight_decay and optional if SGD momentum
        e.g. options: {'name'= 'ADAM', 'lr': 0.001, 'weight_decay': 0}
    :return: optimizer
    see get_opt_by_name_test()
    """
    opt = None
    if opt_d['name'] == 'ADAM':
        opt = torch.optim.Adam(params, lr=opt_d['lr'], weight_decay=opt_d['weight_decay'])
    elif opt_d['name'] == 'SGD':
        opt = torch.optim.SGD(params, lr=opt_d['lr'], momentum=opt_d['momentum'], weight_decay=opt_d['weight_decay'])
    return opt


class OptimizerHandler:
    """
    Optimizer wrapper.
    if update_lr() called 'patience' times:
        new_lr = max(factor * lr, min_lr)
        set_lr(opt, new_lr)
    see OptimizerHandler_test()
    """

    def __init__(self, optimizer: torch.optim, factor: float, patience: int, min_lr: float):
        """
        :param optimizer: class torch.optim: e.g. Adam, SGD ...
        :param factor: percent to keep from old lr. if factor 0.1 and lr was 5: new lr = 0.5
        :param patience: how long to wait before changing lr
        :param min_lr: minimal lr
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.epochs_passed = 0

    def step(self):
        self.optimizer.step()
        return

    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, new_lr: float):
        self.optimizer.param_groups[0]['lr'] = new_lr
        return

    def zero_grad(self):
        self.optimizer.zero_grad()
        return

    def update_lr(self):
        self.epochs_passed += 1
        if self.epochs_passed >= self.patience:
            self.epochs_passed = 0
            old_lr = self.lr()
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.set_lr(new_lr)
            # print('new lr changed to {}'.format(self.lr()))
        return


class EarlyStopping:
    """
    counts how many rounds no improvment in loss
    if that passed patience - return true in should_early_stop()
    see EarlyStopping_test()
    """

    def __init__(self, patience: int):
        """
        :param patience:
        """
        self.patience = patience
        self.counter = 0
        self.best = None
        return

    def should_early_stop(self, loss: float):
        """
        :param loss:
        :return:
        """
        should_stop = False
        if self.best is None:
            self.best = loss
        elif loss < self.best:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
            # print('\t\tpatience {}/{}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                should_stop = True
        return should_stop


def subset_init(c_size: int, A: torch.Tensor, trainable: bool = True) -> torch.Tensor:
    """
    :param c_size:
    :param A:
    :param trainable:
    :return:
    given c_size <= |A|, initialize a tensor C with a random subset of A
    see subset_init_test()
    """
    n = A.shape[0]
    if c_size >= n:
        if A.dtype == 'torch.float64':
            C = A.clone().double()
        elif A.dtype == 'torch.float32':
            C = A.clone().float()
        else:
            C = A.clone()
    else:
        perm = np.random.permutation(n)
        idx = perm[:c_size]
        if A.dtype == 'torch.float64':
            C = A[idx].clone().double()
        elif A.dtype == 'torch.float32':
            C = A[idx].clone().float()
        else:
            C = A[idx].clone()

    set_trainable(C, trainable)
    return C


def augment_x_y_torch(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param X:
    :param y:
    :return:
    creates A=X|y
    see augment_x_y_torch_test()
    """
    assert X.shape[0] == y.shape[0], 'row count must be the same'
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.view(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.view(y.shape[0], 1)
    A = torch.cat((X, y), 1)
    return A


def de_augment_torch(A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    :param A:
    :return:
    creates X|y=A
    see de_augment_torch_test()
    """
    assert 0 < len(A.shape) <= 2, 'supports 2d only'
    if len(A.shape) == 1:  # A is 1 point. change from size (n) to size (1,n)
        A = A.view(1, A.shape[0])
    X, y = A[:, :-1], A[:, -1]
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.view(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.view(y.shape[0], 1)
    return X, y


def split_tensor(Q: torch.Tensor, p: float = 0.9) -> (torch.Tensor, torch.Tensor):
    """
    :param Q:
    :param p:
    :return:
    see split_tensor_test()
    """
    partition = int(p * Q.shape[0])
    Q_1 = Q[:partition]
    Q_2 = Q[partition:]
    return Q_1, Q_2


def shuffle_tensor(arr: torch.Tensor) -> torch.Tensor:
    """
    :param arr:
    :return:
    shuffles an array
    see shuffle_tensor_test()
    """
    if is_torch(arr):
        arr = arr[torch.randperm(arr.shape[0])]
    return arr


def shuffle_tensors(arr_tuple: tuple) -> tuple:
    """
    :param arr_tuple: tuple of arrays (numpy or tensor)
        len(arr) is equal on all arrays
    :return: shuffled arrays
    see shuffle_tensors_test()
    """
    arrays_size = len(arr_tuple[0])
    # rand_perm = torch.randperm(arrays_size)
    rand_perm = np.random.permutation(arrays_size)

    out_tuple = ()
    for arr in arr_tuple:
        arr_shf = arr[rand_perm]
        out_tuple += (arr_shf,)
    return out_tuple


def is_torch(var: any) -> bool:
    # noinspection PyTypeChecker
    return isinstance(var, torch.Tensor)


def count_keys(y: [torch.Tensor, np.array, list], tabs: int = 1) -> None:
    """
    :param y: nx1 array (torch, list, numpy)
    :param tabs:
    see count_keys_test()
    """
    from collections import Counter
    if hasattr(y, "shape"):
        y_shape = y.shape
    else:
        y_shape = len(y)
    print('{}Count classes: (y shape {})'.format(tabs * '\t', y_shape))
    cnt = Counter()
    for value in y:
        ind = value.item() if is_torch(y) else value
        cnt[ind] += 1
    cnt = sorted(cnt.items())
    for item in cnt:
        print('{}\tClass {}: {} samples'.format(tabs * '\t', item[0], item[1]))
    return


def get_torch_version(ack: bool = False, tabs: int = 1) -> str:
    """
    :return:
    see get_torch_version_test()
    """
    # noinspection PyUnresolvedReferences
    string = mt.add_color('{}* PyTorch Version {}'.format(tabs * '\t', torch.__version__), ops=mt.SUCCESS_C)
    string += mt.add_color(' - GPU detected ? ', ops=mt.SUCCESS_C)
    if cuda_on():
        string += mt.add_color('True', ops=mt.SUCCESS_C2)
    else:
        string += mt.add_color('False', ops=mt.FAIL_C2)
    if ack:
        print(string)
    return string


def load_model(model: nn.Module, path: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param model: nn model
    :param path: checkpoint path - ends with .pt or .pth
    :param ack:
    :param tabs:
    :return: None - by reference
    see save_load_model_test()
    """
    if os.path.exists(path):
        dst_allocation = None if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(path, map_location=dst_allocation))
        model.eval()
        if ack:
            size_str = mt.file_or_folder_size(path)
            file_msg = '{}({})'.format(path, size_str)
            print('{}{}'.format(tabs * '\t', mt.LOADED.format(file_msg)))
    else:
        mt.exception_error(mt.NOT_FOUND.format(path), real_exception=False, tabs=tabs)
    return


def save_model(model: nn.Module, path: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param model: nn model to save
    :param path: checkpoint path - ends with .pt or .pth
    :param ack:
    :param tabs:
    :return:
    see save_load_model_test()
    """
    torch.save(model.state_dict(), path)
    if ack:
        size_str = mt.file_or_folder_size(path)
        file_msg = '{}({})'.format(path, size_str)
        print('{}{}'.format(tabs * '\t', mt.SAVED.format(file_msg)))
    return


def model_info(model: nn.Module, title: str = 'model', wv: bool = False, chars: int = 200, tabs: int = 1):
    """
    :param model: nn model with self.title member
    :param title:
    :param wv: with values: print variables values
    :param chars: if wv: prints first 'chars' for each variable
    :param tabs:
    :return:
    see model_info_test()
    """
    print('{}{}:'.format(tabs * '\t', title))
    var_meta = '{}\t{}: {}({:,.0f} params), dtype:{}, trainable:{}, is_cuda:{}'
    sum_params = 0
    for name, param in model.named_parameters():
        layer_params = total_size(param, ignore_first=False)
        sum_params += layer_params
        print(var_meta.format(tabs * '\t', name, size_s(param), layer_params, param.dtype, is_trainable(param),
                              is_cuda(param)))
        # print('{}\t{}'.format(tabs * '\t', to_str(var=param, title=name)))
        if wv:
            print('{}\t\t{}'.format(tabs * '\t', mt.to_str(param.tolist(), 'values', wm=False, chars=chars)))
    print('{}\tTotal {:,.0f} params'.format(tabs * '\t', sum_params))
    return


def layer_info(model: nn.Module, layer: str, title: str = 'layer', wv: bool = False, chars: int = 200, tabs: int = 1):
    """
    :param model: nn model with self.title member
    :param layer: layer partial name
        e.g. 'conv2' -> params conv2.weight conv2.bias will be printed
    :param title:
    :param wv: with values: print variables values
    :param chars: if wv: prints first 'chars' for each variable
    :param tabs:
    :return:
    see layer_info_test()
    """
    print('{}{}:'.format(tabs * '\t', title))
    var_meta = '{}\t{}: {}({:,.0f} params), dtype:{}, trainable:{}, is_cuda:{}'
    for name, param in model.named_parameters():
        if layer in name:
            layer_params = total_size(param, ignore_first=False)
            print(var_meta.format(tabs * '\t', name, size_s(param), layer_params, param.dtype, is_trainable(param),
                                  is_cuda(param)))
            if wv:
                print('{}\t\t{}'.format(tabs * '\t', mt.to_str(param.tolist(), 'values', wm=False, chars=chars)))
    return


def model_summary(model: nn.Module, input_size: tuple, batch_size: int) -> None:
    """
    prints model summary
    # if cuda_on(): model_summary need float variables
    e.g.
        m = MnistModel().float()
        tt.model_summary(m, (1, 28, 28), 64))
    see model_summary_test()
    """
    try:
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from torchsummary import summary
        summary(model, input_size, batch_size)
    except (ModuleNotFoundError, ImportError) as e:
        mt.exception_error(e, real_exception=True)
    return


def model_params_count(model: nn.Module) -> int:
    """
    :param model:
    :return: counts parameters
    see model_params_count_test()
    """
    total_parameters = 0
    for p in list(model.parameters()):
        total_parameters += total_size(p, False)
    return total_parameters


def model_status_str(model: nn.Module) -> str:
    """
    3 options: model fully trainable, fully frozen, both
    see model_status_str_test()
    """
    saw_trainable, saw_frozen = False, False
    for param in model.parameters():
        if is_trainable(param):
            saw_trainable = True
        else:
            saw_frozen = True

    if saw_frozen and saw_trainable:
        msg = 'partly trainable and partly frozen'
    elif saw_trainable:
        msg = 'fully trainable'
    else:
        msg = 'fully frozen'
    return msg


def set_model_status(model: nn.Module, status: bool) -> None:
    """
    :param model:
    :param status:
    :return:
    set model parameters trainable status to 'status'
    see set_model_status_test()
    """
    for param in model.parameters():
        set_trainable(param, status)
    return


def set_layer_status(model: nn.Module, layers: list, status: bool) -> None:
    """
    :param model:
    :param layers: part of the name of the layer in the model.
        e.g. 'conv2' -> params conv2.weight conv2.bias will change trainable status to `status`
    :param status:
    :return:

    see set_layer_status_test()
    """
    for name, param in model.named_parameters():
        for layer in layers:
            if layer in name:
                param.requires_grad_(status)
                break
    return


def model_clone(model: nn.Module) -> nn.Module:
    """
    :param model: deep copy model
    :return:
    see model_clone_test()
    """
    m_clone = copy.deepcopy(model)
    return m_clone


def model_copy_layers(model_src: nn.Module, model_dst: nn.Module, layers: list) -> None:
    """
    :param model_src: assuming same size as model_dst
    :param model_dst: assuming same size as model_src
    :param layers:
    :return:
    see model_copy_layers_test()
    """
    for name, param in model_src.named_parameters():
        current_should_be_copied = any(layer in name for layer in layers)
        if current_should_be_copied:
            model_dst.state_dict()[name].copy_(param)
    return


def model_copy_except_layers(model_src: nn.Module, model_dst: nn.Module, layers: list) -> None:
    """
    :param model_src:
    :param model_dst:
    :param layers:
    :return:
    see model_copy_layer_test()
    """
    for name, param in model_src.named_parameters():
        current_should_be_copied = all(layer not in name for layer in layers)
        if current_should_be_copied:
            model_dst.state_dict()[name].copy_(param)
    return


def size_s_ds(ds: torchvision.datasets) -> str:
    """
    :param ds:
    :return:
    see size_s_ds_test()
    """
    ds_len = len(ds)

    x = ds.data[0]  # load 1st sample as data loader will load
    X_size_post_tr = (ds_len,)
    for d in x.shape:
        X_size_post_tr += (d,)

    y_size = (len(ds.targets),)  # real data
    res = '|X|={}, |y|={}'.format(X_size_post_tr, y_size)
    return res


def get_data_set(
        ds_name: str,
        data_root: str,
        is_train: bool,
        transform=None,
        download: bool = False,
        data_limit: int = None,
        ack: bool = True,
        tabs: int = 1
) -> torchvision.datasets:
    """
    :param ds_name: 'mnist' of 'cifar10'
    :param data_root: folder containing the ds.
        if not containing and download == True: download to data_root
    :param is_train: train(True) or test
    :param transform: transform to images
    :param download: if to download if not found
    :param data_limit: good for debugging - shrinking the dataset from X to Y s.t. Y<X
    :param ack:
    :param tabs:
    :return:
    """
    if ds_name == 'cifar10':
        if transform is None:  # default cifar10 transforms
            transform = DEFAULT_CIFAR10_TR_TRAIN if is_train else DEFAULT_CIFAR10_TR_TEST
        data_set = torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=download, transform=transform)
    elif ds_name == 'mnist':
        if transform is None:  # default mnist transform
            transform = DEFAULT_MNIST_TR
        data_set = torchvision.datasets.MNIST(root=data_root, train=is_train, download=download, transform=transform)
    else:
        mt.exception_error('data_set not valid!', real_exception=False)
        return
    if data_limit is not None:
        data_set.data = data_set.data[:data_limit]
        data_set.targets = data_set.targets[:data_limit]

    if ack:
        prefix = 'train' if is_train else 'test'
        print('{}{}({}) dataset: {}'.format(tabs * '\t', ds_name, prefix, size_s_ds(data_set)))
    return data_set
