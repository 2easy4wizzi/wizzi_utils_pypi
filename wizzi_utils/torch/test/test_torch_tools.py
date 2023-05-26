from wizzi_utils.torch import torch_tools as tt
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt
import numpy as np
# noinspection PyPackageRequirements
import torch
# noinspection PyPackageRequirements
import torch.nn as nn
# noinspection PyPackageRequirements
import torch.nn.functional as func


def cuda_on_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\tcuda available ? {}'.format(tt.cuda_on()))
    return


def set_cuda_scope_and_seed_test():
    mt.get_function_name(ack=True, tabs=0)
    tt.set_cuda_scope_and_seed(seed=42, dtype='FloatTensor')
    a = torch.ones(5)
    print(tt.to_str(var=a, title='\t\tones'))
    tt.set_cuda_scope_and_seed(seed=42, dtype='DoubleTensor')
    b = torch.zeros(5)
    print(tt.to_str(var=b, title='\t\tzeros'))
    return


def add_cuda_test():
    mt.get_function_name(ack=True, tabs=0)
    if tt.cuda_on():
        t_var = torch.ones(1).to(torch.device('cpu'))
        print(tt.to_str(var=t_var, title='\tcpu element'))
        t_var = tt.add_cuda(t_var)
        print(tt.to_str(var=t_var, title='\tgpu element'))
    return


def is_trainable_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = torch.ones(1)
    print('\tis t_var trainable ? {}'.format(tt.is_trainable(t_var)))
    tt.set_trainable(t_var, trainable=True)
    print('\tis t_var trainable ? {}'.format(tt.is_trainable(t_var)))
    return


def is_cuda_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = torch.ones(1).to(torch.device('cpu'))
    print('\tis t_var cuda var ? {}'.format(tt.is_cuda(t_var)))
    if tt.cuda_on():
        t_var = torch.ones(1).to(torch.device('cuda:0'))
        print('\tis t_var cuda var ? {}'.format(tt.is_cuda(t_var)))
    return


def size_s_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = torch.ones(size=(200, 2, 3, 4))
    print('\ttorch.size() = {}'.format(t_var.size()))
    print('\ttt.size_s(t_var) = {}'.format(tt.size_s(t_var)))
    return


def total_size_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = torch.ones(size=(200, 2, 3, 4))
    print('\ttorch.size() = {}'.format(t_var.size()))
    print('\ttt.total_size(t_var) = {}'.format(tt.total_size(t_var, ignore_first=True)))
    print('\ttt.total_size(t_var) = {}'.format(tt.total_size(t_var, ignore_first=False)))
    return


def torch_to_numpy_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = torch.ones(size=(200, 2, 3, 4))
    print(tt.to_str(var=t_var, title='\tt_var'))
    t_var = tt.torch_to_numpy(t_var)
    print(tt.to_str(var=t_var, title='\tt_var'))
    return


def numpy_to_torch_test():
    mt.get_function_name(ack=True, tabs=0)
    np_var = np.ones(shape=(200, 2, 3, 4))
    print(tt.to_str(var=np_var, title='\tt_var'))
    np_var = tt.numpy_to_torch(np_var)
    print(tt.to_str(var=np_var, title='\tt_var'))
    return


def to_str_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = tt.torch_normal(shape=(200, 2, 3, 4), miu=3.5, std=1.2)
    print(tt.to_str(var=t_var, title='\tt_var', chars=25, fp=3, rec=True))
    t_var = t_var.double()
    print(tt.to_str(var=t_var, title='\tt_var', chars=25, fp=5, rec=False))
    print(tt.to_str(var=t_var, title='\tt_var', chars=25, fp=5, wm=False, rec=False))
    np_var = tt.torch_to_numpy(t_var)
    print(tt.to_str(var=np_var, title='\tnp_var', chars=25, fp=5, rec=True))
    return


def save_load_tensor_test():
    mt.get_function_name(ack=True, tabs=0)
    # e.g.saving and loading tensor:
    print('\tExample A(tensor):')
    mt.create_dir(mtt.TEMP_FOLDER1)
    path = '{}/{}.pt'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    a = torch.ones(size=(2, 3, 29))
    print(tt.to_str(a, '\t\ta'))
    tt.save_tensor(a, path=path, tabs=2)
    a2 = tt.load_tensor(path, tabs=2)
    print(tt.to_str(a2, '\t\ta2'))
    mt.delete_file(path, tabs=2)

    # e.g.saving and loading tensors dict:
    print('\tExample B(dict):')
    path = '{}/{}.pt'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    b = torch.ones(size=(2, 3, 29))
    c = torch.ones(size=(2, 3, 29))
    b_c = {'b': b, 'c': c}
    print(tt.to_str(b_c, '\t\tb_c'))
    tt.save_tensor(b_c, path=path, tabs=2)
    b_c2 = tt.load_tensor(path, tabs=2)
    print(tt.to_str(b_c2, '\t\tb_c2', rec=True))
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1, tabs=2)
    return


def torch_uniform_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = tt.torch_uniform(shape=(200, 2, 3, 4), range_low=0.3, range_high=1.2)
    print(tt.to_str(var=t_var, title='\tt_var'))
    return


def torch_normal_test():
    mt.get_function_name(ack=True, tabs=0)
    t_var = tt.torch_normal(shape=(200, 2, 3, 4), miu=3.5, std=1.2)
    print(tt.to_str(var=t_var, title='\tt_var'))
    return


def opt_to_str_test():
    mt.get_function_name(ack=True, tabs=0)
    a = tt.torch_normal(shape=(200, 2, 3, 4), miu=3.5, std=1.2)
    opt = torch.optim.Adam([a], lr=0.001, weight_decay=0.0001)
    print('\topt={}'.format(tt.opt_to_str(opt)))
    return


def get_lr_test():
    mt.get_function_name(ack=True, tabs=0)
    a = tt.torch_normal(shape=(200, 2, 3, 4), miu=3.5, std=1.2)
    opt = torch.optim.Adam([a], lr=0.001, weight_decay=0.0001)
    print('\topt lr={}'.format(tt.get_lr(opt)))
    return


def set_lr_test():
    mt.get_function_name(ack=True, tabs=0)
    a = tt.torch_normal(shape=(200, 2, 3, 4), miu=3.5, std=1.2)
    opt = torch.optim.Adam([a], lr=0.001, weight_decay=0.0001)
    print('\topt lr={}'.format(tt.get_lr(opt)))
    tt.set_lr(opt, new_lr=0.2)
    print('\topt lr={}'.format(tt.get_lr(opt)))
    return


def get_opt_by_name_test():
    mt.get_function_name(ack=True, tabs=0)
    a = tt.torch_normal(shape=(200, 2, 3, 4), miu=3.5, std=1.2)
    opt = tt.get_opt_by_name(
        opt_d={'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0},
        params=[a]
    )
    print('\topt={}'.format(tt.opt_to_str(opt)))

    opt = tt.get_opt_by_name(
        opt_d={'name': 'ADAM', 'lr': 0.001, 'weight_decay': 0.0001},  # L2 regularization
        params=[a]
    )
    print('\topt={}'.format(tt.opt_to_str(opt)))

    opt = tt.get_opt_by_name(
        opt_d={'name': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0},
        params=[a]
    )
    print('\topt={}'.format(tt.opt_to_str(opt)))

    opt = tt.get_opt_by_name(
        opt_d={'name': 'SGD', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0001},  # L2 regularization
        params=[a]
    )
    print('\topt={}'.format(tt.opt_to_str(opt)))
    return


def OptimizerHandler_test():
    mt.get_function_name(ack=True, tabs=0)

    A = tt.torch_normal(shape=(6, 2), miu=3.5, std=1.2)
    q = tt.torch_normal(shape=(2, 1), miu=3.5, std=1.2)
    tt.set_trainable(A, trainable=True)
    optimizer = torch.optim.Adam([A], lr=0.1, weight_decay=0.0001)
    optimizer = tt.OptimizerHandler(optimizer, factor=0.1, patience=2, min_lr=0.000001)

    for epoch in range(0, 10):
        print('\tepoch {} - lr = {:.5f}'.format(epoch, tt.get_lr(optimizer)))
        optimizer.zero_grad()
        # noinspection PyCompatibility
        loss = (((A[:, :-1] @ q[:-1] + q[-1]).view(A.shape[0]) - A[:, -1]) ** 2).sum()
        loss.backward()
        optimizer.step()
        optimizer.update_lr()  # <- this counts the epochs and tries to change the lr
        # update_lr add +1 to counter. if counter >= patience: counter = 0 and new_lr= max(old_lr * factor, min_lr)
    return


def EarlyStopping_test():
    mt.get_function_name(ack=True, tabs=0)
    es = tt.EarlyStopping(patience=3)
    best_loss = 100000000
    loss = None

    for epoch in range(0, 10):
        if epoch == 0:
            loss = 100
        if epoch > 0:
            loss = 50
        if loss < best_loss:
            best_loss = loss
            print('\tbest_loss = {} - improved'.format(best_loss))
        else:
            print('\tbest_loss = {} - no improvement found on epoch {}'.format(best_loss, epoch))
        if es.should_early_stop(loss):
            print('\tEarly stopping on epoch {} out of 10'.format(epoch))
            break
    return


def subset_init_test():
    mt.get_function_name(ack=True, tabs=0)
    A = tt.torch_normal(shape=(6, 2), miu=3.5, std=1.2)
    C = tt.subset_init(c_size=3, A=A, trainable=False)
    C2 = tt.subset_init(c_size=1, A=A, trainable=True)
    print(tt.to_str(A, '\tA'))
    print(tt.to_str(C, '\tC'))
    print(tt.to_str(C2, '\tC2'))
    return


def augment_x_y_torch_test():
    mt.get_function_name(ack=True, tabs=0)
    X = tt.torch_normal(shape=(6, 2), miu=3.5, std=1.2)
    Y = tt.torch_normal(shape=(6,), miu=3.5, std=1.2)
    A = tt.augment_x_y_torch(X, Y)
    print(tt.to_str(X, '\tX'))
    print(tt.to_str(Y, '\tY'))
    print(tt.to_str(A, '\tA'))
    return


def de_augment_torch_test():
    mt.get_function_name(ack=True, tabs=0)
    A = tt.torch_normal(shape=(6, 3), miu=3.5, std=1.2)
    X, Y = tt.de_augment_torch(A)
    print(tt.to_str(A, '\tA'))
    print(tt.to_str(X, '\tX'))
    print(tt.to_str(Y, '\tY'))
    return


def split_tensor_test():
    mt.get_function_name(ack=True, tabs=0)
    A = tt.torch_normal(shape=(100, 3), miu=3.5, std=1.2)
    A1, A2 = tt.split_tensor(A, p=0.81)
    print(tt.to_str(A, '\tA'))
    print(tt.to_str(A1, '\tA1'))
    print(tt.to_str(A2, '\tA2'))
    return


def shuffle_tensor_test():
    mt.get_function_name(ack=True, tabs=0)
    A = tt.torch_normal(shape=(6,), miu=3.5, std=10.2)
    print(tt.to_str(A, '\tA'))
    A = tt.shuffle_tensor(A)
    print(tt.to_str(A, '\tA'))
    return


def shuffle_tensors_test():
    mt.get_function_name(ack=True, tabs=0)
    A = np.array([1, 2, 3, 4, 5, 6])
    A = tt.numpy_to_torch(A)
    B = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    print(tt.to_str(A, '\tA'))
    print(tt.to_str(B, '\tB'))
    A, B = tt.shuffle_tensors(
        arr_tuple=(A, B)
    )
    print(tt.to_str(A, '\tA'))
    print(tt.to_str(B, '\tB'))
    return


def count_keys_test():
    mt.get_function_name(ack=True, tabs=0)
    data_root = mtt.DATASETS
    mt.create_dir(data_root, ack=True)
    ds = tt.get_data_set('cifar10', data_root, is_train=False, download=True)
    tt.count_keys(ds.targets, tabs=1)
    ds = tt.get_data_set('mnist', data_root, is_train=False, download=True)
    tt.count_keys(ds.targets, tabs=1)
    return


def get_torch_version_test():
    mt.get_function_name(ack=True, tabs=0)
    tt.get_torch_version(ack=True, tabs=1)
    return


class MnistExample(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self):
        super(MnistExample, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = func.log_softmax(x, dim=1)
        return output


def save_load_model_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    tt.layer_info(net1, layer='conv1', title='net1 conv1 values', wv=True, tabs=1)
    mt.create_dir(mtt.TEMP_FOLDER1)
    path = '{}/{}.pt'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    tt.save_model(net1, path=path)

    net2 = MnistExample()
    if tt.cuda_on():
        net2 = net2.cuda()

    tt.layer_info(net2, layer='conv1', title='net2 conv1 values', wv=True, tabs=1)
    tt.load_model(net2, path)
    tt.layer_info(net2, layer='conv1', title='net2 conv1 values(after load)', wv=True, tabs=1)
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1, tabs=2)
    return


def model_info_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    tt.model_info(net1, 'Mnist model', wv=True, tabs=1)
    return


def layer_info_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    tt.layer_info(net1, layer='conv2', wv=True, tabs=1)
    return


def model_summary_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample().float()  # if cuda_on(): model_summary need float variables
    tt.model_summary(net1, input_size=(1, 28, 28), batch_size=256)
    return


def model_params_count_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    print('\ttotal params {:,}'.format(tt.model_params_count(net1)))
    return


def model_status_str_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    print('\tstatus = {}'.format(tt.model_status_str(net1)))
    return


def set_model_status_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    print('\tstatus = {}'.format(tt.model_status_str(net1)))
    tt.set_model_status(net1, status=False)
    print('\tstatus = {}'.format(tt.model_status_str(net1)))
    tt.set_model_status(net1, status=True)
    print('\tstatus = {}'.format(tt.model_status_str(net1)))
    return


def set_layer_status_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    tt.layer_info(net1, layer='conv2', title='net2')
    tt.set_layer_status(net1, layers=['conv2'], status=False)
    tt.model_info(net1, title='net2(post freezing conv2)')
    return


def model_clone_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    print('\tnet1 address {}'.format(hex(id(net1))))
    net2 = tt.model_clone(net1)
    print('\tnet2 address {}'.format(hex(id(net2))))
    return


def model_copy_layers_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    net2 = MnistExample()
    tt.model_info(net1, title='net1', wv=True, tabs=1)
    tt.model_info(net2, title='net2', wv=True, tabs=1)
    tt.model_copy_layers(net1, net2, layers=['conv1'])
    tt.model_info(net2, title='net2(after copy conv1)', wv=True, tabs=1)
    return


def model_copy_except_layers_test():
    mt.get_function_name(ack=True, tabs=0)
    net1 = MnistExample()
    net2 = MnistExample()
    tt.model_info(net1, title='net1', wv=True, tabs=1)
    tt.model_info(net2, title='net2', wv=True, tabs=1)
    tt.model_copy_except_layers(net1, net2, layers=['conv1'])
    tt.model_info(net2, title='net2(after copy all net1 except conv1)', wv=True, tabs=1)
    return


def size_s_ds_test():
    mt.get_function_name(ack=True, tabs=0)
    data_root = mtt.DATASETS
    mt.create_dir(data_root, ack=True)
    tt.get_data_set('cifar10', data_root, is_train=False, download=True)
    tt.get_data_set('mnist', data_root, is_train=False, download=True)
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    cuda_on_test()
    set_cuda_scope_and_seed_test()
    add_cuda_test()
    is_trainable_test()
    is_cuda_test()
    size_s_test()
    total_size_test()
    torch_to_numpy_test()
    numpy_to_torch_test()
    to_str_test()
    save_load_tensor_test()
    torch_uniform_test()
    torch_normal_test()
    opt_to_str_test()
    get_lr_test()
    set_lr_test()
    get_opt_by_name_test()
    OptimizerHandler_test()
    EarlyStopping_test()
    subset_init_test()
    augment_x_y_torch_test()
    de_augment_torch_test()
    split_tensor_test()
    shuffle_tensor_test()
    shuffle_tensors_test()
    count_keys_test()
    get_torch_version_test()
    save_load_model_test()
    model_info_test()
    layer_info_test()
    model_summary_test()
    model_params_count_test()
    model_status_str_test()
    set_model_status_test()
    set_layer_status_test()
    model_clone_test()
    model_copy_layers_test()
    model_copy_except_layers_test()
    size_s_ds_test()
    print('{}'.format('-' * 20))
    return
