from wizzi_utils.misc import misc_tools as mt
import numpy as np
import os

PLAY_GROUND = './wizzi_utils_playground'

IMAGES_PATH = '{}/images'.format(PLAY_GROUND)
IMAGES_INPUTS = '{}/Input'.format(IMAGES_PATH)
IMAGES_OUTPUTS = '{}/Output'.format(IMAGES_PATH)

VIDEOS_PATH = '{}/videos'.format(PLAY_GROUND)
VIDEOS_INPUTS = '{}/Input'.format(VIDEOS_PATH)
VIDEOS_OUTPUTS = '{}/Output'.format(VIDEOS_PATH)

MODELS = '{}/models'.format(PLAY_GROUND)
DATASETS = '{}/datasets'.format(PLAY_GROUND)
TEMP_FOLDER1 = '{}/temp_folder1'.format(PLAY_GROUND)
TEMP_FOLDER2 = '{}/temp_folder2'.format(PLAY_GROUND)

DEMO_FILE = 'demo_file.txt'
JUST_A_NAME = 'just_a_name'
FAKE_FILE = 'fake_file'

SO_LOGO = 'so_logo'
KITE = 'kite'  # (900, 1352, 3)
GIRAFFE = 'giraffe'
HORSES = 'horses'
DOG = 'dog'  # (576, 768, 3)
EAGLE = 'eagle'  # (512, 773, 3)
PERSON = 'person'  # (424, 640, 3)
DOGS1 = 'dogs1'
FACES = 'faces'
F_MODEL = 'model2'
HAND = 'hand'

# sources: https://github.com/pjreddie/darknet/blob/master/data/
IMAGES_D = {
    SO_LOGO: 'https://cdn.sstatic.net/Sites/stackoverflow/img/logo.png',
    KITE: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg',
    GIRAFFE: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/giraffe.jpg',
    HORSES: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg',
    DOG: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg',
    EAGLE: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg',
    PERSON: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg',
    DOGS1: 'https://media.blogto.com/articles/20210415-dog-parks-toronto.jpg' +
           '?w=1200&cmd=resize_then_crop&height=630&quality=70',
    FACES: 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/' +
           'Young_People_in_Park_-_Stepanakert_-_Nagorno-Karabakh_%2819082779012%29.jpg/' +
           '640px-Young_People_in_Park_-_Stepanakert_-_Nagorno-Karabakh_%2819082779012%29.jpg',
    # F_L: 'https://github.com/vladmandic/blazepose/raw/fe647445507e37469d96da6fde5c8b0980f745bc/outputs/model2.jpg',
    F_MODEL: 'https://github.com/vladmandic/blazepose/raw/fe647445507e37469d96da6fde5c8b0980f745bc/inputs/model2.jpg',
    HAND: 'http://clipart-library.com/images/8TxrGaBgc.jpg',
}

DOG1 = 'dog_in_a_field'
WOMAN_YOGA = 'woman_yoga'
VIDEOS_D = {  # https://www.shutterstock.com/video
    DOG1: 'https://ak.picdn.net/shutterstock/videos/31512310/preview/' +
          'stock-footage-curious-beagle-dog-run-at-grass-chase-moving-camera' +
          '-slow-motion-shot-long-ears-flap-and-fly-in.webm',
    WOMAN_YOGA: 'https://media.istockphoto.com/videos/your-body-knows-what-it-needs-video-id1161189526'
}


def create_demo_file(path: str, ack: bool = True, tabs: int = 1) -> None:
    if not os.path.exists(os.path.dirname(path)):
        mt.create_dir(os.path.dirname(path))
    f = open(path, "w")
    f.write("Now the file has more content!\n" * 1000)
    f.close()
    file_msg = '{}({})'.format(path, mt.file_or_folder_size(path))
    if ack:
        print('{}{}'.format(tabs * '\t', mt.CREATED.format(file_msg)))
    return


def timer_test():
    mt.get_function_name(ack=True, tabs=0)
    start_t = mt.get_timer()
    total = mt.get_timer_delta(s_timer=start_t, with_ms=True)
    total_full = mt.get_timer_delta(s_timer=start_t, with_ms=False)
    print('\tTotal run time {}'.format(total))
    print('\tTotal run time {}'.format(total_full))
    mt.get_timer_delta(s_timer=start_t, with_ms=False, ack=True, tabs=1)
    return


def timer_action_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\ttimer count down example:')
    for i in range(1):
        mt.timer_action(seconds=2, action='take image {}'.format(i), tabs=2)
    # TODO restore after fixing open cv start minimized
    # print('\t press key example:')
    # mt.timer_action(seconds=None, action='taking an image', tabs=2)
    return


def get_time_stamp_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\tdate no day: {}'.format(mt.get_time_stamp(format_s='%Y_%m')))
    print('\tdate: {}'.format(mt.get_time_stamp(format_s='%Y_%m_%d')))
    print('\ttime: {}'.format(mt.get_time_stamp(format_s='%H_%M_%S')))
    mt.get_time_stamp(ack=True, tabs=1)
    print('\ttime stamp for files: {}'.format(mt.get_time_stamp(format_s='%Y_%m_%d_%H_%M_%S')))
    print('\tdate and time with ms: {}'.format(mt.get_time_stamp(format_s='%Y_%m_%d_%H_%M_%S.%f')))
    print('\tdate Israel format {}'.format(mt.get_time_stamp(format_s='%d-%m-%Y %H:%M:%S')))
    return


def get_pc_name_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_pc_name(ack=True, tabs=1)
    print('\tPc name is {}'.format(mt.get_pc_name()))
    return


def get_cuda_version_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_cuda_version(ack=True, tabs=1)
    return


def get_env_variables_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_env_variables(ack=True, tabs=1)
    return


def set_env_variable_test():
    mt.get_function_name(ack=True, tabs=0)
    k = JUST_A_NAME
    mt.set_env_variable(key=k, val=JUST_A_NAME, ack=True, tabs=1)
    print('\tCheck:')
    mt.get_env_variable(key=k, ack=True, tabs=2)
    mt.del_env_variable(key=k, ack=True, tabs=1)
    return


def get_env_variable_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_env_variable(key=JUST_A_NAME, ack=True, tabs=1)
    mt.get_env_variable(key='PATH', ack=True, tabs=1)
    return


def del_env_variable_test():
    mt.get_function_name(ack=True, tabs=0)
    k = JUST_A_NAME
    mt.set_env_variable(key=k, val=JUST_A_NAME, ack=True, tabs=1)

    mt.del_env_variable(key=k, ack=True, tabs=1)
    mt.del_env_variable(key='v2_{}'.format(JUST_A_NAME), ack=True, tabs=1)
    return


def make_cuda_invisible_test():
    mt.get_function_name(ack=True, tabs=0)
    k = 'CUDA_VISIBLE_DEVICES'

    old_value = mt.get_env_variable(key=k)  # save old value
    mt.make_cuda_invisible()  # change to new value
    new_value = mt.get_env_variable(key=k)  # try get new value

    if new_value is not None:
        print('\t{} = {}'.format(k, new_value))
    else:
        print('\tTest Failed')

    # restore old value
    if old_value is None:
        mt.del_env_variable(key=k)  # didn't exist: delete it
    else:
        mt.set_env_variable(key=k, val=old_value)  # existed: restore value

    return


def profiler_test():
    mt.get_function_name(ack=True, tabs=0)
    pr = mt.start_profiler()
    mt.get_function_name(ack=False)
    profiler_str = mt.end_profiler(pr, rows=5, ack=True)
    print(profiler_str)
    return


def os_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\tis_windows     {}'.format(mt.is_windows()))
    print('\tis_linux       {}'.format(mt.is_linux()))

    print('\tis_armv7l      {}'.format(mt.is_armv7l()))
    print('\tis_raspberrypi {}'.format(mt.is_raspberry_pi()))

    print('\tis_aarch64     {}'.format(mt.is_aarch64()))
    print('\tis_jetson_nano {}'.format(mt.is_jetson_nano()))
    return


def main_wrapper_test():
    def temp_function():
        print('hello_world')

    mt.get_function_name(ack=True, tabs=0)
    mt.main_wrapper(
        main_function=temp_function,
        seed=42,
        ipv4=True,
        cuda_off=False,
        torch_v=True,
        tf_v=True,
        cv2_v=True,
        with_pip_list=True,
        with_profiler=False
    )
    return


def to_str_test():
    mt.get_function_name(ack=True, tabs=0)

    # INTS
    x = 1234
    print('\t{}'.format(mt.to_str(var=x)))  # minimal

    x = 12345678912345
    print(mt.to_str(var=x, title='\tvery long int'))

    # FLOATS
    f = 3.4
    print(mt.to_str(var=f, title='\tsmall float', fp=-1))

    f = 3.2123123
    print(mt.to_str(var=f, title='\tlong float(rounded 4 digits)', fp=4))

    f = 1234567890.223123123123123123
    print(mt.to_str(var=f, title='\tbig long float(rounded 3 digits)', fp=3))

    # STRINGS
    s = 'hello world'
    print(mt.to_str(var=s, title='\tregular string'))

    s = ''
    print(mt.to_str(var=s, title='\tempty string'))

    # LISTS
    li = []
    print(mt.to_str(var=li, title='\tempty list'))

    li = [112312312, 3, 4]
    print(mt.to_str(var=li, title='\tlist of ints(recursive print)', rec=True))

    li = [112312312, 3, 4]
    print(mt.to_str(var=li, title='\tlist of ints(no metadata)', wm=False))

    li = [1, 3123123]
    print(mt.to_str(var=li, title='\t1d list of ints(no data)', chars=0))
    print(mt.to_str(var=li, title='\t1d list of ints(all data)', chars=-1))

    li = [1.0000012323, 3123123.22454875123123]
    print(mt.to_str(var=li, title='\t1d list(rounded 7 digits)', fp=7))

    li = [12, 1.2323, 'HI']
    print(mt.to_str(var=li, title='\t1d mixed list'))

    li = [4e-05, 6e-05, 4e-05, 5e-05, 4e-05, 7e-05, 2e-05, 7e-05, 5e-05, 8e-05]
    print(mt.to_str(var=li, title='\te-0 style', fp=6))

    li = [[4e-05, 6e-05, 4e-05, 5e-05, 4e-05, 7e-05, 2e-05, 7e-05, 5e-05, 8e-05]]
    print(mt.to_str(var=li, title='\t2d e-0 style', fp=6, rec=True))

    li = [11235] * 1000
    print(mt.to_str(var=li, title='\t1d long list'))

    li = [[1231.2123123, 15.9], [3.0, 7.55]]
    print(mt.to_str(var=li, title='\t2d list', rec=True))

    li = [(1231.2123123, 15.9), (3.0, 7.55)]
    print(mt.to_str(var=li, title='\t2d list of tuples', rec=True))

    # TUPLES
    t = (1239.123123, 3.12, 9.123123123123)
    print(mt.to_str(var=t, title='\t1d tuple', rec=True))

    # NUMPY
    ni = np.array([4e-05, 6e-02], dtype=float)
    print(mt.to_str(var=ni, title='\t1d np array(fp=5)', fp=5, rec=False))

    ni = np.array([1.0000012323, 3123123.22454875123123], dtype=float)
    print(mt.to_str(var=ni, title='\t1d np array(fp=2)', fp=2, rec=False))

    ni = np.array([[1231.123122, 15.9], [3.0, 7.55], [4e-05, 6e-02]])
    print(mt.to_str(var=ni, title='\t2d np array', rec=True))

    cv_img = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    print(mt.to_str(var=cv_img, title='\tcv_img', chars=20))

    # DICTS
    di = {'a': [1213, 2]}
    print(mt.to_str(var=di, title='\tdict of lists', rec=True))

    di = {'a': [{'k': [1, 2]}, {'c': [7, 2]}]}
    print(mt.to_str(var=di, title='\tnested dict', rec=True))
    return


def save_load_np_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.create_dir(TEMP_FOLDER1)
    path = '{}/{}.npy'.format(TEMP_FOLDER1, JUST_A_NAME)
    a = np.ones(shape=(2, 3, 29))
    print(mt.to_str(a, '\ta'))
    mt.save_np(a, path=path)
    a2 = mt.load_np(path, ack=True)
    print(mt.to_str(a2, '\ta2'))
    mt.delete_dir_with_files(TEMP_FOLDER1)
    _ = mt.load_np('{}.npy'.format(path), ack=True)
    return


def save_load_npz_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.create_dir(TEMP_FOLDER1)
    path = '{}/{}.npz'.format(TEMP_FOLDER1, JUST_A_NAME)
    b = np.ones(shape=(2, 3, 29))
    c = np.ones(shape=(2, 3, 29))
    b_c = {'b': b, 'c': c}
    print(mt.to_str(b_c, '\tb_c'))
    mt.save_npz(b_c, path=path)
    b_c2 = mt.load_npz(path)
    print(mt.to_str(b_c2, '\tb_c2', rec=True))
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def save_load_pkl_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.create_dir(TEMP_FOLDER1)
    path = '{}/{}.pkl'.format(TEMP_FOLDER1, JUST_A_NAME)
    a = {'2': 'a', 'b': 9, 'x': np.ones(shape=3)}
    print(mt.to_str(a, '\ta'))
    mt.save_pkl(data_dict=a, path=path)
    a2 = mt.load_pkl(path=path)
    print(mt.to_str(a2, '\ta2'))
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def get_uniform_dist_by_dim_test():
    mt.get_function_name(ack=True, tabs=0)
    A = np.array([[1, 100], [7, 210], [3, 421]])
    lows, highs = mt.get_uniform_dist_by_dim(A)
    print(mt.to_str(A, '\tA'))
    print(mt.to_str(lows, '\tlows'))
    print(mt.to_str(highs, '\thighs'))
    A = A.tolist()
    print(mt.to_str(A, '\tA'))
    lows, highs = mt.get_uniform_dist_by_dim(A)
    print(mt.to_str(lows, '\tlows'))
    print(mt.to_str(highs, '\thighs'))
    A = mt.np_uniform(shape=(500, 2), lows=[3, 200], highs=[12, 681])
    print(mt.to_str(A, '\tA(lows=[3, 200],highs=[12, 681])'))
    lows, highs = mt.get_uniform_dist_by_dim(A)
    print(mt.to_str(lows, '\tlows'))
    print(mt.to_str(highs, '\thighs'))
    return


def get_normal_dist_by_dim_test():
    mt.get_function_name(ack=True, tabs=0)
    A = np.array([[1, 100], [7, 210], [3, 421]])
    means, stds = mt.get_normal_dist_by_dim(A)
    print(mt.to_str(A, '\tA'))
    print(mt.to_str(means, '\tmeans'))
    print(mt.to_str(stds, '\tstds'))
    A = A.tolist()
    print(mt.to_str(A, '\tA'))
    means, stds = mt.get_normal_dist_by_dim(A)
    print(mt.to_str(means, '\tmeans'))
    print(mt.to_str(stds, '\tstds'))
    A = mt.np_normal(shape=(500, 2), mius=[3, 200], stds=[12, 121])
    print(mt.to_str(A, '\tA(mius=[3, 200],stds=[12, 121])'))
    means, stds = mt.get_normal_dist_by_dim(A)
    print(mt.to_str(means, '\tmeans'))
    print(mt.to_str(stds, '\tstds'))
    return


def np_uniform_test():
    mt.get_function_name(ack=True, tabs=0)
    A = mt.np_uniform(shape=(500, 2), lows=[3, 200], highs=[12, 681])
    print(mt.to_str(A, '\tA(lows=[3, 200],highs=[12, 681])'))
    return


def np_normal_test():
    mt.get_function_name(ack=True, tabs=0)
    A = mt.np_normal(shape=(500, 2), mius=[3, 200], stds=[12, 121])
    print(mt.to_str(A, '\tA(mius=[3, 200],stds=[12, 121])'))
    return


def generate_new_data_from_old_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\tgenerate uniform data example')
    old_data = mt.np_uniform(shape=(500, 2), lows=[3, 200], highs=[12, 681])
    print(mt.to_str(old_data, '\t\told_data(lows=[3, 200],highs=[12, 681])'))
    new_data = mt.generate_new_data_from_old(old_data, new_data_n=4000, dist='uniform')
    lows, highs = mt.get_uniform_dist_by_dim(new_data)
    print(mt.to_str(new_data, '\t\tnew_data'))
    print(mt.to_str(lows, '\t\tlows'))
    print(mt.to_str(highs, '\t\thighs'))

    print('\tgenerate normal data example')
    old_data = mt.np_normal(shape=(500, 2), mius=[3, 200], stds=[12, 121])
    print(mt.to_str(old_data, '\t\told_data(mius=[3, 200],stds=[12, 121])'))
    new_data = mt.generate_new_data_from_old(old_data, new_data_n=4000, dist='normal')
    means, stds = mt.get_normal_dist_by_dim(new_data)
    print(mt.to_str(new_data, '\t\tnew_data'))
    print(mt.to_str(means, '\t\tmeans'))
    print(mt.to_str(stds, '\t\tstds'))
    return


def np_random_integers_test():
    mt.get_function_name(ack=True, tabs=0)
    random_ints = mt.np_random_integers(low=5, high=20, size=(2, 3))
    print(mt.to_str(random_ints, '\trandom_ints from 5-20'))
    return


def augment_x_y_numpy_test():
    mt.get_function_name(ack=True, tabs=0)
    X = mt.np_random_integers(low=5, high=20, size=(10, 3))
    Y = mt.np_random_integers(low=0, high=10, size=(10,))
    print(mt.to_str(X, '\tX'))
    print(mt.to_str(Y, '\tY'))
    A = mt.augment_x_y_numpy(X, Y)
    print(mt.to_str(A, '\tA'))
    return


def de_augment_numpy_test():
    mt.get_function_name(ack=True, tabs=0)
    A = mt.np_random_integers(low=5, high=20, size=(10, 4))
    print(mt.to_str(A, '\tA'))
    X, Y = mt.de_augment_numpy(A)
    print(mt.to_str(X, '\tX'))
    print(mt.to_str(Y, '\tY'))
    return


def nCk_test():
    mt.get_function_name(ack=True, tabs=0)
    A = np.random.randint(low=-10, high=10, size=(3, 2))
    print(mt.to_str(A, '\tA'))

    # let's iterate on every 2 different indices of A
    combs_count = mt.nCk(len(A), k=2, as_int=True)
    print('\t{}C2={}:'.format(len(A), combs_count))  # result is 3

    combs_list = mt.nCk(len(A), k=2)  # result is [[0, 1], [0, 2], [1, 2]]
    for i, comb in enumerate(combs_list):
        print('\t\tcomb {}={}. A[comb]={}'.format(i, comb, A[comb].tolist()))
    return


def redirect_std_test():
    mt.get_function_name(ack=True, tabs=0)
    old_stdout, summary_str = mt.redirect_std_start()
    print('\t\tbla bla bla')
    print('\t\tline2')
    string = mt.redirect_std_finish(old_stdout, summary_str)
    print('\tcaptured output:')
    print(string, end='')  # there is '\n' at the end of the last line
    return


def get_line_number_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_line_number(ack=True)
    return


def get_function_name_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_function_name(ack=True)
    return


def get_file_name_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_file_name(ack=True)
    return


def get_base_file_name_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_base_file_name(ack=True)
    return


def get_function_name_and_line_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_function_name_and_line(ack=True)
    return


def get_base_file_and_function_name_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.get_base_file_and_function_name(ack=True)
    return


def add_color_test():
    mt.get_function_name(ack=True, tabs=0)
    print(mt.add_color(string='\tred ,bold and underlined', ops=['Red', 'bold', 'underlined']))
    print('\t{}'.format(mt.add_color(string='blue ,bold and underlined', ops=['BlUe', 'bo', 'un'])))
    print(mt.add_color(string='\tjust bold', ops='bold'))
    print(mt.add_color(string='\treverse color and bg color', ops='re'))
    print(mt.add_color(string='\tred with background_dark_gray', ops=['red', 'background_dark_gray']))
    print('\t{}'.format(mt.add_color(string='background_light_yellow', ops='background_light_yellow')))
    print(mt.add_color(string='\tblack and background_magenta', ops=['black', 'background_magenta']))
    my_str = 'using mt.to_str()'
    my_str = mt.to_str(var=my_str, title='\t{}'.format(my_str))
    print(mt.add_color(my_str, ops='Green'))
    return


def logger_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.create_dir(TEMP_FOLDER1)
    path = '{}/{}_log_{}.txt'.format(TEMP_FOLDER1, JUST_A_NAME, mt.get_time_stamp(format_s='%Y_%m_%d_%H_%M_%S'))
    mt.init_logger(logger_path=path)
    mt.log_print(line='\tline 1')
    mt.flush_logger()
    mt.log_print(line='line 2', tabs=1)
    mt.log_print(line='line 3', tabs=3)
    mt.close_logger()

    print('reading from {}'.format(path))
    mt.read_file_lines(fp=path, ack=True)
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def create_and_delete_empty_dir_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.create_dir(dir_path=TEMP_FOLDER1)
    mt.delete_empty_dir(dir_path=TEMP_FOLDER1)

    mt.create_dir(dir_path='{}/{}'.format(TEMP_FOLDER1, os.path.basename(TEMP_FOLDER1)))

    mt.delete_empty_dir(dir_path=TEMP_FOLDER1)  # will fail
    # remove empty dir 1 by 1
    mt.delete_empty_dir(dir_path='{}/{}'.format(TEMP_FOLDER1, os.path.basename(TEMP_FOLDER1)))  # will work
    mt.delete_empty_dir(dir_path=TEMP_FOLDER1)  # will work
    return


def create_and_delete_dir_with_files_test():
    mt.get_function_name(ack=True, tabs=0)
    base_path = TEMP_FOLDER1
    inner_dir = '{}/{}'.format(base_path, os.path.basename(TEMP_FOLDER1))
    create_demo_file(path='{}/{}'.format(base_path, DEMO_FILE))
    create_demo_file(path='{}/v2_{}'.format(base_path, DEMO_FILE))
    create_demo_file(path='{}/{}'.format(inner_dir, DEMO_FILE))
    create_demo_file(path='{}/v2_{}'.format(inner_dir, DEMO_FILE))

    # mt.delete_empty_dir(dir_path=base_path)  # will fail
    mt.delete_dir_with_files(dir_path=base_path)
    return


def find_files_in_folder_test():
    mt.get_function_name(ack=True, tabs=0)
    test_f = TEMP_FOLDER1
    inner_f = '{}/inner_folder'.format(test_f)
    txt1 = '{}/{}'.format(test_f, DEMO_FILE)
    txt2 = '{}/v2_{}'.format(test_f, DEMO_FILE)
    txt3 = '{}/{}'.format(inner_f, DEMO_FILE)  # inner file
    zip1 = '{}/{}.zip'.format(test_f, JUST_A_NAME)
    create_demo_file(txt1)
    create_demo_file(txt2)
    create_demo_file(txt3)
    create_demo_file(zip1)
    mt.find_files_in_folder(dir_path=test_f, file_suffix='.txt', ack=True)
    mt.find_files_in_folder(dir_path=test_f, file_suffix='.zip', ack=True)
    mt.find_files_in_folder(dir_path=test_f, ack=True)  # all files in folder
    mt.delete_dir_with_files(test_f)
    return


def find_files_in_folder_sorted_test():
    mt.get_function_name(ack=True, tabs=0)
    test_f = TEMP_FOLDER1

    print('simple numerical test:')
    files = ['0', '32', '3', '5', '4', '31']
    for f in files:
        create_demo_file('{}/{}.jpg'.format(test_f, f), ack=False)

    mt.find_files_in_folder(dir_path=test_f, sort=False, ack=True)
    mt.find_files_in_folder(dir_path=test_f, sort=True, ack=True)
    mt.delete_dir_with_files(test_f)

    print('complex numerical test:')
    for f in files:
        create_demo_file('{}/img_{}demo.jpg'.format(test_f, f), ack=False)
    mt.find_files_in_folder(dir_path=test_f, sort=False, ack=True)
    mt.find_files_in_folder(dir_path=test_f, sort=True, ack=True)
    mt.delete_dir_with_files(test_f)

    print('time stamp test:')
    for i in range(5):
        ts = mt.get_time_stamp(format_s='%Y_%m_%d_%H_%M_%S_%f')
        if i % 2:  # trim a digit from the end to give sorting something to do (else sort and no sort - same result)
            ts = ts[:-1]
        create_demo_file('{}/img_{}.jpg'.format(test_f, ts), ack=False)

    mt.find_files_in_folder(dir_path=test_f, sort=False, ack=True)
    mt.find_files_in_folder(dir_path=test_f, sort=True, ack=True)
    mt.delete_dir_with_files(test_f)
    return


def move_file_test():
    mt.get_function_name(ack=True, tabs=0)
    file_src = '{}/{}'.format(TEMP_FOLDER1, DEMO_FILE)
    file_dst = '{}/v2_{}'.format(TEMP_FOLDER1, DEMO_FILE)
    create_demo_file(file_src)
    mt.move_file(file_src=file_src, file_dst=file_dst)
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def copy_file_test():
    mt.get_function_name(ack=True, tabs=0)
    file_src = '{}/{}'.format(TEMP_FOLDER1, DEMO_FILE)
    file_dst = '{}/v2_{}'.format(TEMP_FOLDER1, DEMO_FILE)
    create_demo_file(file_src)
    mt.copy_file(file_src=file_src, file_dst=file_dst)
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def delete_file_test():
    mt.get_function_name(ack=True, tabs=0)
    path = '{}/{}'.format(TEMP_FOLDER1, DEMO_FILE)
    create_demo_file(path)
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def delete_files_test():
    mt.get_function_name(ack=True, tabs=0)
    path1 = '{}/v1_{}'.format(TEMP_FOLDER1, DEMO_FILE)
    path2 = '{}/v2_{}'.format(TEMP_FOLDER1, DEMO_FILE)
    create_demo_file(path1)
    create_demo_file(path2)
    mt.delete_files(files=[path1, path2])
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def sleep_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.sleep(seconds=2, ack=True, tabs=1)
    return


def reverse_tuple_or_list_test():
    mt.get_function_name(ack=True, tabs=0)
    my_tuple = (0, 0, 255)
    print(mt.to_str(my_tuple, '\tmy_tuple'))
    print(mt.to_str(mt.reverse_tuple_or_list(my_tuple), '\tmy_tuple_reversed'))
    my_list = [0, 0, 255]
    print(mt.to_str(my_list, '\tmy_list'))
    print(mt.to_str(mt.reverse_tuple_or_list(my_list), '\tmy_list_reversed'))
    return


def round_list_test():
    mt.get_function_name(ack=True, tabs=0)
    li = [1.23123123, 12.123123123123123123, 1.2, 1.0, 6e-06]
    print(mt.to_str(var=li, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_list(li, fp=1), title='\tfp=1', fp=-1))
    print(mt.to_str(var=mt.round_list(li, fp=6), title='\tfp=6', fp=-1))

    li = [[1.23123123, 12.123123123123123123, 1.2, 1.0, 6e-06, 2]]
    print(mt.to_str(var=li, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_list(li, fp=6), title='\tfp=6', fp=-1))

    l2 = [5e-05, 0.00014, 5e-10, 0.0001, 6e-07, 5e-05, 8e-05, 6e-05, 5e-05, 5e-05, 2]
    print(mt.to_str(var=l2, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_list(l2, fp=5), title='\tfp=5', fp=-1))

    l2 = ['x', 'y', 'z', 0.3]
    print(mt.to_str(var=l2, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_list(l2, fp=5, warn=True), title='\tfp=5', fp=-1))
    return


def round_tuple_test():
    mt.get_function_name(ack=True, tabs=0)
    li = (1.23123123, 12.123123123123123123, 1.2, 1.0)
    print(mt.to_str(var=li, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_tuple(li, fp=1), title='\tfp=1', fp=-1))
    print(mt.to_str(var=mt.round_tuple(li, fp=3), title='\tfp=3', fp=-1))
    l2 = (5e-05, 0.00014, 5e-10, 0.0001, 6e-07, 5e-05, 8e-05, 6e-05, 5e-05, 5e-05)
    print(mt.to_str(var=l2, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_tuple(l2, fp=5), title='\tfp=5', fp=-1))
    l2 = ('x', 'y', 'z')
    print(mt.to_str(var=l2, title='\torigin', fp=-1))
    print(mt.to_str(var=mt.round_tuple(l2, fp=5, warn=True), title='\tfp=5', fp=-1))
    return


def shuffle_np_array_test():
    mt.get_function_name(ack=True, tabs=0)
    A = mt.np_normal(shape=(6,), mius=3.5, stds=10.2)
    print(mt.to_str(A, '\tA'))
    A = mt.shuffle_np_array(A)
    print(mt.to_str(A, '\tA'))
    return


def shuffle_np_arrays_test():
    mt.get_function_name(ack=True, tabs=0)
    A = np.array([1, 2, 3, 4, 5, 6])
    B = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    print(mt.to_str(A, '\tA'))
    print(mt.to_str(B, '\tB'))
    A, B = mt.shuffle_np_arrays(
        arr_tuple=(A, B)
    )
    print(mt.to_str(A, '\tA'))
    print(mt.to_str(B, '\tB'))
    return


def array_info_print_test():
    B = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    mt.array_info_print(B, 'B')
    return


def get_key_by_value_test():
    mt.get_function_name(ack=True, tabs=0)
    j = {"x": 3, "a": "dx"}
    print('\tfirst key that has value 3 is {}'.format(mt.get_key_by_value(j, value=3)))
    print('\tfirst key that has value "dx" is {}'.format(mt.get_key_by_value(j, value="dx")))
    return


def to_hex_and_bin_test():
    mt.get_function_name(ack=True, tabs=0)
    variable = 'no meaning to to content'
    print('\taddress of variable is 0d{}'.format(id(variable)))
    print('\taddress of variable is 0x{}'.format(mt.to_hex(id(variable))))
    print('\taddress of variable is 0b{}'.format(mt.to_bin(id(variable))))
    return


def dict_as_table_test():
    mt.get_function_name(ack=True, tabs=0)
    table = {'gilad': 3, 'a': 11233.1213, 'aasdasd': 9913123, 'b': 'hello'}
    mt.dict_as_table(table=table, title='my table', fp=2, tabs=1)
    # mt.dict_as_table(table=table, title='my table', fp=5, tabs=2)
    return


def is_same_type_test():
    mt.get_function_name(ack=True, tabs=0)
    li = ['s', 123, 'a', 'b']
    print('\tis li {} homogeneous ? {}'.format(li, mt.is_same_type(li)))
    li = ['s', 'a', 'b']
    print('\tis li {} homogeneous ? {}'.format(li, mt.is_same_type(li)))
    t = (1.2, 123, 12, 13)
    print('\tis t {} homogeneous ? {}'.format(t, mt.is_same_type(t)))
    t = (1, 123, 12, 13)
    print('\tis t {} homogeneous ? {}'.format(t, mt.is_same_type(t)))
    return


def hard_disc_test():
    mt.get_function_name(ack=True, tabs=0)
    print(mt.hard_disc(one_liner=True, tabs=1))
    print(mt.hard_disc(one_liner=False, tabs=1))
    return


def ram_size_test():
    mt.get_function_name(ack=True, tabs=0)
    print(mt.ram_size(one_liner=True, tabs=1))
    print(mt.ram_size(one_liner=False, tabs=1))
    return


def cpu_info_test():
    mt.get_function_name(ack=True, tabs=0)
    print(mt.cpu_info(one_liner=True, tabs=1))
    print(mt.cpu_info(one_liner=False, tabs=1))
    return


def file_or_folder_size_test():
    mt.get_function_name(ack=True, tabs=0)
    mt.file_or_folder_size('{}.txt'.format(FAKE_FILE), ack=True, tabs=1)

    path = '{}/{}'.format(TEMP_FOLDER1, DEMO_FILE)
    create_demo_file(path)
    mt.file_or_folder_size(path, ack=True, tabs=1)
    mt.delete_file(path)

    create_demo_file('{}/{}'.format(TEMP_FOLDER1, DEMO_FILE))
    create_demo_file('{}/v2_{}'.format(TEMP_FOLDER1, DEMO_FILE))
    create_demo_file('{}/{}/{}'.format(TEMP_FOLDER1, os.path.basename(TEMP_FOLDER1), DEMO_FILE))
    mt.file_or_folder_size(TEMP_FOLDER1, ack=True, tabs=1)
    mt.delete_dir_with_files(TEMP_FOLDER1)
    return


def compress_and_extract_test():
    mt.get_function_name(ack=True, tabs=0)
    # single file
    print('\tSingle file tests:')
    f_name = JUST_A_NAME
    file_to_compress_path = '{}/{}.txt'.format(TEMP_FOLDER1, f_name)
    create_demo_file(path=file_to_compress_path)  # temp file

    for file_t in ['zip', '7z', 'tar']:
        print('\t{}:'.format(file_t))
        compressed_file_path = '{}/{}'.format(TEMP_FOLDER1, f_name)
        mt.compress_file_or_folder(src=file_to_compress_path, dst_path=compressed_file_path, file_type=file_t, ack=True,
                                   tabs=2)
        compressed_file_path += '.{}'.format(file_t)
        dst_folder = '{}/{}_{}'.format(TEMP_FOLDER1, f_name, file_t)
        mt.extract_file(src=compressed_file_path, dst_folder=dst_folder, file_type=file_t, tabs=2)
        mt.delete_file(compressed_file_path, tabs=2)  # remove compressed
        mt.delete_dir_with_files(dst_folder, tabs=2)  # remove extracted

    mt.delete_file(file_to_compress_path)  # remove temp file

    # folder
    print('\tFolder tests:')
    folder_to_compress_path = '{}/{}'.format(TEMP_FOLDER1, os.path.basename(TEMP_FOLDER1))
    f1 = '{}/{}'.format(folder_to_compress_path, DEMO_FILE)
    create_demo_file(path=f1)  # temp file 1
    f2 = '{}/v2_{}.txt'.format(folder_to_compress_path, DEMO_FILE)
    create_demo_file(path=f2)  # temp file
    for file_t in ['zip', '7z', 'tar']:
        print('\t{}:'.format(file_t))
        compressed_folder_path = '{}/{}'.format(TEMP_FOLDER1, os.path.basename(TEMP_FOLDER1))
        mt.compress_file_or_folder(src=folder_to_compress_path, dst_path=compressed_folder_path, file_type=file_t,
                                   ack=True, tabs=2)
        compressed_folder_path += '.{}'.format(file_t)
        dst_folder = '{}/{}_{}'.format(TEMP_FOLDER1, os.path.basename(TEMP_FOLDER1), file_t)
        mt.extract_file(src=compressed_folder_path, dst_folder=dst_folder, file_type=file_t, tabs=2)
        mt.delete_file(compressed_folder_path, tabs=2)  # remove compressed
        mt.delete_dir_with_files(dst_folder, tabs=2)  # remove extracted
    mt.delete_dir_with_files(TEMP_FOLDER1)  # remove temp folder with temp file 1 and 2
    return


def classFPS_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    # fps = mt.FPS(last_k=50, cache_size=100, summary_title='classFPS_test')
    # tabs = 1
    # print('{}test empty get_fps() = {:.4f}'.format(tabs * '\t', fps.get_fps()))
    # for t in range(500):
    #     ack = (t + 1) % 40 == 0
    #     fps.start(ack_progress=ack, tabs=tabs)
    #     # do_work of round t
    #     mt.sleep(seconds=0.03)
    #     if t == 0:
    #         mt.sleep(seconds=1)
    #     fps.update(ack_progress=ack, tabs=tabs + 1)
    #     if ack:
    #         print('{}\tget_fps() = {:.4f}'.format(tabs * '\t', fps.get_fps()))
    # fps.finalize(tabs)

    fps = mt.FPS(last_k=2, cache_size=2, summary_title='classFPS_test')
    tabs = 1
    print('{}test empty get_fps() = {:.4f}'.format(tabs * '\t', fps.get_fps()))
    for t in range(10):
        ack = (t + 1) % 1 == 0
        fps.start(ack_progress=ack, tabs=tabs)
        # do_work of round t
        mt.sleep(seconds=0.03)
        if t == 0:
            mt.sleep(seconds=1)
        fps.update(ack_progress=ack, tabs=tabs + 1)
        if ack:
            print('{}\tget_fps() = {:.4f}'.format(tabs * '\t', fps.get_fps()))
    fps.finalize(tabs)

    return


def classFPS_many_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    # if you measure more than 1
    fps_list = [mt.FPS(summary_title='work1'), mt.FPS(summary_title='work2')]
    tabs = 1
    for t in range(3):
        for i in range(len(fps_list)):
            fps_list[i].start(ack_progress=True, tabs=tabs, with_title=True)
            # do_work of round t
            mt.sleep(seconds=0.03)
            fps_list[i].update(ack_progress=True, tabs=tabs + 1, with_title=True)
    for fps in fps_list:
        fps.finalize(tabs)
    return


def try_say_welcome():
    mt.get_function_name_and_line(ack=True, tabs=0)
    try:
        from wizzi_utils.tts.tts import MachineBuddy
        MachineBuddy.speak('Welcome to wizzi utils package', block=True)  # Static use:
        # MachineBuddy.speak('There is a dog pooping outside')  # Static use:
    except(ImportError, ModuleNotFoundError) as e:
        mt.exception_error('Can\'t do try_say_welcomec. {}'.format(e), real_exception=True)
    return


def get_linkable_exception_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    try:
        1 / 0
    except Exception as e:
        mt.exception_error(e, real_exception=True, tabs=1)
    return


def generate_requirements_file_test(real_req: bool = False):
    mt.get_function_name_and_line(ack=True, tabs=0)
    if real_req:  # for saving a snapshot for the users
        proj_root = mt.get_repo_root(repo_name='wizzi_utils_pypi')
        fp = os.path.abspath('{}/resources/wizzi_utils_requirements.txt'.format(proj_root))
    else:
        fp = '{}/requirements.txt'.format(PLAY_GROUND)
    mt.generate_requirements_file(fp_out=fp)
    _ = mt.read_file_lines(fp=fp, ack=True)
    return


def run_shell_command_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    if mt.is_windows():
        mt.run_shell_command(cmd='dir')
    else:
        mt.run_shell_command(cmd='ls')
    return


def run_shell_command_and_get_out_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    if mt.is_windows():
        out = mt.run_shell_command_and_get_out(cmd='dir', ack_out=False)
    else:
        out = mt.run_shell_command_and_get_out(cmd='ls', ack_out=False)
    for i, line in enumerate(out):
        print('\t{}){}'.format(i, line))
    return


def add_sudo_to_cmd_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    if mt.is_linux():
        mt.get_function_name_and_line(ack=True, tabs=0)
        cmd = 'ls'
        cmd_sudo = mt.add_sudo_to_cmd(cmd, sudo_password='my_password')
        mt.run_shell_command(cmd=cmd_sudo)
    return


def size_remain_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    if mt.is_windows():
        mt.size_remain(partition='C:', ack=True)
        mt.size_remain(partition='E:', ack=True)
    else:
        mt.size_remain(partition='/dev/root', ack=True)
    return


def convert_size_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    gb_bytes = 1073741824  # 1 GB in bytes
    mb_bytes = 1048576  # 1 MB in bytes

    size_int = gb_bytes
    size_str = mt.convert_size(size_bytes=size_int)
    print('\t{} is {}'.format(size_int, size_str))
    size_int2 = mt.convert_size_to_bytes(size_gb=1, size_mb=0)
    print('\t1GB is {}(should be {})'.format(size_int2, size_int))

    size_int3 = mb_bytes * 200  # 200MB
    size_str = mt.convert_size(size_bytes=size_int3)
    print('\t{} is {}'.format(size_int3, size_str))
    size_int2 = mt.convert_size_to_bytes(size_gb=0, size_mb=200)
    print('\t200MB is {}(should be about {})'.format(size_int2, size_int3))

    size_int3 = gb_bytes * 2 + mb_bytes * 500  # 2.5 GB
    size_str = mt.convert_size(size_bytes=size_int3)
    print('\t{} is {}'.format(size_int3, size_str))
    size_int2 = mt.convert_size_to_bytes(size_gb=2, size_mb=500)
    print('\t2.5GB is {}(should be {})'.format(size_int2, size_int3))
    return


def size_remain_greater_than_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    if mt.is_windows():
        print('checking if size remain on C: >= 1.5GB:')
        mt.size_remain_greater_than(partition='C:', gb=1, mb=500, ack=True)
        print('checking if size remain on C: >= 100GB and 10MB')
        mt.size_remain_greater_than(partition='C:', gb=100, mb=10, ack=True)
        print('checking if size remain on C: >= 500MB')
        mt.size_remain_greater_than(partition='C:', gb=0, mb=500, ack=True)
        print('checking if size remain on E: >= 500MB')
        mt.size_remain_greater_than(partition='E:', gb=0, mb=500, ack=True)
        print('checking if size remain on C: >= 2500MB(2.5GB)')
        mt.size_remain_greater_than(partition='C:', gb=0, mb=2500, ack=True)
    else:
        print('checking if size remain on /dev/root >= 1.5GB:')
        mt.size_remain_greater_than(partition='/dev/root', gb=1, mb=500, ack=True)
    return


def get_emoji_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    emojis = mt.EMOJIS.get_emojis()
    for em in emojis:
        print('\tthis is a {} {}'.format(em.name, mt.get_emoji(em)))
    return


def has_admin_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    mt.has_admin(ack=True)
    return


def read_file_lines_test():
    mt.get_function_name_and_line(ack=True, tabs=0)

    src_path = '{}/{}'.format(TEMP_FOLDER1, DEMO_FILE)
    create_demo_file(src_path)  # create a file

    # add junk
    f = open(src_path, "w")
    f.write("  first     line \n")
    f.write("           2nd     line               \n")
    f.close()

    mt.read_file_lines(fp=src_path, ack=True)
    return


def get_path_file_system_info_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    _, _, _, _ = mt.get_path_file_system_info(path=PLAY_GROUND, ack=True)
    if mt.is_windows():
        if os.path.exists('C:\\Windows\\System32'):
            mt.get_path_file_system_info(path='C:\\Windows\\System32', ack=True)
    if mt.is_linux():
        mt.get_path_file_system_info(path='/home', ack=True)

    _, _, _, _ = mt.get_path_file_system_info(path='./fake_file.py', ack=True)
    return


def get_repo_root_test():
    mt.get_function_name_and_line(ack=True, tabs=0)
    mt.get_repo_root(repo_name='NO_REPO', ack=True)
    mt.get_repo_root(repo_name='2021wizzi_utils', ack=True)
    return


def rename_folder_test():
    mt.get_function_name_and_line(ack=True, tabs=0)

    _ = mt.rename_folder(dir_path='./NoSuchDir', new_name='NoSuchDir2')  # expected exception

    dir_path = '{}/NewDir'.format(PLAY_GROUND)
    mt.create_dir(dir_path=dir_path)
    create_demo_file(path='{}/dm.txt'.format(dir_path))
    _ = mt.rename_folder(dir_path=dir_path, new_name='NewDir')  # expected exception
    new_dir_fp = mt.rename_folder(dir_path=dir_path, new_name='NewDir2')  # expect success
    mt.delete_dir_with_files(dir_path=new_dir_fp)
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name(depth=1)))
    try_say_welcome()
    timer_test()
    timer_action_test()
    get_time_stamp_test()
    get_pc_name_test()
    get_cuda_version_test()
    get_env_variables_test()
    set_env_variable_test()
    get_env_variable_test()
    del_env_variable_test()
    make_cuda_invisible_test()
    profiler_test()
    os_test()
    main_wrapper_test()
    to_str_test()
    save_load_np_test()
    save_load_npz_test()
    save_load_pkl_test()
    get_uniform_dist_by_dim_test()
    get_normal_dist_by_dim_test()
    np_uniform_test()
    np_normal_test()
    generate_new_data_from_old_test()
    np_random_integers_test()
    augment_x_y_numpy_test()
    de_augment_numpy_test()
    nCk_test()
    redirect_std_test()
    get_line_number_test()
    get_function_name_test()
    get_file_name_test()
    get_base_file_name_test()
    get_function_name_and_line_test()
    get_base_file_and_function_name_test()
    add_color_test()
    logger_test()
    create_and_delete_empty_dir_test()
    create_and_delete_dir_with_files_test()
    move_file_test()
    copy_file_test()
    find_files_in_folder_test()
    find_files_in_folder_sorted_test()
    delete_file_test()
    delete_files_test()
    sleep_test()
    reverse_tuple_or_list_test()
    round_list_test()
    round_tuple_test()
    shuffle_np_array_test()
    shuffle_np_arrays_test()
    array_info_print_test()
    get_key_by_value_test()
    to_hex_and_bin_test()
    dict_as_table_test()
    is_same_type_test()
    hard_disc_test()
    ram_size_test()
    cpu_info_test()
    compress_and_extract_test()
    file_or_folder_size_test()
    classFPS_test()
    get_linkable_exception_test()
    generate_requirements_file_test()
    run_shell_command_test()
    run_shell_command_and_get_out_test()
    add_sudo_to_cmd_test()
    size_remain_test()
    size_remain_greater_than_test()
    convert_size_test()
    get_emoji_test()
    has_admin_test()
    read_file_lines_test()
    get_path_file_system_info_test()
    get_repo_root_test()
    rename_folder_test()
    print('{}'.format('-' * 20))
    return
