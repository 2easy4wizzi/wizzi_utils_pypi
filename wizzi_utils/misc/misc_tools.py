import os
import datetime
from timeit import default_timer as timer
from typing import Callable
import cProfile
import pstats
import io
import numpy as np
import random
import inspect
import sys
import time
import pickle
from itertools import combinations
import shutil
import math
import re
import psutil
import platform
import glob
import ctypes
from pathlib import Path
from traceback import TracebackException
# noinspection PyProtectedMember
# from pip import _internal
import subprocess
import pkg_resources
from enum import Enum

LINES = '-' * 80
NOT_FOUND = '{} Not found'
CREATED = '{} Created'
EXISTS = '{} Exists'
DELETED = '{} Deleted'
MOVED = '{} Moved to {}'
COPIED = '{} Copied to {}'
SAVED = '{} Saved'
LOADED = '{} Loaded'
UPLOADED = '{} Uploaded to {}'
DOWNLOADED = '{} Downloaded to {}'
CONTENT = 'Content: {}'
SUCCESS_C = 'light_green'
SUCCESS_C2 = ['black', 'bold', 'background_green']
FAIL_C = 'red'
FAIL_C2 = ['black', 'bold', 'background_red']

CONST_COLOR_SHORTCUTS = {
    'b': 'blue',
    'g': 'green',
    'r': 'red',
    'c': 'cyan',
    'm': 'magenta',
    'y': 'yellow',
    'k': 'black',
    'w': 'white',
    'bo': 'bold',
    'un': 'underlined',
    're': 'reverse',
}

CONST_COLOR_MAP = {
    'reset_all': "\033[0m",
    'bold': "\033[1m",
    'underlined': "\033[4m",
    'reverse': "\033[7m",  # switch font color and background color

    'blue': "\033[34m",
    'background_blue': "\033[44m",
    'light_blue': "\033[94m",
    'background_light_blue': "\033[104m",

    'green': "\033[32m",
    'background_green': "\033[42m",
    'light_green': "\033[92m",
    'background_light_green': "\033[102m",

    'red': "\033[31m",
    'background_red': "\033[41m",
    'light_red': "\033[91m",
    'background_light_red': "\033[101m",

    'cyan': "\033[36m",
    'background_cyan': "\033[46m",
    'background_light_cyan': "\033[106m",
    'light_cyan': "\033[96m",

    'magenta': "\033[35m",
    'background_magenta': "\033[45m",
    'light_magenta': "\033[95m",
    'background_light_magenta': "\033[105m",

    'yellow': "\033[33m",
    'background_yellow': "\033[43m",
    'light_yellow': "\033[93m",
    'background_light_yellow': "\033[103m",

    'light_gray': "\033[37m",
    'background_light_gray': "\033[47m",
    'dark_gray': "\033[90m",
    'background_dark_gray': "\033[100m",

    'black': "\033[97m",
    'background_black': "\033[107m",

    'white': "\033[30m",
    'background_white': "\033[40m",
}


def get_linkable_exception() -> str:
    """
    :return: e.g. File "D:/workspace/2021wizzi_utils/wizzi_utils/misc/test/test_misc_tools.py", line 851,
                            in get_linkable_exception_test: division by zero
    where file is clickable
    see get_linkable_exception_test
    """
    _, value, tb = sys.exc_info()
    info_str = ''
    for i, line in enumerate(TracebackException(type(value), value, tb, limit=None).format(chain=True)):
        if "File \"" in line:  # care only for the line with: filename(clickable), line, func name
            # print(i, line.split('\n')[0])
            info_str = line.split('\n')[0].strip()  # remove the exception itself
    return info_str


def exception_error(e: (Exception, str), real_exception: bool = False, depth: int = 2, tabs: int = 1):
    """
    Aux function - print exception error in red with function name
    :param e: error. e.g. <class 'ModuleNotFoundError'> or str of your choosing
    :param real_exception: if True: gets info from sys. else, manually get file and line
    :param depth: if real_exception False, needs the depth to get file and line
    :param tabs:
    :return:
    see get_linkable_exception_test
    """
    if real_exception:
        err_meta = get_linkable_exception()
    else:
        err_meta = get_function_name_and_line(depth=depth)
    error_str = '{}{}: {}'.format(tabs * '\t', err_meta, e)
    print(add_color(string=error_str, ops='Red'))
    return


def chop_microseconds(delta: datetime.timedelta) -> datetime.timedelta:
    """
    Aux function - removes micro seconds from datetime.timedelta object
    e.g. 0:00:02.000001 -> 0:00:02
    :param delta:
    :return:
    """
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def get_timer_delta(s_timer: float, e_timer: float = None, with_ms: bool = False, ack: bool = False,
                    tabs: int = 1) -> datetime.timedelta:
    """
    :param s_timer: begin time
    :param e_timer: end time - optional. if None end time is now
    :param with_ms: if microseconds needed - set to true. else: no microseconds
    :param ack: print time passed
    :param tabs:
    :return:
    see timer_test()
    """
    if e_timer is None:
        e_timer = get_timer()
    d = datetime.timedelta(seconds=(e_timer - s_timer))
    if not with_ms:
        d = chop_microseconds(d)
    if ack:
        print('{}Time passed {}'.format(tabs * '\t', d))
    return d


def get_timer() -> float:
    """
    sets a timer beginning
    :return:
    see timer_test()
    """
    return timer()


def timer_action(seconds: int, action: str = '', tabs: int = 1) -> None:
    """
    :param seconds:
    :param action:
    :param tabs:
    :return:
    counts till seconds or block
    see timer_action_test()
    """
    if seconds is None:
        input('{}Press "Enter" key for {}...'.format(tabs * '\t', action))
    else:
        time_in_future = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
        print('{}{} IN: {}'.format(tabs * '\t', action, seconds), end='', flush=True)
        while time_in_future > datetime.datetime.now():
            time.sleep(1)
            seconds -= 1
            print(' {}'.format(seconds), end='', flush=True)
        print('')
    return


def get_time_stamp(format_s: str = '%Y_%m_%d_%H_%M_%S', ack: bool = False, tabs: int = 1) -> str:
    """
    :param format_s: date time format
    :param ack: prints current time
    :param tabs:
    the default is for files
    :return:
    see get_current_date_hour_test()

    '%Y-%m-%d %H:%M:%S.%f' with ms
    """
    now = datetime.datetime.now()
    time_stamp = now.strftime(format_s)
    if ack:
        print('{}timeStamp = {}'.format(tabs * '\t', time_stamp))
    return time_stamp


def get_pc_name(ack: bool = False, tabs: int = 1) -> str:
    """
    :param ack:
    :param tabs:
    :return: pc name as str
    see get_pc_name_test()
    """
    try:
        pc_name = platform.uname()[1]
        if ack:
            print('{}* Computer Name: {}'.format(tabs * '\t', pc_name))
    except ModuleNotFoundError as e:
        pc_name = ''
        exception_error(e, real_exception=True)
    return pc_name


def get_cudnn_version(cudnn_path: str) -> str:
    cudnn_v = None
    if os.path.exists(cudnn_path):
        # extract CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL
        f_obj = open(cudnn_path, 'r')
        ma, mi, pa = None, None, None
        for line in f_obj:
            ls = line.strip()
            if ls.startswith('#define CUDNN_MAJOR'):
                ls_spl = ls.split(' ')
                if len(ls_spl) == 3:
                    ma = ls_spl[2]
            elif ls.startswith('#define CUDNN_MINOR'):
                ls_spl = ls.split(' ')
                if len(ls_spl) == 3:
                    mi = ls_spl[2]
            elif ls.startswith('#define CUDNN_PATCHLEVEL'):
                ls_spl = ls.split(' ')
                if len(ls_spl) == 3:
                    pa = ls_spl[2]
        f_obj.close()
        if ma and mi and pa:
            cudnn_v = '{}.{}.{}'.format(ma, mi, pa)
    return cudnn_v


def get_cuda_version(ack: bool = False, tabs: int = 1) -> str:
    """
    :param ack:
    :param tabs:
    :return: cuda version if found on environment variables
    see get_cuda_version_test()
    """
    cuda_v = None
    cudnn_v = None

    if is_windows():
        cuda_path = get_env_variable(key='CUDA_PATH')
        if cuda_path is not None:
            cuda_v = os.path.basename(cuda_path)
            cudnn_v = get_cudnn_version(cudnn_path='{}/include/cudnn.h'.format(cuda_path))

    elif is_linux():
        cuda_path = get_env_variable(key='CUDA_ROOT')
        cuda_v_file = '{}/version.txt'.format(cuda_path)
        if os.path.exists(cuda_v_file):
            f_obj = open(cuda_v_file, 'r')
            for line in f_obj:
                ls = line.strip()
                if ls.startswith('CUDA Version'):
                    ls_spl = ls.split(' ')
                    if len(ls_spl) == 3:
                        cuda_v = line.strip().split(' ')[2]
                        break
        if cuda_v is not None:
            cudnn_path = '{}/include/cudnn.h'.format(cuda_path)
            if is_jetson_nano():  # cudnn location on jetson_nano
                cudnn_path = '/usr/include/aarch64-linux-gnu/cudnn_version_v8.h'
            cudnn_v = get_cudnn_version(cudnn_path=cudnn_path)
            if cudnn_v is None:  # TRY via env variable
                cudnn_path = get_env_variable(key='CUDNN_FILE_PATH')
                if cudnn_path is not None:
                    cudnn_v = get_cudnn_version(cudnn_path=cudnn_path)

    if cuda_v is None:
        cuda_v = add_color('* No CUDA_PATH found', ops=FAIL_C)
    else:
        cuda_v = '{}* CUDA Version: {}'.format(tabs * '\t', cuda_v)
        if cudnn_v is not None:
            cuda_v += ' (cuDNN Version {})'.format(cudnn_v)
        else:
            cuda_v += ' (No cuDNN found)'
        cuda_v = add_color(cuda_v, ops=SUCCESS_C)

    if ack:
        print(cuda_v)
    return cuda_v


def get_env_variables(ack: bool = False, tabs: int = 1) -> dict:
    """
    :param ack:
    :param tabs:
    :return: dict with envs
    see get_env_variables_test()
    """
    env_d = dict(os.environ)
    if ack:
        print('{}Environment variables:'.format(tabs * '\t'))
        for k, v in env_d.items():
            print('{}\t{} = {}'.format(tabs * '\t', k, v))
    return env_d


def set_env_variable(key: str, val: str, ack: bool = False, tabs: int = 1) -> None:
    """
    :param key:
    :param val:
    :param ack:
    :param tabs:
    insert new env variable
    see set_env_variable_test()
    """
    key = key.upper()
    os.environ[key] = val
    if ack:
        print('{}Inserted to environment: {} = {}'.format(tabs * '\t', key, val))
    return


def get_env_variable(key: str, ack: bool = False, tabs: int = 1) -> str:
    """
    :param key:
    :param ack:
    :param tabs:
    :return: env variable value
    see get_env_variable_test()
    """
    key = key.upper()
    ret_val = os.environ[key] if key in os.environ else None
    if ack:
        if ret_val is not None:
            print('{}{} = {}'.format(tabs * '\t', key, ret_val))
        else:
            exception_error(NOT_FOUND.format(key), real_exception=False, tabs=tabs)
    return ret_val


def del_env_variable(key: str, ack: bool = False, tabs: int = 1) -> None:
    """
    :param key:
    :param ack:
    :param tabs:
    :return: env variable value
    see get_env_variable_test()
    """
    key = key.upper()
    if key in os.environ:
        del os.environ[key]
        if ack:
            print('{}{}'.format(tabs * '\t', DELETED.format(key)))
    else:
        exception_error(NOT_FOUND.format(key), real_exception=False, tabs=tabs)
    return


def silence_cv_warn() -> None:
    """
    disables on windows:
    [ WARN:0] global .... src/cap_msmf.cpp (438)
    `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback
    :return:
    if doesn't work on linux try:
    set_env_variable(key='OPENCV_VIDEOIO_DEBUG', val='0')
    """
    set_env_variable(key='OPENCV_VIDEOIO_PRIORITY_MSMF', val='0')
    return


def make_cuda_invisible() -> None:
    """
        disable gpu 0
        FUTURE - support disabling many GPUS
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0 available
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1, 0'  # GPU 0 not available
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, -1, 0'  # GPU 0 available but 1 disabled
        os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, -1, 0'  # GPU 1,2 available but 0 disabled

        read more: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
        see make_cuda_invisible_test()
    """
    set_env_variable(key='CUDA_VISIBLE_DEVICES', val='-1, 0')
    return


def start_profiler() -> cProfile.Profile:
    """
    starts profiling
    :return: profiling object that is needed for end_profiler()
    see profiler_test()
    """
    pr = cProfile.Profile()
    pr.enable()
    return pr


def end_profiler(pr: cProfile.Profile, rows: int = 40, ack: bool = False) -> str:
    """
    profiling output
    :param pr: object returned from start_profiler()
    :param rows: how many rows to print sorted by 'cumulative' run time
    :param ack:
    :return: profiler output as string
    see profiler_test()
    """
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(rows)
    profiler_str = s.getvalue()
    if ack:
        print('{}'.format(profiler_str))
    return profiler_str


def set_seed(seed: int = 42) -> None:
    """
    :param seed: setting numpy and random seeds
    :return:
    see main_wrapper_test() - uses set_seed
    """
    np.random.seed(seed)
    random.seed(seed)
    return


def version() -> str:
    """
    :return:
    """
    # v = pkg_resources.require("wizzi_utils")[0].version
    v = '8.0.2'
    return v


def is_windows() -> bool:
    """
    :return:
    see os_test() - tested
    """
    return platform.system() == "Windows"


def is_linux() -> bool:
    """
    :return:
    see os_test()
    """
    return platform.system() == "Linux"


def is_armv7l() -> bool:
    """
    armv7l is 32 bit processor.
    :return:
    see os_test()
    """
    try:
        res = os.uname()[4] == "armv7l"
    except AttributeError:
        res = False
    return res


def is_raspberry_pi() -> bool:
    """
    armv7l is 32 bit processor.
    :return:
    see os_test()
    """
    res = is_linux() and is_armv7l()
    return res


def is_aarch64() -> bool:
    """
    ARM64 is the 64-bit extension of the ARM architecture
    :return:
    see os_test()
    """
    try:
        res = os.uname()[4] == "aarch64"
    except AttributeError:
        res = False
    return res


def is_jetson_nano() -> bool:
    """
    :return:
    see os_test()
    """
    res = is_linux() and is_aarch64()
    return res


def get_system_info() -> str:
    """
    print(sys.platform):
    https://docs.python.org/3/library/sys.html#sys.platform
        AIX 'aix'
        Linux 'linux'
        Windows 'win32'
        Windows/Cygwin 'cygwin'
        macOS 'darwin'

    print(os.name):
        Windows 'nt'
        RPi 'posix' # maybe for all linux
    print(platform.system()):
        Windows 'Windows'
    :return:
    """
    named_tuple = platform.uname()
    # t = tuple(named_tuple)
    return str(named_tuple)


def get_py_version() -> str:
    """
    on linux there is \n and then gcc
    e.g.
    3.7.3 (default, Jan 22 2021, 20:04:44)
    [GCC 8.3.0]
    :return:
    """
    py_ver = sys.version.replace('\n', '')
    return py_ver


def main_wrapper(
        main_function: Callable,
        seed: int = -1,
        ipv4: bool = False,
        cuda_off: bool = False,
        torch_v: bool = False,
        tf_v: bool = False,
        cv2_v: bool = False,
        with_pip_list: bool = False,
        with_profiler: bool = False
) -> None:
    """
    :param main_function: the function to run
    :param seed: if -1 no seed, else set_seed(seed=seed)
    :param ipv4: print computer ipv4
    :param cuda_off: make gpu invisible and force run on cpu
    :param torch_v: print torch version
    :param tf_v: print tensorflow lite version
    :param cv2_v: print opencv version
    :param with_pip_list: print pip list (all libraries installed)
    :param with_profiler: run profiler
    template:
    wu.main_wrapper(
        main_function=main,
        seed=42,
        ipv4=False,
        cuda_off=False,
        torch_v=False,
        tf_v=False,
        cv2_v=False,
        with_profiler=False
    )
    see main_wrapper_test()
    :return:
    """
    print(LINES)
    start_timer = get_timer()

    print('main_wrapper:')
    print('* Run started at {}'.format(get_time_stamp(format_s='%d-%m-%Y %H:%M:%S')))
    print('* Python Version {}'.format(get_py_version()))
    print('* Operating System {}'.format(get_system_info()))
    print('* Interpreter: {}'.format(sys.executable))
    print('* wizzi_utils Version {}'.format(version()))
    print('* Working Dir: {}'.format(os.getcwd()))
    try:
        from wizzi_utils.socket.socket_tools import get_mac_address_uuid
        print('* Computer Mac: {}'.format(get_mac_address_uuid()))
    except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
        print(add_color('* {}'.format(err), 'r'))

    print('* CPU Info: {}'.format(cpu_info(one_liner=True, tabs=0)))
    print('* Physical Memory: {}'.format(hard_disc(one_liner=True, tabs=0)))
    print('* RAM: {}'.format(ram_size(one_liner=True, tabs=0)))

    if ipv4:
        try:
            from wizzi_utils.socket.socket_tools import get_active_ipv4
            print('* Computer ipv4: {}'.format(get_active_ipv4()))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            print(add_color('* {}'.format(err), 'r'))

    cuda_msg = get_cuda_version(ack=False, tabs=0)
    if cuda_off:
        make_cuda_invisible()
        cuda_msg += ' (Turned off)'
    print(cuda_msg)

    if torch_v:
        try:
            from wizzi_utils.torch.torch_tools import get_torch_version
            print(get_torch_version(ack=False, tabs=0))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            print(add_color('* torch not Found. error: {}'.format(err), 'r'))

    if cv2_v:
        try:
            from wizzi_utils.open_cv.open_cv_tools import get_cv_version
            print(get_cv_version(ack=False, tabs=0))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            print(add_color('* opencv-python not Found. error: {}'.format(err), 'r'))

    if tf_v:
        try:
            from wizzi_utils.tflite.tflite_tools import get_tflite_version
            print(get_tflite_version(ack=False, tabs=0))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            print(add_color('* tflite_runtime not Found. error: {}'.format(err), 'r'))

    if seed > -1:
        set_seed(seed=seed)
        print('* Seed was initialized to {}'.format(seed))

    if with_pip_list:
        print('* Pip list output:')
        # _internal.main(['list'])
        pkgs = get_pip_freeze_out(ack=False)
        for i, pkg in enumerate(pkgs):
            print('\t{}'.format(pkg))

    print('Function {} started:'.format(main_function))
    print(LINES)
    pr = start_profiler() if with_profiler else None

    main_function()
    if with_profiler:
        print(end_profiler(pr, rows=40))

    print(LINES)
    print('Total run time {}'.format(get_timer_delta(start_timer)))
    return


def get_data_str(data_str_raw: str, chars: int) -> str:
    """
    Aux function for to_str()
    :param data_str_raw:
    :param chars:
    :return: data_str
    """
    data_str_raw = data_str_raw.replace('\n', '').replace('  ', '')
    if chars == -1:  # all data
        chars = len(data_str_raw) + 1

    data_str_rep = ': {}'.format(data_str_raw[:chars])

    if len(data_str_raw) > chars > 0:
        data_str_rep += ' ...too long'
    return data_str_rep


def is_int(var: any) -> bool:
    return isinstance(var, (int, np.int, np.int32, np.uint8))


def is_float(var: any) -> bool:
    return isinstance(var, (float, np.float, np.float32, np.float64))


def is_str(var: any) -> bool:
    return isinstance(var, str)


def is_list(var: any) -> bool:
    return isinstance(var, list)


def is_tuple(var: any) -> bool:
    return isinstance(var, tuple)


def is_numpy(var: any) -> bool:
    return isinstance(var, np.ndarray)


def is_dict(var: any) -> bool:
    return isinstance(var, dict)


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
        chars==0: no chars
        chars==-1: all chars
    :param fp: float_precision: round number if possible(float, list or np array of floats...)
            fp>=0 round
            fp==-1: no rounding
    :param wm: with_meta: with meta data such as type, len/shape, dtype...
    :param rec: recursive: to keep printing if there are more items inside e.g. np.array(shape=(2,3,4)) -> 3 prints
    :return: informative string of the variable
    see to_str_test()
    """
    if fp >= 0:
        int_str_f = '{:,}'
        float_str_f = '{:,.%xf}' % fp
    else:
        int_str_f, float_str_f = None, None

    string = title
    type_s = str(type(var)).replace('<class \'', '').replace('\'>', '')  # clean type name

    if is_float(var):
        if wm:
            string += '({})'.format(type_s)
        if chars != 0:  # -1 for all, or x>0 for x chars
            if fp == 0:
                data_str_raw = int_str_f.format(var)
            elif fp > 0:
                data_str_raw = float_str_f.format(var)
            else:
                data_str_raw = str(var)
            string += get_data_str(data_str_raw, chars)

    elif is_int(var):
        if wm:
            string += '({})'.format(type_s)
        if chars != 0:  # -1 for all, or x>0 for x chars
            data_str_raw = int_str_f.format(var)
            string += get_data_str(data_str_raw, chars)

    elif is_str(var):
        if wm:
            string += '({}'.format(type_s)
            string += ',len={})'.format(var.__len__())
        if chars != 0:  # -1 for all, or x>0 for x chars
            string += get_data_str(var, chars)

    elif is_list(var) or is_tuple(var):
        if wm:
            string += '({}'.format(type_s)
            string += ',len={})'.format(var.__len__())

        if chars != 0:  # -1 for all, or x>0 for x chars
            if fp >= 0:
                if all((is_int(item) or is_float(item)) for item in var):  # 1d list of int and floats
                    if all(is_int(item) for item in var):  # if all ints - no rounding
                        f_format = '{:,}'
                    else:
                        f_format = '{:,.%xf}' % fp
                    new_v = [f_format.format(li_item) for li_item in var]
                else:  # >1d list or 1d with not just ints and floats
                    new_v = round_list(var, fp=fp)

                if is_tuple(var):
                    new_v = tuple(new_v)
            else:
                new_v = var
            data_str_raw = str(new_v).replace('\'', '')

            string += get_data_str(data_str_raw, chars)

        if rec and len(var) > 0:
            inner_str = to_str(var=var[0], title='{}[0]'.format(title), chars=chars, fp=fp, rec=rec)
            string += '\n\t{}'.format(inner_str)

    elif is_numpy(var):
        if wm:
            string += '({}'.format(type_s)
            string += ',s={}'.format(var.shape)
            string += ',dtype={})'.format(var.dtype)
        if chars != 0:  # -1 for all, or x>0 for x chars
            # new_v = var.tolist()
            if fp >= 0:
                if all((is_int(item) or is_float(item)) for item in var):  # 1d list of int and floats
                    if all(is_int(item) for item in var):  # if all ints - no rounding
                        f_format = '{:,}'
                    else:
                        f_format = '{:,.%xf}' % fp
                    new_v = [f_format.format(li_item) for li_item in var]
                else:  # >1d list or 1d with not just ints and floats
                    new_v = np.around(var.tolist(), fp).tolist()
            else:
                new_v = var
            data_str_raw = str(new_v).replace('\'', '')
            string += get_data_str(data_str_raw, chars)

        if rec and var.shape[0] > 0:  # recursive call
            inner_str = to_str(var=var[0], title='{}[0]'.format(title), chars=chars, fp=fp, rec=rec)
            string += '\n\t{}'.format(inner_str)

    elif is_dict(var):
        if wm:
            string += '({}'.format(type_s)
            string += ',len={}'.format(var.__len__())
            if len(var.keys()) <= 5:  # only small amount of keys with be printed.
                string += ',keys={})'.format(list(var.keys()))

        if chars != 0:  # -1 for all, or x>0 for x chars
            string += get_data_str(str(var), chars)

        if rec and len(var) > 0:  # recursive call
            first_k = next(iter(var))
            inner_str = to_str(var=var[first_k], title='{}[{}]'.format(title, first_k), chars=chars, fp=fp, rec=rec)
            string += '\n\t{}'.format(inner_str)

    else:
        # all unidentified elements get default print (title(type): data)
        # must have str() to this type
        if wm:
            string += '({})'.format(type_s)

        if chars != 0:  # -1 for all, or x>0 for x chars
            string += get_data_str(str(var), chars)
    return string


def save_np(t: np.array, path: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param t: numpy array
    :param path: suffix '.npy' added automatically if not exists
    :param ack:
    :param tabs:
    :return:
    see save_load_np_test()
    """
    np.save(path, t)
    if ack:
        size_s = file_or_folder_size(path)
        file_msg = '{}({})'.format(path, size_s)
        print('{}{}'.format(tabs * '\t', SAVED.format(file_msg)))
    return


def load_np(path: str, ack: bool = True, tabs: int = 1) -> np.array:
    """
    :param path:
    :param ack:
    :param tabs:
    :return: numpy array
    see save_load_np_test()
    """
    if os.path.exists(path):
        t = np.load(path)
        if ack:
            size_s = file_or_folder_size(path)
            file_msg = '{}({})'.format(path, size_s)
            print('{}{}'.format(tabs * '\t', LOADED.format(file_msg)))
    else:
        exception_error(NOT_FOUND.format(path), real_exception=False, tabs=tabs)
        t = None
    return t


def save_npz(arrays_dict: dict, path: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param arrays_dict: e.g. { 'a': np.ones(3) }
    :param path:
    :param ack:
    :param tabs:
    :return:
    save a dict of numpy arrays
    see save_load_npz_test()
    """
    np.savez(path, **{n: a for n, a in arrays_dict.items()})
    if ack:
        size_s = file_or_folder_size(path)
        file_msg = '{}({})'.format(path, size_s)
        print('{}{}. Keys={}'.format(tabs * '\t', SAVED.format(file_msg), arrays_dict.keys()))
    return


def load_npz(path: str, ack: bool = True, tabs: int = 1) -> dict:
    """
    :param path:
    :param ack:
    :param tabs:
    :return: numpy array
    see save_load_npz_test()
    """
    if os.path.exists(path):
        arrays_obj = np.load(path)
        arrays_dict = {}
        # noinspection PyUnresolvedReferences
        for k in arrays_obj.files:
            arrays_dict[k] = arrays_obj[k]
        if ack:
            size_s = file_or_folder_size(path)
            file_msg = '{}({})'.format(path, size_s)
            print('{}{}. Keys={}'.format(tabs * '\t', LOADED.format(file_msg), arrays_dict.keys()))
    else:
        exception_error(NOT_FOUND.format(path), real_exception=False, tabs=tabs)
        arrays_dict = None
    return arrays_dict


def save_pkl(data_dict: dict, path: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param data_dict:
    :param path:
    :param ack:
    :param tabs:
    :return:
    see save_load_pkl_test()
    """
    file_obj = open(path, "wb")
    pickle.dump(data_dict, file_obj)
    file_obj.close()
    if ack:
        size_s = file_or_folder_size(path)
        file_msg = '{}({})'.format(path, size_s)
        print('{}{}'.format(tabs * '\t', SAVED.format(file_msg)))
    return


def load_pkl(path: str, ack: bool = True, tabs: int = 1) -> dict:
    """
    :param path:
    :param ack:
    :param tabs:
    :return:
    see save_load_pkl_test()
    """
    if os.path.exists(path):
        file_obj = open(path, "rb")
        data_dict = pickle.load(file_obj)
        file_obj.close()
        if ack:
            size_s = file_or_folder_size(path)
            file_msg = '{}({})'.format(path, size_s)
            print('{}{}'.format(tabs * '\t', LOADED.format(file_msg)))
    else:
        exception_error(NOT_FOUND.format(path), real_exception=False, tabs=tabs)
        data_dict = None
    return data_dict


def get_uniform_dist_by_dim(A: [np.array, list]) -> (np.array, np.array):
    """
    :param A:
    :return:
    for every dimension gets the lowest and highest
    see get_uniform_dist_by_dim_test()
    """
    lows = np.min(A, axis=0)
    highs = np.max(A, axis=0)
    return lows, highs


def get_normal_dist_by_dim(A: [np.array, list]) -> (np.array, np.array):
    """
    :param A:
    :return:
    see get_normal_dist_by_dim_test()
    """
    means = np.mean(A, axis=0)
    stds = np.std(A, axis=0)
    return means, stds


def np_uniform(shape: tuple, lows: [list, int], highs: [list, int]) -> np.array:
    """
    :param shape:
    :param lows:
    :param highs:
    :return:
    see np_uniform_test()
    """
    ret = np.random.uniform(low=lows, high=highs, size=shape)
    return ret


def np_normal(shape: tuple, mius: [list, int, float], stds: [list, int, float]) -> np.array:
    """
    :param shape:
    :param mius:
    :param stds:
    :return:
    see np_normal_test()
    """
    ret = np.random.normal(loc=mius, scale=stds, size=shape)
    return ret


def generate_new_data_from_old(old_data: np.array, new_data_n: int, dist: str = 'normal'):
    """
    :param old_data:
    :param new_data_n:
    :param dist:
    :return:
    see generate_new_data_from_old_test()
    """
    d = old_data.shape[1]
    if dist == 'uniform':
        lows, highs = get_uniform_dist_by_dim(old_data)
        new_data = np_uniform(shape=(new_data_n, d), lows=lows, highs=highs)
    else:  # else normal
        means, stds = get_normal_dist_by_dim(old_data)
        new_data = np_normal(shape=(new_data_n, d), mius=means, stds=stds)
    return new_data


def np_random_integers(low: int, high: int, size: tuple) -> np.array:
    """
    :param low:
    :param high:
    :param size:
    :return:
    see np_random_integers_test()
    """
    ret = np.random.random_integers(low=low, high=high, size=size)
    return ret


def augment_x_y_numpy(X: np.array, y: np.array) -> np.array:
    """
    :param X:
    :param y:
    :return:
    see augment_x_y_numpy_test()
    """
    assert X.shape[0] == y.shape[0], 'row count must be the same'
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.reshape(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.reshape(y.shape[0], 1)
    A = np.column_stack((X, y))
    return A


def de_augment_numpy(A: np.array) -> (np.array, np.array):
    """
    :param A:
    :return:
    see de_augment_numpy_test()
    """
    if len(A.shape) == 1:  # A is 1 point. change from size (n) to size (1,n)
        A = A.reshape(1, A.shape[0])
    X, y = A[:, :-1], A[:, -1]
    if len(X.shape) == 1:  # change x size()=[n,] to size()=[n,1]
        X = X.reshape(X.shape[0], 1)
    if len(y.shape) == 1:  # change y size()=[n,] to size()=[n,1]
        y = y.reshape(y.shape[0], 1)
    return X, y


def nCk(n: int, k: int, as_int: bool = False):
    """
    :param n:
    :param k:
    :param as_int:
    :return: if as_int True: the result of nCk, else the combinations of nCk
    n choose k
    see nCk_test()
    """
    range_list = np.arange(0, n, 1)
    combs = list(combinations(range_list, k))
    combs = [list(comb) for comb in combs]
    if as_int:
        combs = len(combs)
    return combs


def redirect_std_start() -> (io.TextIOWrapper, io.StringIO):
    """
    redirect all prints to summary_str
    :return:
        io.TextIOWrapper - to revert back the prints to sys.stdout
        io.StringIO - to extract output
    see redirect_std_test()
    """
    old_stdout = sys.stdout
    sys.stdout = summary_str = io.StringIO()
    return old_stdout, summary_str


def redirect_std_finish(old_stdout: io.TextIOWrapper, summary_str: io.StringIO) -> str:
    """
    :param old_stdout: to revert back the prints to sys.stdout
    :param summary_str: to extract output
    :return:
    redirect all prints back to std out and return a string of what was captured"
    see redirect_std_test()
    """
    sys.stdout = old_stdout
    return summary_str.getvalue()


def get_line_number(depth: int = 1, ack: bool = False, tabs: int = 1) -> str:
    """
    :param depth:
    :param ack:
    :param tabs:
    :return:
    see get_line_number_test()
    """
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.lineno)
        if ack:
            print('{}Line {}:'.format(tabs * '\t', ret_val))
    except IndexError as e:
        exception_error(e, real_exception=True)
    return ret_val


def get_function_name(depth: int = 1, ack: bool = False, tabs: int = 1) -> str:
    """
    :param depth:
    :param ack:
    :param tabs:
    :return:
    see
    """
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.function)
        if ack:
            print('{}{}:'.format(tabs * '\t', ret_val))
    except IndexError as e:
        exception_error(e, real_exception=True)
    return ret_val


def get_file_name(depth: int = 1, ack: bool = False, tabs: int = 1) -> str:
    """
    :param depth:
    :param ack:
    :param tabs:
    :return:
    see get_file_name_test()
    """
    ret_val = ''
    try:
        scope_1_back = inspect.stack()[depth]  # stack()[0] is this function
        ret_val = '{}'.format(scope_1_back.filename)
        if ack:
            print('{}{}:'.format(tabs * '\t', ret_val))
    except IndexError as e:
        exception_error(e, real_exception=True)
    return ret_val


def get_base_file_name(depth: int = 1, ack: bool = False, tabs: int = 1) -> str:
    """
    :param depth:
    :param ack:
    :param tabs:
    :return:
    see get_base_file_name_test()
    """
    # +1 because of this function
    file_name = get_file_name(depth + 1)
    base_name = os.path.basename(file_name)
    if ack:
        print('{}{}:'.format(tabs * '\t', base_name))
    return base_name


def get_function_name_and_line(depth: int = 1, ack: bool = False, tabs: int = 1) -> str:
    """
    :param depth:
    :param ack:
    :param tabs:
    :return:
    see get_function_name_and_line_test()
    """
    # +1 because of this function
    ret_val = '{}::{}'.format(get_function_name(depth + 1), get_line_number(depth + 1))
    if ack:
        print('{}{}:'.format(tabs * '\t', ret_val))
    return ret_val


def get_base_file_and_function_name(depth: int = 1, ack: bool = False, tabs: int = 1) -> str:
    """
    :param depth:
    :param ack:
    :param tabs:
    :return:
    see get_base_file_and_function_name_test()
    """
    # +1 because of this function
    ret_val = '{}::{}'.format(get_base_file_name(depth + 1), get_line_number(depth + 1))
    if ack:
        print('{}{}:'.format(tabs * '\t', ret_val))
    return ret_val


def add_color(
        string: str,
        ops: (str, list) = None
) -> str:
    """
    :param string:
    :param ops:
        str: color, bg_color, bold, underline, reverse
        list: every combination of color, bg_color, bold, underline, reverse
    colors:
    * X = { blue(b), green(g), red(r), cyan(c), magenta(m), yellow(y) }
        for x in X: x, background_x, light_x, background_light_x
    * more colors:
        light_gray, background_light_gray, dark_gray, background_dark_gray
        black, background_black
        white, background_white

    special options:
    * bold(bo), underlined(un), reverse(re)

    to see all colors and options:
        for k, v in wu.CONST_COLOR_MAP.items():
            print('{}{}{}'.format(v, k, wu.CONST_COLOR_MAP['reset_all']))
    see add_color_test()
    """
    if ops is None:
        ops = ['r']
    elif is_str(ops):
        ops = [ops]
    elif is_list(ops):
        pass  # good to go

    out_string = string
    # first to lower and check shortcuts
    colors = ''
    for op in ops:
        op = op.lower()
        if op in CONST_COLOR_SHORTCUTS:
            op = CONST_COLOR_SHORTCUTS[op]
        if op in CONST_COLOR_MAP:
            colors += CONST_COLOR_MAP[op]
    if colors != '':
        out_string = '{}{}{}'.format(colors, string, CONST_COLOR_MAP['reset_all'])

    return out_string


def create_dir(dir_path: str, ack: bool = True, tabs: int = 1):
    """
    :param dir_path:
    :param ack:
    :param tabs:
    :return:
    see create_and_delete_dir_test()
    """
    if not os.path.exists(dir_path):
        try:
            # os.mkdir(dir_path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            if ack:
                print('{}{}'.format(tabs * '\t', CREATED.format(dir_path)))
        except OSError as e:
            exception_error(e, real_exception=True)
    else:
        exception_error(EXISTS.format(dir_path), real_exception=False, tabs=tabs)
    return


def delete_empty_dir(dir_path: str, ack: bool = True, tabs: int = 1):
    """
    :param dir_path:
    :param ack:
    :param tabs:
    :return:
    see create_and_delete_empty_dir_test()
    """
    if os.path.exists(dir_path):
        files_and_dirs = []
        for _dir_path, dir_names, file_names in os.walk(dir_path):
            for d in dir_names:
                # fd = '{}/{}'.format(_dir_path.replace('\\', '/'), d)
                files_and_dirs.append(d)
            for f in file_names:
                # fp = '{}/{}'.format(_dir_path.replace('\\', '/'), f)
                files_and_dirs.append(f)
        if len(files_and_dirs) > 0:
            err = '{} HAS {} FILES/DIRS {} - use delete_dir_with_files()'.format(dir_path, len(files_and_dirs),
                                                                                 files_and_dirs)
            exception_error(err, real_exception=False, tabs=tabs)
        else:
            try:
                os.rmdir(dir_path)
                if ack:
                    print('{}{}'.format(tabs * '\t', DELETED.format(dir_path)))
            except OSError as e:
                exception_error(e, real_exception=True)
    else:
        exception_error(NOT_FOUND.format(dir_path), real_exception=False, tabs=tabs)
    return


def delete_dir_with_files(dir_path: str, ack: bool = True, tabs: int = 1):
    """
    :param dir_path:
    :param ack:
    :param tabs:
    :return:
    see create_and_delete_dir_test()
    """
    if os.path.exists(dir_path):
        files_and_dirs = []
        for _dir_path, dir_names, file_names in os.walk(dir_path):
            for d in dir_names:
                # fd = '{}/{}'.format(_dir_path.replace('\\', '/'), d)
                files_and_dirs.append(d)
            for f in file_names:
                # fp = '{}/{}'.format(_dir_path.replace('\\', '/'), f)
                files_and_dirs.append(f)
        try:
            size_s = file_or_folder_size(dir_path)
            shutil.rmtree(dir_path)
            status = '({} files/dirs, size {} - {})'.format(len(files_and_dirs), size_s, files_and_dirs)
            if ack:
                print('{}{} - {}'.format(tabs * '\t', DELETED.format(dir_path), status))
        except OSError as e:
            exception_error(e, real_exception=True)
    else:
        exception_error(NOT_FOUND.format(dir_path), real_exception=False, tabs=tabs)
    return


def find_files_in_folder(
        dir_path: str,
        file_suffix: str = '',
        sort: bool = False,
        ack: bool = False,
        tabs: int = 1
) -> list:
    """
    prune dir RECURSIVELY for files full paths

    :param dir_path:  root dir.
    :param file_suffix:  e.g. '.jgp'
        if file_suffix=='', returns all files in dir_path
    :param sort: sort results
        false: no sort
        true: sort alphanumerical.
        e.g. your files are enumerated but contain letters in the name (e.g. t1, t2, t11, t3):
            no   sort order: t1, t11, t2, t3
            with sort order: t1, t2, t3, t11
        Tested (in find_files_in_folder_sorted_test()) on same format names e.g. img_x.jpg where x from i to j
        also tested with timestamps

    :param ack: prints base names of files found
    :param tabs:
    :return:
    see find_files_in_folder_test() and find_files_in_folder_sorted_test()
    """

    def __try_cast_to_int(string_part: str) -> (str, int):
        try:
            return int(string_part)
        except ValueError:
            return string_part

    def __alphanumerical_key(file_fp: str):
        """
        splits string into a list of string and numbers chunks
        e.g. file_fp='x19b': ['x', 19, 'b']
        """
        return [__try_cast_to_int(c) for c in re.split('(\\d+)', os.path.basename(file_fp))]

    def __alphanumerical_sort(strings_list: list) -> None:
        """
        sorts the given list with numbers as key
        """
        strings_list.sort(key=__alphanumerical_key)
        return

    if os.path.exists(dir_path):
        dir_path = os.path.abspath(dir_path)  # to get output of full absolute path
        all_files_found = glob.glob('{}/**/*{}'.format(dir_path, file_suffix), recursive=True)

        all_files_processed = []
        for fp in all_files_found:
            fp = fp.replace('\\', '/')  # replace windows folder \ with /
            if fp.startswith('./'):  # e.g. ./test.txt -> test.txt
                fp = fp[2:]
            if os.path.isfile(fp):  # collect only files not dirs
                all_files_processed.append(fp)

        if sort:
            __alphanumerical_sort(all_files_processed)

        if ack:
            all_files_base = [os.path.basename(fp) for fp in all_files_processed]
            if file_suffix == '':
                msg = '{}found {} files in folder "{}":'
                print(msg.format(tabs * '\t', len(all_files_processed), dir_path))
            else:
                msg = '{}found {} files that ends with {} in folder "{}":'
                print(msg.format(tabs * '\t', len(all_files_processed), file_suffix, dir_path))
            print('{}\t{}'.format(tabs * '\t', all_files_base))

    else:
        all_files_processed = None
        exception_error(NOT_FOUND.format(dir_path), real_exception=False, tabs=tabs)
    return all_files_processed


def move_file(file_src: str, file_dst: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param file_src:
    :param file_dst:
    :param ack:
    :param tabs:
    :return:
    see move_file_test()
    """
    if os.path.exists(file_src):
        size_s = file_or_folder_size(file_src)
        if os.path.exists(os.path.dirname(file_dst)):
            shutil.move(file_src, file_dst)
            # os.rename(file_src, file_dst)
            # os.replace(file_src, file_dst)
            if ack:
                file_dst_msg = '{}({})'.format(file_dst, size_s)
                print('{}{}'.format(tabs * '\t', MOVED.format(file_src, file_dst_msg)))
        else:
            exception_error(NOT_FOUND.format(os.path.dirname(file_dst)), real_exception=False, tabs=tabs)
    else:
        exception_error(NOT_FOUND.format(file_src), real_exception=False, tabs=tabs)
    return


def copy_file(file_src: str, file_dst: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param file_src:
    :param file_dst:
    :param ack:
    :param tabs:
    :return:
    see copy_file_test()
    """
    if os.path.exists(file_src):  # origin file exists
        size_s = file_or_folder_size(file_src)  # target folder exists
        if os.path.exists(os.path.dirname(file_dst)):  # target file doesn't exists
            if not os.path.exists(file_dst):
                shutil.copyfile(file_src, file_dst)
                if ack:
                    file_dst_msg = '{}({})'.format(file_dst, size_s)
                    print('{}{}'.format(tabs * '\t', COPIED.format(file_src, file_dst_msg)))
            else:
                exception_error(EXISTS.format(file_dst), real_exception=False, tabs=tabs)
        else:
            exception_error(NOT_FOUND.format(os.path.dirname(file_dst)), real_exception=False, tabs=tabs)
    else:
        exception_error(NOT_FOUND.format(file_src), real_exception=False, tabs=tabs)
    return


def delete_file(file: str, ack: bool = True, tabs: int = 1) -> None:
    """
    :param file:
    :param ack:
    :param tabs:
    see delete_file_test()
    """
    if os.path.exists(file):
        size_s = file_or_folder_size(file)
        os.remove(file)
        if ack:
            file_msg = '{}({})'.format(file, size_s)
            print('{}{}'.format(tabs * '\t', DELETED.format(file_msg)))
    else:
        exception_error(NOT_FOUND.format(file), real_exception=False, tabs=tabs)
    return


def delete_files(files: list, ack: bool = True, tabs: int = 1) -> None:
    """
    :param files:
    :param ack:
    :param tabs:
    :return:
    see delete_files_test()
    """
    size_s = 0
    for file in files:
        if os.path.exists(file):
            size_s += file_or_folder_size(file, as_str=False)
            delete_file(file, ack=False, tabs=tabs)
    if ack:
        file_msg = '{}({})'.format(files, convert_size(size_s))
        print('{}{}'.format(tabs * '\t', DELETED.format(file_msg)))
    return


def sleep(seconds: (int, float), ack: bool = False, tabs: int = 1):
    """
    :param seconds:
    :param ack:
    :param tabs:
    :return:
    see sleep_test()
    """
    if ack:
        print('{}Sleeping {} seconds'.format(tabs * '\t', seconds))
    time.sleep(seconds)
    return


def reverse_tuple_or_list(orig: [tuple, list]) -> [tuple, list]:
    """
    :param orig: list or tuple
    :return: dst: reversed list or tuple
    see reverse_tuple_or_list_test()
    """
    dst = orig[::-1]
    return dst


def round_tuple(t: tuple, fp: int = 3, warn: bool = False):
    """
    :param t: tuple 1D of floats
    :param fp: float precision >=0
    :param warn: if to should warning in case of rounding failure(e.g. string in list)
    :return: new_t: rounded tuple
    """
    if fp >= 0:
        new_t = tuple(round_list(li=t, fp=fp, warn=warn))
    else:
        new_t = t
    return new_t


def round_list(li: (list, tuple), fp: int = 3, warn: bool = False) -> list:
    """
    :param li: list or tuple 1D of floats
    :param fp: float precision >= 0
    :param warn: if to should warning in case of rounding failure(e.g. string in list)
    :return: new_li: rounded list
    see round_list_test()
    """

    new_li = li
    if fp >= 0 and len(li) > 0:
        try:
            new_np = np.array(li)
            # if >1d list of floats\ints - this will work
            if new_np.dtype in [float, np.float, np.float32, np.float64]:
                new_li = np.around(new_np, fp).tolist()
        except TypeError as e:  # list has non ints\floats elements or uneven size ...
            if warn:
                exception_error(e, real_exception=True)
    return new_li


def shuffle_np_array(arr: np.array) -> np.array:
    """
    :param arr:
    :return:
    shuffles an array
    see shuffle_np_array_test()
    """
    if is_numpy(arr):
        arr = arr[np.random.permutation(arr.shape[0])]
    return arr


def shuffle_np_arrays(arr_tuple: tuple) -> tuple:
    """
    :param arr_tuple: tuple of arrays (numpy)
        len(arr) is equal on all arrays
    :return: shuffled arrays

    see shuffle_np_arrays_test()
    """
    arrays_size = len(arr_tuple[0])
    rand_perm = np.random.permutation(arrays_size)

    out_tuple = ()
    for arr in arr_tuple:
        arr_shf = arr[rand_perm]
        out_tuple += (arr_shf,)
    return out_tuple


def array_info_print(
        array: np.array,
        title: str = 'var',
        chars: (int, str) = 100,
        fp: (int, None) = 2,
        wm: bool = True,
        rec: bool = False,
        tabs: int = 1
) -> None:
    """
    :param array: (recommended 1d)
    :param title:
    :param chars:
    :param fp:
    :param wm:
    :param rec:
    :param tabs:
    :return:
    prints to_str and then mean, std and sum
    see array_info_print_test()
    """
    print(to_str(var=array, title='{}{}'.format(tabs * '\t', title), chars=chars, fp=fp, wm=wm, rec=rec))
    mean_s = 'mean={:,.%xf}' % fp
    mean_s = mean_s.format(np.mean(array))
    std_s = 'std={:,.%xf}' % fp
    std_s = std_s.format(np.std(array))
    sum_s = 'sum={:,.%xf}' % fp
    sum_s = sum_s.format(np.sum(array))
    print('{}\t{}, {}, {}'.format(tabs * '\t', mean_s, std_s, sum_s))
    return


def get_key_by_value(d: dict, value: any) -> str:
    """
    :param d: dict
    :param value:
    Notice that it will return the first key of the value given. if value not unique...
    :return: the key of the value
    see get_key_by_value_test()
    """
    key = None
    for k, v in d.items():
        if v == value:
            key = k
            break
    return key


def to_hex(num: int) -> str:
    """
    :param num:
    :return:
    convert decimal to hex as str
    see to_hex_and_bin_test()
    """
    string = '{:X}'.format(num)
    return string


def to_bin(num: int) -> str:
    """
    :param num:
    :return:
    convert decimal to bin as str
    see to_hex_and_bin_test()
    """
    string = '{:b}'.format(num)
    return string


def dict_as_table(table: dict, title: str = 'table', fp: int = 2, ack: bool = True, tabs: int = 1) -> str:
    """
    :param table: dict of strings to values. keys must be strings
    :param title:
    :param fp: if v is number - float precision
    :param ack:
    :param tabs:
    :return:
    see dict_as_table_test()
    """
    same_type, type_keys = is_same_type(table.keys())
    string = None
    if same_type and type_keys == 'str':
        str_s = 0  # calculate longest key
        for k in table.keys():
            if len(k) > str_s:
                str_s = len(k)
        string = '{}{}:'.format(tabs * '\t', title)
        s_format_all = '{}\t{:%d}  ==> {}' % str_s
        for i, (k, v) in enumerate(table.items()):
            string += '\n'
            string += s_format_all.format(tabs * '\t', k, to_str(v, title='key_{}'.format(i), chars=500, fp=fp))
            # string += s_format_all.format(tabs * '\t', k, v)
        if ack:
            print(string)
    else:
        exception_error('supports only dict with keys from type str', real_exception=False, tabs=tabs)

    # str_s = 0
    # for k in table.keys():
    #     if len(k) > str_s:
    #         str_s = len(k)
    # s_format_all = '{}\t{:%d}  ==> {}' % str_s
    # # s_format = '{}\t{:%d}  ==> {:%d}' % (str_s, str_s)
    # # s_format_to_num = '{}\t{:%d}  ==> {:%d,}' % (str_s, str_s)
    # # s_format_to_list = '{}\t{:%d}  ==> {}' % str_s
    # for k, v in table.items():
    #     print(s_format_all.format(tabs * '\t', k, to_str(v)))
    #     # if is_int(v) or is_float(v):
    #     #     print(s_format_to_num.format(tabs * '\t', k, round(v, fp)))
    #     # elif is_list(v):
    #     #     print(s_format_to_list.format(tabs * '\t', k, v))
    #     # else:
    #     #     print(s_format.format(tabs * '\t', k, v))
    return string


def is_same_type(container: (list, tuple)) -> (bool, str):
    """
    :param container:
    :return:
    check if all items in list or tuple are of the same type
    if they are, return the type
    see is_same_type_test
    """
    same_type, first_type = True, None
    if len(container) > 0:
        it = iter(container)
        first_type = type(next(it))
        # for item in container:
        #     print(type(item))
        same_type = all((type(x) is first_type) for x in it)
        if not same_type:
            first_type = None
        else:
            first_type = str(first_type).replace('<class \'', '').replace('\'>', '')

    return same_type, first_type


def convert_size(size_bytes: int) -> str:
    """
    :param size_bytes:
    :return:
    get size in bytes and return string e.g. '241.19 MB'
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def hard_disc(one_liner: bool = False, tabs: int = 1):
    """
    :param one_liner: per partition
    :param tabs:
    :return:
    see hard_disc_test()
    """
    partitions = psutil.disk_partitions()

    if one_liner:
        string = '{}'.format(tabs * '\t')
        for i, partition in enumerate(partitions):
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                string_device = '{}: Total {}, Used {}({:.2f}%), Free {}'.format(
                    partition.device.replace(':\\', ''),
                    convert_size(partition_usage.total),
                    convert_size(partition_usage.used),
                    partition_usage.percent,
                    convert_size(partition_usage.free),
                )
            except PermissionError as e:
                string_device = add_color('{}: PermissionError: {}'.format(partition.device.replace(':\\', ''), e))

            string += string_device
            if i < (len(partitions) - 1):
                string += ', '
    else:
        string = '{}disk space:'.format(tabs * '\t')
        for i, partition in enumerate(partitions):
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                string_device = '\n{}\t{}:'.format(tabs * '\t', partition.device.replace(':\\', ''))
                string_device += '\n{}\t\tTotal {}'.format(tabs * '\t', convert_size(partition_usage.total))
                string_device += '\n{}\t\tUsed {}({:.2f}%)'.format(tabs * '\t', partition_usage.used,
                                                                   partition_usage.percent)
                string_device += '\n{}\t\tFree {}'.format(tabs * '\t', convert_size(partition_usage.free))
            except PermissionError as e:
                string_device = '\n{}\t{}:'.format(tabs * '\t', partition.device.replace(':\\', ''))
                string_device += add_color('\n{}\t\tPermissionError: {}'.format(tabs * '\t', e))

            string += string_device
            if i < (len(partitions) - 1):
                string += ', '
    return string


def ram_size(one_liner: bool = False, tabs: int = 1):
    """
    :param one_liner:
    :param tabs:
    :return:
    see ram_size_test()
    """
    mem = psutil.virtual_memory()
    if one_liner:
        string = '{}Total {}, Used {}({}%), Available {} '.format(
            tabs * '\t',
            convert_size(mem.total),
            convert_size(mem.used),
            mem.percent,
            convert_size(mem.available),
        )
    else:
        string = '{}RAM Info:'.format(tabs * '\t')
        string += '\n{}\tTotal {}'.format(tabs * '\t', convert_size(mem.total))
        string += '\n{}\tUsed {}({}%)'.format(tabs * '\t', convert_size(mem.used), mem.percent)
        string += '\n{}\tAvailable {}'.format(tabs * '\t', convert_size(mem.available))
    return string


def cpu_info(one_liner: bool = False, tabs: int = 1):
    """
    :param one_liner:
    :param tabs:
    :return:
    see cpu_info_test()
    """
    if one_liner:
        uname = platform.uname()
        cpufreq = psutil.cpu_freq()
        string = '{}'.format(tabs * '\t')
        string += '{}, '.format(uname.machine)
        string += '{}, '.format(uname.processor)
        string += 'Physical cores {}, '.format(psutil.cpu_count(logical=False))
        string += 'Total cores {}, '.format(psutil.cpu_count(logical=True))
        string += 'Frequency {:.2f}Mhz, '.format(cpufreq.current)
        string += 'CPU Usage {}%'.format(psutil.cpu_percent())
    else:
        uname = platform.uname()
        cpufreq = psutil.cpu_freq()
        string = '{}System Info:'.format(tabs * '\t')
        string += '\n{}\tSystem {}'.format(tabs * '\t', uname.system)
        string += '\n{}\tName {}'.format(tabs * '\t', uname.node)
        string += '\n{}\tRelease {}'.format(tabs * '\t', uname.release)
        string += '\n{}\tVersion {}'.format(tabs * '\t', uname.version)
        string += '\n{}\tMachine {}'.format(tabs * '\t', uname.machine)
        string += '\n{}\tProcessor {}'.format(tabs * '\t', uname.processor)
        string += '\n{}\tPhysical cores {}'.format(tabs * '\t', (psutil.cpu_count(logical=False)))
        string += '\n{}\tTotal cores {}'.format(tabs * '\t', (psutil.cpu_count(logical=True)))
        string += '\n{}\tMax Frequency {:.2f}Mhz'.format(tabs * '\t', cpufreq.max)
        string += '\n{}\tMin Frequency {:.2f}Mhz'.format(tabs * '\t', cpufreq.min)
        string += '\n{}\tCurrent Frequency {:.2f}Mhz'.format(tabs * '\t', cpufreq.current)
        string += '\n{}\tCPU Usage {}%'.format(tabs * '\t', psutil.cpu_percent())
        # noinspection PyTypeChecker
        per_core_raw: list = psutil.cpu_percent(percpu=True, interval=1)
        per_core = ''
        for i, percentage in enumerate(per_core_raw):
            # print(f"Core {i}: {percentage}%")
            per_core += 'C{}-{}% '.format(i, percentage)
        string += '\n{}\tCPU Usage per core {}'.format(tabs * '\t', per_core)
    return string


def wizzi_utils_requirements():
    print('A snapshot of my environment packages:')

    misc_file = get_file_name(depth=1)
    misc_dir = os.path.dirname(misc_file)
    main_src_dir = '{}/../'.format(misc_dir)

    req_file = '{}/wizzi_utils_requirements.txt'.format(main_src_dir)
    _ = read_file_lines(fp=req_file, ack=True)
    return


def last_exception(tabs: int = 1):
    """
    prints last_exception occurred
    :return:
    """
    exception_error(sys.exc_info()[1], tabs=tabs)
    return


def file_or_folder_size(path: str, as_str: bool = True, ack: bool = False, tabs: int = 1) -> (int, str):
    """
    :param path:
    :param as_str:
    :param ack:
    :param tabs:
    :return: file or folder size
      as_str True: as str - see convert_size()
      as_str False: in bytes
    see file_or_folder_size_test()
    """
    size_s = 0
    if os.path.exists(path):
        if os.path.isfile(path):
            size_s = os.stat(path).st_size
            prefix = 'file'
        else:
            for _dir_path, dir_names, file_names in os.walk(path):
                for f in file_names:
                    fp = os.path.join(_dir_path, f)
                    if not os.path.islink(fp):  # not symbolic link
                        size_s += os.path.getsize(fp)
            # size_s = sum(d.stat().st_size for d in os.scandir(path) if d.is_file())
            prefix = 'folder'

        if as_str:
            size_s = convert_size(size_s)
        if ack:
            print('{}{} {} size = {}'.format(tabs * '\t', prefix, path, size_s))
    else:
        exception_error(NOT_FOUND.format(path), real_exception=False, tabs=tabs)
    return size_s


def full_path_no_limit(path_file_or_dir: str) -> str:
    """
    without this basename can't exceed 70 chars
    with    this basename can't exceed 255 chars
    return full path with no size limit on the length
    :param path_file_or_dir:
    :return:
    """
    path_file_or_dir = "\\\\?\\" + os.path.abspath(path_file_or_dir)
    return path_file_or_dir


def extract_file(
        src: str,
        dst_folder: str = 'ex',
        file_type: str = 'zip',
        ack: bool = True,
        tabs: int = 1):
    """
    :param src: compressed file (zip, 7z, tar)
    :param dst_folder: destination folder.
        if folder not empty and in the zip there is a file that exists in the folder, it will be overwritten
    :param file_type: zip, 7z, tar
    :param ack:
    :param tabs:
    :return:
    see compress_and_extract_test()
    """
    if os.path.exists(src):
        if file_type in ['zip', 'tar', '7z']:
            if file_type == '7z':
                try:
                    if ('7z', ['.7z'], '7zip archive') not in shutil.get_unpack_formats():
                        # noinspection PyPackageRequirements,PyUnresolvedReferences
                        from py7zr import unpack_7zarchive
                        shutil.register_unpack_format(name='7z', extensions=['.7z'], function=unpack_7zarchive,
                                                      description='7zip archive')
                except (ModuleNotFoundError, AttributeError) as e:
                    exception_error(e, real_exception=True, tabs=tabs)
                    return
            shutil.unpack_archive(filename=src, extract_dir=dst_folder)

            if ack:
                msg = '{}extracted {}({}) to {}({})'.format(
                    tabs * '\t',
                    src, file_or_folder_size(src),
                    dst_folder, file_or_folder_size(dst_folder)
                )
                print(add_color(msg, ops=SUCCESS_C))
    else:
        exception_error(NOT_FOUND.format(src), real_exception=False, tabs=tabs)
    return


def compress_file_or_folder(
        src: str,
        dst_path: str = './file',
        file_type: str = 'zip',
        ack: bool = True,
        tabs: int = 1) -> None:
    """
    :param src: folder or file to compress
    :param dst_path: the compressed file name (no suffix)
        e.g. if file_type=='zip':  dst_path=./a -> ./a.zip
    :param file_type: 'zip', 'tar', '7z'
    :param ack:
    :param tabs:
    :return:
    see compress_and_extract_test()
    """
    if os.path.exists(src):
        if file_type in ['zip', 'tar', '7z']:
            if file_type == '7z':
                try:
                    if ('7z', '7zip archive') not in shutil.get_archive_formats():
                        # noinspection PyPackageRequirements,PyUnresolvedReferences
                        from py7zr import pack_7zarchive
                        shutil.register_archive_format(name='7z', function=pack_7zarchive, description='7zip archive')
                except (ModuleNotFoundError, AttributeError) as e:
                    exception_error(e, real_exception=True, tabs=tabs)
                    return
            if os.path.isfile(src):
                src_file = os.path.basename(src)
                src_dir = os.path.dirname(src)
                shutil.make_archive(base_name=dst_path, format=file_type, root_dir=src_dir, base_dir=src_file)
            else:
                shutil.make_archive(base_name=dst_path, format=file_type, root_dir=src)

            if ack:
                dst_path += '.{}'.format(file_type)
                msg = '{}compressed {}({}) to {}({})'.format(
                    tabs * '\t',
                    src, file_or_folder_size(src),
                    dst_path, file_or_folder_size(dst_path)
                )
                print(add_color(msg, ops=SUCCESS_C))
    else:
        exception_error(NOT_FOUND.format(src), real_exception=False, tabs=tabs)
    return


class FPS:
    """
    measure times easily
    see classFPS_test()
    """

    def __init__(self, last_k: int = 100, summary_title: str = None, cache_size: int = 10000) -> None:
        """
        :param last_k: average of the last k measures
        :param summary_title:
        :param cache_size: >=last_k
        in case your program runs on an infinite loop, limit the size of the times_elapsed list.
        |times_elapsed| <= cache_size + last_k
        e.g. let cache_size=1000 and last_k=100
        till iter 1100 times_elapsed fills
        on iter 1100:
            pop first 1000 and add only the sum to run_time_sum
            times_elapsed = times_elapsed last 100 indices
        on iter 2100: |times_elapsed| = 1100
            pop first 1000 and add only the sum to run_time_sum
            times_elapsed = times_elapsed last 100 indices
        on iter 2350 run end:
        |times_elapsed| = 350
        run_time_sum is a float sum of first 2000 iters
        total time run_time_sum + sum(times_elapsed)
        total time avg = total time / iter_num
        last_k avg = sum(times_elapsed[-last_k:]) / last_k
        """
        self.timer_begin = None
        self.timer_end = None
        self.times_elapsed = []
        self.cache_size = cache_size
        if cache_size < last_k:
            self.cache_size = last_k
        self.run_time_sum = 0.0
        self.iter_num = 0
        self.summary_title = summary_title
        self.last_k = last_k
        self.avg_epsilon = 0.0001
        return

    def __del__(self) -> None:
        return

    def __str__(self) -> str:
        return ''

    def start(self, ack_progress: bool = False, tabs: int = 1, with_title: bool = False) -> None:
        if ack_progress:
            string = tabs * '\t'
            if with_title:
                string += '{}:'.format(self.summary_title)
            string += 'Iter {} Started:'.format(self.iter_num + 1)
            print(string)
        self.timer_begin = get_timer()
        self.timer_end = None
        return

    def update(self, ack_progress: bool = False, tabs: int = 1, with_title: bool = False) -> None:
        if self.timer_begin is None:
            exception_error(e='Must call FPS::start before update', real_exception=False, tabs=0)
        else:
            self.timer_end = get_timer()
            time_elapsed = self.timer_end - self.timer_begin
            self.iter_num += 1
            self.times_elapsed.append(time_elapsed)
            if len(self.times_elapsed) >= self.last_k + self.cache_size:
                # if |self.times_elapsed| > self.last_k + self.cache_size:
                # take the first 'self.cache_size' and save the sum.
                # new |self.times_elapsed|=self.last_k
                self.run_time_sum += np.sum(self.times_elapsed[:self.cache_size])
                self.times_elapsed = self.times_elapsed[-self.last_k:]

            if ack_progress:
                self.so_far(tabs, with_title)
            self.timer_begin = None
            self.timer_end = None
        return

    def get_fps(self) -> float:
        fps = -1.0
        if len(self.times_elapsed) > 0:
            avg_times = np.mean(self.times_elapsed[-self.last_k:])
            if avg_times > self.avg_epsilon:
                fps = 1 / avg_times
        return fps

    def so_far(self, tabs: int = 1, with_title: bool = False) -> None:
        if len(self.times_elapsed) > 0:
            string = tabs * '\t'
            if with_title:
                string += '{}:'.format(self.summary_title)
            string += 'Iter {} Done: '.format(self.iter_num)

            # rt_str = self.convert()
            rt_str = '{:.3f}s'.format(self.times_elapsed[-1])

            string += 'iterTime={}, '.format(rt_str)
            avg_times = np.mean(self.times_elapsed[-self.last_k:])
            if len(self.times_elapsed) > 1:
                string += 'totalTime={:.3f}s, '.format(np.sum(self.times_elapsed[-self.last_k:]))
                string += 'avgTime={:.3f}s, '.format(avg_times)

            if avg_times > self.avg_epsilon:
                fps = '{:.2f} FPS'.format(1 / avg_times)
            else:
                fps = 'FPS very high'.format(tabs * '\t')
            string += add_color(fps, ops='r')
            print(string)
        return

    def finalize(self, tabs: int = 1) -> None:
        tabs_s = tabs * '\t'
        if len(self.times_elapsed) > 0:
            # title
            title = 'Summary '
            if self.summary_title is not None:
                title += 'of {}'.format(self.summary_title)
            # add rounds to title
            title += '({} iters)'.format(self.iter_num)
            print('{}{}'.format(tabs_s, add_color(title, ops='underlined')))

            # total time
            t_time = np.sum(self.times_elapsed) + self.run_time_sum
            total = '{}\tTotal   Run Time = {:.3f}s'.format(tabs_s, t_time)
            print(total)

            # average time
            avg_times = t_time / self.iter_num
            avg_str = '{}\tAverage Run Time(all t) = {:.3f}s'.format(tabs_s, avg_times)

            # FPS
            if avg_times > self.avg_epsilon:
                fps = ' {:.2f} FPS'.format(1 / avg_times)
            else:
                fps = 'FPS very high'
            avg_str += add_color(fps, ops='r')

            print(avg_str)

            # average time last k
            avg_times_k = np.mean(self.times_elapsed[-self.last_k:])
            std_times_k = np.std(self.times_elapsed[-self.last_k:])
            avg_str_k_raw = '{}\tAverage Run Time(last {}) = {:.3f}s (std = {:.3f})'
            avg_str_k = avg_str_k_raw.format(tabs_s, self.last_k, avg_times_k, std_times_k)
            # FPS last k
            if avg_times > self.avg_epsilon:
                fps_k = ' {:.2f} FPS'.format(1 / avg_times_k)
            else:
                fps_k = 'FPS very high'
            avg_str_k += add_color(fps_k, ops='r')

            if fps_k != fps:
                print(avg_str_k)

        else:
            exception_error('Times_elapsed is empty', real_exception=False, tabs=tabs)
        return


def run_shell_command(cmd: str, ack: bool = True) -> None:
    try:
        if ack:
            print(add_color(cmd, ops='b'))
        subprocess.check_call(cmd, shell=True)

    except subprocess.CalledProcessError as e:
        exception_error(e, real_exception=True)
    return


def run_shell_command_and_get_out(cmd: str, ack_cmd: bool = True, ack_out: bool = False) -> list:
    out = ['Failed']
    try:
        if ack_cmd:
            print(add_color(cmd, ops='b'))
        out = subprocess.check_output(cmd, shell=True)
        out = out.decode("utf-8").replace('\t', '').replace('\r', '').strip()
        out = out.split('\n')
        if ack_out:
            for i, line in enumerate(out):
                print('{}){}'.format(i, line))
    except subprocess.CalledProcessError as e:
        exception_error(e, real_exception=True)
    return out


def add_sudo_to_cmd(cmd: str, sudo_password: str) -> str:
    cmd_wrap = 'echo {}|sudo -S {}'.format(sudo_password, cmd)
    return cmd_wrap


def get_pip_freeze_out(ack: bool = True) -> list:
    """
    you can check out
    https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt
    many solutions on pip freeze
    :param ack:
    :return:
    """
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    if ack:
        print('pip freeze out:')
        for i, pkg in enumerate(installed_packages_list):
            print('{}){}'.format(i, pkg))
    return installed_packages_list


def remove_colors(string: str) -> str:
    # print(string)
    # print(repr(string))
    for name, value in CONST_COLOR_MAP.items():
        # print(name)
        # print(value)
        if value in string:
            string = string.replace(value, '')
    # print(string)
    # print(repr(string))
    return string


def generate_requirements_file(fp_out: str, ack: bool = True) -> None:
    """
    :param fp_out: full path of output file.
    :param ack:
    :return:
    """
    d_name = os.path.dirname(fp_out)
    if not os.path.exists(d_name):
        exception_error(e='folder doesn\'t exists: {}'.format(os.path.abspath(d_name)))
    else:
        out = open(file=fp_out, mode='w', encoding='utf-8')
        # _internal.main(['list'])
        # _internal.main(['freeze'])
        # run_shell_command(cmd='pip list', ack=True)
        # out = run_shell_command_and_get_out(cmd='pip list', ack_cmd=True, ack_out=True)
        # run_shell_command(cmd="which python", ack=True)

        # main_wrapper()
        out.write('# main_wrapper:\n')
        out.write('# * Run started at {}\n'.format(get_time_stamp(format_s='%d-%m-%Y %H:%M:%S')))
        out.write('# * Python Version {}\n'.format(get_py_version()))
        out.write('# * Operating System {}\n'.format(remove_colors(get_system_info())))
        out.write('# * Interpreter: {}\n'.format(sys.executable))
        out.write('# * wizzi_utils Version {}\n'.format(version()))
        out.write('# * Working Dir: {}\n'.format(os.getcwd()))
        try:
            from wizzi_utils.socket.socket_tools import get_mac_address_uuid
            out.write('# * Computer Mac: {}\n'.format(remove_colors(get_mac_address_uuid())))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            print(add_color('* {}'.format(err), 'r'))

        out.write('# * CPU Info: {}\n'.format(remove_colors(cpu_info(one_liner=True, tabs=0))))
        out.write('# * Physical Memory: {}\n'.format(remove_colors(hard_disc(one_liner=True, tabs=0))))
        out.write('# * RAM: {}\n'.format(remove_colors(ram_size(one_liner=True, tabs=0))))

        try:
            from wizzi_utils.socket.socket_tools import get_active_ipv4
            out.write('# * Computer ipv4: {}\n'.format(get_active_ipv4()))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            out.write('# * {}\n'.format(err))

        cuda_msg = get_cuda_version(ack=False, tabs=0)
        cuda_msg = remove_colors(cuda_msg)
        out.write('# {}\n'.format(cuda_msg))

        try:
            from wizzi_utils.open_cv.open_cv_tools import get_cv_version
            cv_v = get_cv_version(ack=False, tabs=0)
            cv_v = remove_colors(cv_v)
            out.write('# {}\n'.format(cv_v))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            out.write('# * opencv-python not Found. error: {}\n'.format(err))

        try:
            from wizzi_utils.torch.torch_tools import get_torch_version
            torch_v = get_torch_version(ack=False, tabs=0)
            torch_v = remove_colors(torch_v)
            out.write('# {}\n'.format(torch_v))
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as err:
            out.write('# * torch not Found. error: {}\n'.format(err))

        pkgs = get_pip_freeze_out(ack=False)
        # TODO future: remove wheel, pip, setuptools ???
        # they are missing on pip freeze and appear here
        for pkg in pkgs:
            out.write('{}\n'.format(pkg))

        out.close()
        if ack:
            fp_out = os.path.abspath(fp_out)
            file_or_folder_size(path=fp_out, ack=True)
    return


def size_remain(partition: str, ack: bool = False) -> (int, str):
    """
    :param partition:
    e.g. windows: partition='C:' partition='D:'
         linux:   partition='/dev/root'
    :param ack:
    :return:
    """
    partitions = psutil.disk_partitions()
    free_space_int = None
    free_space_str = None
    for i, current_partition in enumerate(partitions):
        # print(partition, current_partition)
        if partition in current_partition.device:
            try:
                # print(current_partition.device)
                partition_usage = psutil.disk_usage(current_partition.mountpoint)
                free_space_int = partition_usage.free
                free_space_str = convert_size(free_space_int)
                if ack:
                    print('\ton {} free space int={}, str={}'.format(partition, free_space_int, free_space_str))
            except PermissionError as e:
                exception_error('PermissionError: {}'.format(e))

    return free_space_int, free_space_str


def size_remain_greater_than(partition: str, gb: int, mb: int, ack: bool = False) -> bool:
    """
    check if partition has more than `gb` + `mb` free
    :param partition:
    e.g. windows: partition='C:' partition='D:'
         linux:   partition='/dev/root'
    :param gb:
    :param mb:
    :param ack:
    :return:
    """
    greater_than = False
    # first get remaining space on partition
    free_space_int, free_space_str = size_remain(partition=partition, ack=False)
    if free_space_int is not None:
        if ack:
            # convert gb and mb into bytes
            size_bites_int = convert_size_to_bytes(size_gb=gb, size_mb=mb)
            if size_bites_int > 0:
                combined_size = gb + mb / 1000
                info = 'on {} free space {}'.format(partition, free_space_str)
                if free_space_int > size_bites_int:
                    info += ' GREATER than'
                    greater_than = True
                else:
                    info += ' SMALLER than'

                info += ' {} GB'.format(combined_size)
                if greater_than:
                    info = add_color(info, ops=SUCCESS_C2)
                else:
                    info = add_color(info, ops=FAIL_C2)
                print('\t{}'.format(info))
    return greater_than


def convert_size_to_bytes(size_gb: int, size_mb: int) -> int:
    """
    :param size_gb:  size in gigabytes
    :param size_mb:  size in gigabytes
    :return:
    size_gb = 1.0 -> size_bytes = 1073741824
    """
    size_bytes: int = 0
    if size_gb > 0:
        size_bytes = size_gb * 1024 ** 3
    if size_mb > 0:
        size_bytes += size_mb * 1024 ** 2
    return int(size_bytes)


class EMOJIS(Enum):
    # https://www.geeksforgeeks.org/python-program-to-print-emojis/
    GRINNING_FACE = "\U0001F600"
    GRINNING_FACE_WITH_BIG_EYS = "\U0001F603"
    GRINNING_FACE_WITH_SMILING_EYS = "\U0001F604"
    GRINNING_SQUINTING_FACE = "\U0001F606"
    GRINNING_FACE_WITH_SWEAT = "\U0001F605"
    ROLLING_ON_THE_FLOOR_LAUGHING = "\U0001F923"
    FACE_WITH_TEARS_OF_JOY = "\U0001F602"
    SLIGHTLY_SMILING_FACE = "\U0001F642"
    UPSIDE_DOWN_FACE = "\U0001F643"
    WINKING_FACE = "\U0001F609"
    SMILING_FACE_WITH_SMILING_EYES = "\U0001F60A"
    SMILING_FACE_WITH_HALO = "\U0001F607"
    SMILING_FACE_WITH_3_HEARTS = "\U0001F970"
    SMILING_FACE_WITH_HEART_EYES = "\U0001F60D"
    STAR_STRUCK = "\U0001F929"
    FACE_BLOWING_A_KISS = "\U0001F618"
    KISSING_FACE = "\U0001F617"
    KISSING_FACE_WITH_CLOSED_EYES = "\U0001F61A"
    KISSING_FACE_WITH_SMILING_EYES = "\U0001F619"
    FACE_SAVORING_FOOD = "\U0001F60B"
    FACE_WITH_TONGUE = "\U0001F61B"
    WINKING_FACE_WITH_TONGUE = "\U0001F61C"
    ZANY_FACE = "\U0001F92A"
    SQUINTING_FACE_WITH_TONGUE = "\U0001F61D"
    MONEY_MOUTH_FACE = "\U0001F911"
    HUGGING_FACE = "\U0001F917"
    FACE_WITH_HAND_OVER_MOUTH = "\U0001F92D"
    SHUSHING_FACE = "\U0001F92B"
    THINKING_FACE = "\U0001F914"
    ZIPPER_MOUTH_FACE = "\U0001F910"
    FACE_WITH_RAISED_EYBROW = "\U0001F928"
    NATURAL_FACE = "\U0001F610"
    EXPRESSIONLESS_FACE = "\U0001F611"
    FACE_WITHOUT_MOUTH = "\U0001F636"
    SMIRKING_FACE = "\U0001F60F"
    UNAMUSED_FACE = "\U0001F612"
    FACE_WITH_ROLLING_EYES = "\U0001F644"
    GRIMACING_FACE = "\U0001F62C"
    LYING_FACE = "\U0001F925"
    RELIEVED_FACE = "\U0001F60C"
    PENSIVE_FACE = "\U0001F614"
    SLEEPY_FACE = "\U0001F62A"
    DROOLING_FACE = "\U0001F924"
    SLEEPING_FACE = "\U0001F634"
    FACE_WITH_MEDICAL_MASK = "\U0001F637"
    FACE_WITH_THERMOMETER = "\U0001F912"
    FACE_WITH_HEAD_BANDAGE = "\U0001F915"
    NAUSEATED_FACE = "\U0001F922"

    @staticmethod
    def get_names():
        return [em.name for em in EMOJIS]

    @staticmethod
    def get_values():
        return [em.value for em in EMOJIS]

    @staticmethod
    def get_emojis():
        return [em for em in EMOJIS]


def get_emoji(emoji: EMOJIS) -> str:
    return emoji.value


def has_admin(ack: bool = True) -> (str, bool):
    """
    checks if user has admin privileges
    pycharm users:
        close all pycharm if open
        right click pycharm icon > run as administrator
        has_admin should yield True
    :param ack:
    :return:
    TODO test on linux
    """
    user_name, is_admin = 'Unknown', False
    if is_windows():
        user_name = get_env_variable(key='USERNAME')
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception as e:
            exception_error(e, real_exception=True)
            is_admin = False

    elif is_linux():
        if 'SUDO_USER' in os.environ and os.geteuid() == 0:
            user_name = get_env_variable(key='SUDO_USER')
            is_admin = True
        else:
            user_name = get_env_variable(key='USERNAME')
            is_admin = False
    if ack:
        print('Username {} is admin ? {}'.format(user_name, is_admin))
    return user_name, is_admin


def read_file_lines(
        fp: str,
        strip: bool = True,  # remove trailing white spaces from both sides
        multi_spaces: bool = True,  # change multi spaces to 1 space
        clear_n: bool = True,  # remove \n (new line)
        clear_r: bool = True,  # remove \r (cartridge)
        clear_t: bool = True,  # remove \t (tabs)
        ack: bool = False,
) -> list:
    lines = []
    if os.path.exists(fp):
        with open(fp) as file:
            lines_raw = file.readlines()
            for line in lines_raw:
                if strip:
                    line = line.strip()
                if clear_n:
                    line = line.replace('\n', '')
                if clear_r:
                    line = line.replace('\r', '')
                if clear_t:
                    line = line.replace('\t', '')
                if multi_spaces:
                    line = ' '.join(line.split())
                lines.append(line)
            if ack:
                for i, line in enumerate(lines):
                    # print('{}line {:3}: {}'.format(tabs * '\t', i, line))
                    # print('\t line {}: {}'.format(i, repr(line)))
                    print('{}'.format(line))
    else:
        exception_error(e='File {} not found'.format(os.path.abspath(fp)))
    return lines


def get_path_file_system_info(path: str, ack: bool = False) -> (str, str, str, str):
    """
    :param path: of file or dir
    :param ack:
    :return: mount name, device name, file system type
    e.g. let my_py.py script be somewhere on D: drive
         find_device_name_of_path('./my_py.py)
         dir D:/...somewhere.../my_py.py
            mount=D:
            device_name=D:
            file system type=NTFS
            free space=714.2 GB
    """
    cur_mount_name, device_name, fs_type, free_space = 'N/A', 'N/A', 'N/A', 'N/A'
    if not os.path.exists(path):
        exception_error(e='File {} not found'.format(os.path.abspath(path)))
        return cur_mount_name, device_name, fs_type, free_space

    cur_mount_name = os.path.abspath(path)
    while not os.path.ismount(cur_mount_name):
        cur_mount_name = os.path.dirname(cur_mount_name)

    for p in psutil.disk_partitions():
        if p.mountpoint == cur_mount_name:
            cur_mount_name = cur_mount_name.replace('\\', '')  # strip \ chars
            device_name = p.device.replace('\\', '')  # strip \ chars
            fs_type = p.fstype  # empty string is valid
            break
    _, free_space = size_remain(partition=device_name, ack=False)
    if ack:
        pref = 'file' if os.path.isfile(path) else 'dir'
        print(f'{pref} {os.path.abspath(path)}:')
        print(f'\tmount={cur_mount_name}')
        print(f'\tdevice_name={device_name}')
        print(f'\tfile system type={fs_type}')
        print(f'\tfree space={free_space}')
    return cur_mount_name, device_name, fs_type, free_space


def add_resource_folder_to_path(resources_dir: str) -> None:
    """
    :param resources_dir: full path of folder that the py file will "see"
    :return:
    e.g. when i wanted to record using x264 coded on Windows, there was a dll missing.
    the solution i found was to download the dll and place it next to the py file so it can "see" it.
    that worked, but now there is a dll file "stuck" next to my py file. this fuction will allow you to place the dll
    (and all other resources files) in a folder and add it to the PATH variable
    see wu.cvt.test.video_creator_size_test() for an example
    """
    if not os.path.exists(resources_dir):
        print('not found {}'.format(os.path.abspath(resources_dir)))
    elif not os.path.isdir(resources_dir):
        print('not a directory {}'.format(os.path.abspath(resources_dir)))
    else:
        path_env = get_env_variable(key='PATH')
        path_env = '{};{}'.format(path_env, os.path.abspath(resources_dir))  # add resources_dir to PATH env variable
        set_env_variable(key='PATH', val=path_env)
    return


def get_repo_root(repo_name: str = 'repo', ack: bool = False) -> str:
    """
    :param repo_name:
    :param ack:
    :return:
    anywhere from the project - we should have prefix/repo_name/suffix
    return prefix/repo_name
    prefix change on each system
    """
    repo_root_path = get_file_name(depth=1)
    if repo_name in repo_root_path:
        while not repo_root_path.endswith(repo_name):  # strip dir until we get to repo_name
            repo_root_path = os.path.dirname(repo_root_path)
        if ack:
            print('\tRepoRoot={}'.format(repo_root_path))
    else:
        exception_error('{} not found in current path'.format(repo_name))
    return repo_root_path


def rename_folder(dir_path: str, new_name: str, ack: bool = True) -> str:
    """
    renames a folder and returns the path of the new folder
    :param dir_path: path of the target folder
    :param new_name: new dir name
    :param ack:
    :return:
    """
    dir_path = os.path.abspath(dir_path)
    new_name_fp = None
    if not os.path.exists(dir_path):
        exception_error(e='folder {} not found. can\'t rename...'.format(dir_path))
    else:
        new_name_fp = '{}/{}'.format(os.path.dirname(dir_path), new_name)
        if os.path.exists(new_name_fp):
            exception_error(e='new folder name {} exist. can\'t rename...'.format(new_name_fp))
        else:
            os.rename(dir_path, new_name_fp)
            if ack:
                files_n = len(find_files_in_folder(dir_path=new_name_fp, file_suffix=''))
                print('\t{} renamed to {} ({} files)'.format(dir_path, new_name_fp, files_n))
    return new_name_fp
