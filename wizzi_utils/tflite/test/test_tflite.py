from wizzi_utils.tflite import tflite_tools as tflt
from wizzi_utils.misc import misc_tools as mt


def get_tflite_version_test():
    mt.get_function_name(ack=True, tabs=0)
    tflt.get_tflite_version(ack=True)
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    get_tflite_version_test()
    print('{}'.format('-' * 20))
    return
