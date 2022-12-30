from wizzi_utils.json import json_tools as jt
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt


def json_to_string_test():
    mt.get_function_name(ack=True, tabs=0)
    cfg = {'x': 3, 'a': 'dx', 'b': None}
    one_liner = jt.json_to_string(cfg, indent=-1, sort_keys=False)
    print('\tdata in a one liner: {}'.format(one_liner))
    print('\tdata with 4 indentations:')
    print(jt.json_to_string(cfg, indent=4, sort_keys=False, tabs=1))  # pretty print - 4 indentations
    one_liner_sorted = jt.json_to_string(cfg, indent=-1, sort_keys=True)
    print('\tdata in a one liner with sorted keys: {}'.format(one_liner_sorted))
    return


def string_to_json_test():
    mt.get_function_name(ack=True, tabs=0)
    j_str = '{"x": 3, "a": "dx", "b":null}'
    j = jt.string_to_json(j_str)
    one_liner = jt.json_to_string(j, indent=-1, sort_keys=False)
    print('\t{}'.format(one_liner))
    return


def save_load_json_test():
    mt.get_function_name(ack=True, tabs=0)
    j = {'x': 3, 'a': 'dx', 'b': None}
    j2 = {'x2': 39, 'aa': 'dax', 'b7': None}
    mt.create_dir(mtt.TEMP_FOLDER1)
    j_path = '{}/{}.json'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    j2_path = '{}/{}2.json'.format(mtt.TEMP_FOLDER1, mtt.JUST_A_NAME)
    jt.save_json(j_path, j, ack=True)
    jt.save_json(j2_path, j2, ack=True)
    _ = jt.load_json(j_path, ack=True)  # just 1 json file
    _ = jt.load_jsons([j_path, j2_path], ack=True)  # load multiple files
    mt.delete_dir_with_files(mtt.TEMP_FOLDER1)
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    json_to_string_test()
    string_to_json_test()
    save_load_json_test()
    print('{}'.format('-' * 20))
    return
