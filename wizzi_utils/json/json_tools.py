import os
import json
from wizzi_utils.misc import misc_tools as mt


def json_to_string(j: dict, indent: int = -1, sort_keys: bool = False, tabs: int = 0) -> str:
    """
    :param j: dict
    :param indent: how many indents
    :param sort_keys: sort dict keys
    :param tabs:
    :return: string rep of j
    see json_to_string_test()
    """
    if indent == -1:
        indent = None
    string = json.dumps(j, indent=indent, sort_keys=sort_keys)
    if tabs > 0:
        string = '\t' * tabs + string
        string = string.replace('\n', '\n{}'.format(tabs * '\t'))
    return string


def string_to_json(j_str: str) -> json:
    """
    changes a string to a json dict
    def string_to_json_test():
    """
    return json.loads(j_str)


def load_json(file_path: str, ack: bool = True, tabs: int = 1) -> dict:
    """
    loads a dict in json format from path
    see save_load_json_test()
    """
    ret_dict = {}
    if os.path.exists(file_path):
        ret_dict = json.load(open(file_path))
        if ack:
            size_s = mt.file_or_folder_size(file_path)
            file_msg = '{}({})'.format(file_path, size_s)
            print('{}{}. {}'.format(tabs * '\t', mt.LOADED.format(file_msg),
                                    mt.CONTENT.format(json_to_string(ret_dict))))
    else:
        mt.exception_error(mt.NOT_FOUND.format(file_path), real_exception=False, tabs=tabs)
    return ret_dict


def load_jsons(files_path: list, ack: bool = True, tabs: int = 1) -> dict:
    """
    loads several of json files format from paths and concat to one dict
    asserts if a key found on 2 of the files
    see save_load_json_test()
    """
    all_in_one_dict = {}
    len_keys_json = 0
    size_s = 0
    for file_path in files_path:
        size_s += mt.file_or_folder_size(file_path, as_str=False)
        j = load_json(file_path, ack=False, tabs=tabs)
        len_keys_json += len(j)
        all_in_one_dict.update(j)

    assert len_keys_json == len(all_in_one_dict), 'Duplicated keys found: {}'.format(files_path)
    if ack:
        file_msg = '{}({})'.format(files_path, mt.convert_size(size_s))
        print('{}{}. {}'.format(tabs * '\t', mt.LOADED.format(file_msg),
                                mt.CONTENT.format(json_to_string(all_in_one_dict))))
    return all_in_one_dict


def save_json(file_path: str,
              j: dict,
              indent: int = -1,
              sort_keys: bool = False,
              ack: bool = True,
              tabs: int = 1
              ) -> None:
    """
    see save_load_json_test()
    """
    json.dump(j, open(file_path, 'w'), indent=indent, sort_keys=sort_keys)
    if ack:
        size_s = mt.file_or_folder_size(file_path)
        file_msg = '{}({})'.format(file_path, size_s)
        print('{}{}. {}'.format(tabs * '\t', mt.SAVED.format(file_msg),
                                mt.CONTENT.format(json_to_string(j, indent=indent, sort_keys=sort_keys))))
    return
