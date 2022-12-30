from wizzi_utils.misc import misc_tools as mt
import os
# noinspection PyPackageRequirements
import yaml
# noinspection PyPackageRequirements
from pydrive.auth import GoogleAuth
# noinspection PyPackageRequirements
from pydrive.auth import RefreshError
# noinspection PyPackageRequirements
from pydrive.drive import GoogleDrive
# noinspection PyPackageRequirements
from pydrive.files import GoogleDriveFile
# noinspection PyPackageRequirements
from pydrive.settings import InvalidConfigError


def read_yaml_to_dict(yaml_path: str) -> dict:
    """
    :param yaml_path:
    :return:
    """
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            yaml_d = yaml.load(f, Loader=yaml.FullLoader)
    else:
        mt.exception_error(mt.NOT_FOUND.format(yaml_path), real_exception=False, tabs=1)
        yaml_d = None
    return yaml_d


class google_handler:
    """
    Before starting - know i made an assumption that a file/dir name is unique.
    From some reason, google allow same file or dir name in the same dir. e.g. 'root/a.txt' twice
    If you have duplicates - this will be notified upon creation of this class

    api: https://pythonhosted.org/PyDrive/quickstart.html
    example: https://www.dezyre.com/recipes/upload-files-to-google-drive-using-python
    instructions:
        HOST: google account that handles the project
        CLIENT: google account of any user that want to use the project above
        * HOST could be a CLIENT
     HOST:
         1 https://console.cloud.google.com/
         2 Dashboard - create project
         3 enter project - enable Api and services - type google drive and enable
         4 CREATE CREDENTIALS - client id - desktop app and download the dot json file -
            can rename to client_secret.json
     CLIENT:
         1 HOST must add CLIENT (including himself):
             in OAuth consent screen: +Add users, under test users, add user email
         2 pip install pydrive
             see more at https://pythonhosted.org/PyDrive/
         3 create dir for the client connection e.g. GDrive2
             place in it the client_secret.json file from HOST step 4
             create file settings.yaml with content:
                 client_config_backend: file
                 client_config_file: ./GDrive2/client_secret.json
                 save_credentials: True
                 save_credentials_backend: file
                 save_credentials_file: ./GDrive2/credentials.json
                 oauth_scope: ['https://www.googleapis.com/auth/drive']
         4 on first run - it will open a browser and ask to enter user and pass. user must be added on CLIENT step 1
             after first run credentials will be created in the path stated on the settings.yaml save_credentials_file
             no more login required
    """

    # noinspection PyPackageRequirements
    def __init__(self,
                 yaml_path: str,
                 tabs: int = 1,
                 dir_color: (str, list) = 'reverse',
                 file_color: (str, list) = 'blue',
                 title_color: (str, list) = 'underlined'
                 ):
        if not os.path.exists(yaml_path):
            mt.exception_error('cant find {}'.format(yaml_path), real_exception=False, tabs=0)
            print('File name settings.yaml should look like(you can copy paste):')
            print('\tclient_config_backend: file')
            print('\tclient_config_file: ./GDrive2/client_secret.json')
            print('\tsave_credentials: True')
            print('\tsave_credentials_backend: file')
            print('\tsave_credentials_file: ./GDrive2/credentials.json')
            print('\toauth_scope: [\'https://www.googleapis.com/auth/drive\']')
            print('*** assumes you created a folder named GDrive2 in current dir')
            print('*** assumes your client_secret HOST downloaded and gave you in GDrive2 named client_secret.json')
            print('*** the credentials_file will be saved after 1st run in GDrive2 under name credentials.json')
            exit(-1)
        self.yaml_path = yaml_path
        self.yaml_d = read_yaml_to_dict(self.yaml_path)
        try:
            # assuming we have valid client secret file and a settings yaml,
            # upon connection there are 3 options:
            # 1 no cred file - easy - browser opens to log in
            # 2 valid cred file - easy - connection made and no need for anything more
            # 3 expired cred file - rename the current cred file and try connecting agian to log in
            self.all_files_dict = None
            self.gauth = GoogleAuth(settings_file=yaml_path)
            self.drive = GoogleDrive(self.gauth)
            try:  # option 3
                self.__refresh_all_files_list()  # updates self.all_files_list
                # if expired cred file - this will raise refresh error
            except RefreshError as e:
                mt.exception_error(e, real_exception=True)
                if 'save_credentials_file' in self.yaml_d:
                    cred_path = self.yaml_d['save_credentials_file']
                    mt.exception_error('{} expired'.format(cred_path), real_exception=False)
                    ts = mt.get_time_stamp()
                    # rename the current just for backup
                    mt.move_file(file_src=cred_path, file_dst='{}.expired_{}'.format(cred_path, ts))
                    # reconnect to get browser log in
                    self.gauth = GoogleAuth(settings_file=yaml_path)
                    self.drive = GoogleDrive(self.gauth)
                    self.__refresh_all_files_list()  # side effect - open a browser to sign in
                else:  # no cred file found
                    raise e

            if self.all_files_dict is not None:
                duplicates_list = []
                for full_path, file_d in self.all_files_dict.items():
                    if file_d['id'] == 'Duplicated':
                        duplicates_list.append(full_path)

                if len(duplicates_list) > 0:
                    mt.exception_error('Found duplicated files or dirs:', real_exception=False, tabs=tabs)
                    for dup in duplicates_list:
                        print(mt.add_color('\t{}{}'.format(tabs * '\t', dup), ops='red'))
                    exit(-1)

            self.tabs = tabs
            self.dir_color = dir_color
            self.file_color = file_color
            self.title_color = title_color
        except InvalidConfigError as e:
            print('Ask host to do step HOST.4 in the instructions (create credentials and download them)')
            mt.exception_error('client secrets file no found on path stated in settings.yaml', real_exception=False)
            mt.exception_error(e, real_exception=True)
            exit(-1)
        except RefreshError as e:
            print('delete your credentials.json file and try again')
            mt.exception_error('next run you will be asked to log in and a new credentials.json file will be created',
                               real_exception=False)
            mt.exception_error(e, real_exception=True)
            exit(-1)
        return

    def __str__(self):
        string = '{}{}'.format(self.tabs * '\t', mt.add_color(string='google_handler:', ops=self.title_color))
        string += '\n{}'.format(
            mt.dict_as_table(self.yaml_d, title=os.path.abspath(self.yaml_path), fp=2, ack=False, tabs=self.tabs + 1))
        string += '\n\t{}tabs={}'.format(self.tabs * '\t', self.tabs)
        string += '\n\t{}dir_color={}'.format(self.tabs * '\t', self.dir_color)
        string += '\n\t{}file_color={}'.format(self.tabs * '\t', self.file_color)
        string += '\n\t{}title_color={}'.format(self.tabs * '\t', self.title_color)
        return string

    def __del__(self):
        print('google_handler destroyed')

    def __map_all_files(self, id_str: str = 'root', prefix: str = 'root') -> dict:
        """
        maps all files in google drive for access by name (and not id)
        if a file exist more than once on the same dir - can't use the map. have to use id
        :param id_str: start with root and recursively go over all files id
        :param prefix: the path till this part.
        :return: dict of dicts. each dict has name, id, type. if type is dir - also files list
        """
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(id_str)}).GetList()
        dir_dict = {}
        for file1 in file_list:
            file_name, file_id, file_type = file1['title'], file1['id'], file1['mimeType']
            full_path = '{}/{}'.format(prefix, file_name)

            if full_path in dir_dict:  # maybe found duplicate
                # if file_type == dir_dict[full_path]['type']:  # could be file named 'a' and dir named 'a'
                file_id = 'Duplicated'  # overwriting the id field

            dir_dict[full_path] = {'name': file_name, 'id': file_id, 'type': file_type}
            # print('path: {}, id: {}, type: {}, '.format(full_path, file_id, file_type))
            if file_type.endswith('folder'):
                sub_dict = self.__map_all_files(id_str=file_id, prefix=full_path)
                dir_dict.update(sub_dict)
        return dir_dict

    def __refresh_all_files_list(self) -> None:
        """
        if changes were made - this refresh the all_files_list member
        :return:
        """
        self.all_files_dict = {'root': {'name': 'root', 'id': 'root', 'type': 'application/vnd.google-apps.folder'}}
        all_files_dict = self.__map_all_files(id_str='root', prefix='root')
        self.all_files_dict.update(all_files_dict)
        return

    def __get_file(self, full_path_on_drive: str, tabs: int = 1) -> GoogleDriveFile:
        file = None
        if full_path_on_drive not in self.all_files_dict:
            mt.exception_error(mt.NOT_FOUND.format(full_path_on_drive), real_exception=False, tabs=tabs, depth=3)
        else:
            file = self.drive.CreateFile({'id': self.all_files_dict[full_path_on_drive]['id']})
            if file['mimeType'].endswith('folder'):
                file = None
                mt.exception_error('{} is a dir'.format(full_path_on_drive), real_exception=False, tabs=tabs, depth=3)
        return file

    def __get_dir(self, full_path_on_drive: str, tabs: int = 1) -> GoogleDriveFile:
        _dir = None
        if full_path_on_drive not in self.all_files_dict:
            mt.exception_error(mt.NOT_FOUND.format(full_path_on_drive), real_exception=False, tabs=tabs, depth=3)
        else:
            _dir = self.drive.CreateFile({'id': self.all_files_dict[full_path_on_drive]['id']})
            if not _dir['mimeType'].endswith('folder'):
                _dir = None
                mt.exception_error('{} is a file'.format(full_path_on_drive), real_exception=False, tabs=tabs, depth=3)
        return _dir

    def __create_file_in_dir(self, full_drive_path: str, tabs: int = 1) -> GoogleDriveFile:
        file = None
        dir_path_on_drive, new_file_name = os.path.dirname(full_drive_path), os.path.basename(full_drive_path)

        if dir_path_on_drive not in self.all_files_dict:  # dir must exist
            mt.exception_error(mt.NOT_FOUND.format(dir_path_on_drive), real_exception=False, tabs=tabs, depth=3)
        elif full_drive_path in self.all_files_dict:  # already a file with this name in dir_path_on_drive
            mt.exception_error(mt.EXISTS.format(full_drive_path), real_exception=False, tabs=tabs, depth=3)
        else:
            query = {'parents': [{'id': self.all_files_dict[dir_path_on_drive]['id']}], 'title': new_file_name}
            file = self.drive.CreateFile(metadata=query)
        return file

    def __create_dir_in_dir(self, full_drive_path: str, tabs: int = 1) -> GoogleDriveFile:
        _dir = None
        dir_path_on_drive, dir_name = os.path.dirname(full_drive_path), os.path.basename(full_drive_path)

        if dir_path_on_drive not in self.all_files_dict:  # base dir must exist
            mt.exception_error(mt.NOT_FOUND.format(dir_path_on_drive), real_exception=False, tabs=tabs, depth=3)
        elif full_drive_path in self.all_files_dict:  # already a dir with this name in dir_path_on_drive
            mt.exception_error(mt.EXISTS.format(full_drive_path), real_exception=False, tabs=tabs, depth=3)
        else:
            query = {
                'parents': [{'id': self.all_files_dict[dir_path_on_drive]['id']}],
                'title': dir_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            _dir = self.drive.CreateFile(metadata=query)
        return _dir

    def list_all_files(self, starting_dir: str = 'root', tabs: int = 1) -> None:
        """
        prints all files in a directory. if starting_dir is none - prints from root
        :param starting_dir: dir as string e.g. 'root', 'root/test' ...
        :param tabs:
        :return:
        see list_all_files_test()
        """
        title = 'Listing all files from dir {}:'.format(starting_dir)
        msg = mt.add_color(string=title, ops=self.title_color)
        print('{}{}'.format(tabs * '\t', msg))
        if starting_dir not in self.all_files_dict:
            mt.exception_error(mt.NOT_FOUND.format(starting_dir), real_exception=False, tabs=tabs + 1, depth=3)
        else:
            for full_path, file_d in self.all_files_dict.items():
                if full_path.startswith(starting_dir):
                    depth = len(full_path.split('/')) - 1  # number of dirs
                    depth -= len(starting_dir.split('/')) - 1  # normalize to be the zero tabs
                    msg = '{}'.format((depth + tabs) * '\t')
                    if file_d['type'].endswith('folder'):
                        msg += mt.add_color('{}'.format(file_d['name']), ops=self.dir_color)
                    else:
                        msg += mt.add_color('{}'.format(file_d['name']), ops=self.file_color)
                    msg += ' {}'.format(file_d['id'])
                    msg += ' {}'.format(file_d['type'])
                    msg += ' {}'.format(full_path)
                    print(msg)
        return

    def read_file(self, full_path_on_drive: str, tabs: int = 1) -> str:
        """
        if file exists on drive - reads it's content. e.g. json, txt ....
        :param full_path_on_drive:
        :param tabs:
        :return:
        see upload_read_and_delete_test()
        """
        content = None
        file = self.__get_file(full_path_on_drive)
        if file is not None:
            content = file.GetContentString()
            msg = '{}{}'.format(tabs * '\t', mt.LOADED.format(full_path_on_drive))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
        return content

    def download_file(self, full_path_on_drive: str, local_save_path: str, tabs: int = 1) -> bool:
        """
        if file exists on drive - download to local_save_path
        :param full_path_on_drive:
        :param local_save_path:
        :param tabs:
        :return:
        see upload_read_download_delete_test()
        see upload_download_delete_file_test()
        """
        success = False
        file = self.__get_file(full_path_on_drive)
        if file is not None:
            file.GetContentFile(local_save_path)
            success = True
            file_msg = '{}({})'.format(local_save_path, mt.file_or_folder_size(local_save_path))
            msg = '{}{}'.format(tabs * '\t', mt.DOWNLOADED.format(full_path_on_drive, file_msg))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
        return success

    def upload_content_to_new_file(self, dst_full_path_on_drive: str, content: str, tabs: int = 1) -> bool:
        """
        :param dst_full_path_on_drive: dir_path_on_drive|new_file_name
        :param content:
        :param tabs:
        :return:
        if dir_path_on_drive exists on drive
            and no file name dst_full_path_on_drive exists:
                upload the content to dst_full_path_on_drive
        see upload_read_download_delete_test()
        """
        success = False
        file = self.__create_file_in_dir(dst_full_path_on_drive)
        if file is not None:
            file.SetContentString(content)
            file.Upload()
            success = True
            msg = '{}{}'.format(tabs * '\t', mt.UPLOADED.format('content', dst_full_path_on_drive))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
            self.__refresh_all_files_list()
        return success

    def update_file_content(self, file_full_path_on_drive: str, new_content: str, tabs: int = 1) -> bool:
        """
        :param file_full_path_on_drive:
        :param new_content:
        :param tabs:
        :return:
        see update_file_content_test()
        """
        success = False
        file = self.__get_file(file_full_path_on_drive)
        if file is not None:
            file.SetContentString(new_content)
            file.Upload()
            success = True
            msg = '{}{} Changed content'.format(tabs * '\t', file_full_path_on_drive)
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
        return success

    def rename_file(self, file_full_path_on_drive: str, new_name: str, tabs: int = 1) -> bool:
        """
        :param file_full_path_on_drive:
        :param new_name:
        :param tabs:
        :return:
        see rename_file_test()
        """
        success = False
        file = self.__get_file(file_full_path_on_drive)
        if file is not None:
            dir_path_on_drive, new_file_name = os.path.dirname(file_full_path_on_drive), os.path.basename(
                file_full_path_on_drive)
            new_full_path_on_drive = '{}/{}'.format(dir_path_on_drive, new_name)
            if new_full_path_on_drive not in self.all_files_dict:
                file['title'] = new_name
                file.Upload()
                success = True
                msg = '{}{} Renamed to {}'.format(tabs * '\t', file_full_path_on_drive, new_full_path_on_drive)
                print(mt.add_color(msg, ops=mt.SUCCESS_C))
                self.__refresh_all_files_list()
            else:
                mt.exception_error(mt.EXISTS.format(new_full_path_on_drive), real_exception=False, tabs=tabs, depth=2)
        return success

    def upload_file(self, dst_full_path_on_drive: str, local_file_path: str, tabs: int = 1) -> bool:
        """
        :param dst_full_path_on_drive: dir_path_on_drive|new_file_name
        :param local_file_path:
        :param tabs:
        :return:
        if dir_path_on_drive exists on drive
            and no file name full_drive_path exists:
                upload the local_file_path to dst_full_path_on_drive
        see upload_delete_image_test()
        """
        success = False
        file = self.__create_file_in_dir(dst_full_path_on_drive)
        if file is not None:
            file.SetContentFile(local_file_path)
            file.Upload()
            success = True
            file_msg = '{}({})'.format(local_file_path, mt.file_or_folder_size(local_file_path))
            msg = '{}{}'.format(tabs * '\t', mt.UPLOADED.format(file_msg, dst_full_path_on_drive))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
            self.__refresh_all_files_list()
        return success

    def delete_file(self, full_path_on_drive: str, tabs: int = 1) -> bool:
        """
        deletes file if exists
        :param full_path_on_drive:
        :param tabs:
        :return:
        Future - add option to send to trash and restore from trash
        file.Trash()  # Move file to trash.
        file.UnTrash()  # Move file out of trash.
        see upload_delete_image_test()
        """
        success = False
        file = self.__get_file(full_path_on_drive)
        if file is not None:
            file.Delete()
            success = True
            msg = '{}{}'.format(tabs * '\t', mt.DELETED.format(full_path_on_drive))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
            self.__refresh_all_files_list()
        return success

    def create_dir(self, dst_full_path_on_drive: str, tabs: int = 1) -> bool:
        """
        :param dst_full_path_on_drive: dir_path_on_drive|new_dir_name
        :param tabs:
        :return:
        if dir_path_on_drive exists on drive
            and no dir name dst_full_path_on_drive exists:
                create a new dir named new_dir_name
        see create_and_delete_empty_dir_test()
        """
        success = False
        _dir = self.__create_dir_in_dir(dst_full_path_on_drive)
        if _dir is not None:
            _dir.Upload()
            success = True
            msg = '{}{}'.format(tabs * '\t', mt.CREATED.format(dst_full_path_on_drive))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
            self.__refresh_all_files_list()
        return success

    def delete_empty_dir(self, full_path_on_drive, tabs: int = 1) -> bool:
        """
        :param full_path_on_drive:
        :param tabs:
        :return:
        see create_and_delete_empty_dir_test()
        """
        success = False
        _dir = self.__get_dir(full_path_on_drive)
        if _dir is not None:
            is_empty = True
            for file_path in self.all_files_dict.keys():
                if file_path.startswith(full_path_on_drive) and file_path != full_path_on_drive:
                    is_empty = False
            if is_empty:
                _dir.Delete()
                success = True
                msg = '{}{}'.format(tabs * '\t', mt.DELETED.format(full_path_on_drive))
                print(mt.add_color(msg, ops=mt.SUCCESS_C))
                self.__refresh_all_files_list()
            else:
                mt.exception_error('{} is not empty. use delete_dir_with_files()'.format(full_path_on_drive),
                                   real_exception=False, tabs=tabs, depth=2)
        return success

    def delete_dir_with_files(self, full_path_on_drive, tabs: int = 1) -> bool:
        """
        :param full_path_on_drive:
        :param tabs:
        :return:
        see create_and_delete_dir_with_files_test()
        """
        success = False
        _dir = self.__get_dir(full_path_on_drive)
        if _dir is not None:
            files_c = 0
            for file_path in self.all_files_dict.keys():
                if file_path.startswith(full_path_on_drive) and file_path != full_path_on_drive:
                    files_c += 1

            _dir.Delete()
            success = True
            _dir_msg = '{}({} files)'.format(full_path_on_drive, files_c)
            msg = '{}{}'.format(tabs * '\t', mt.DELETED.format(_dir_msg))
            print(mt.add_color(msg, ops=mt.SUCCESS_C))
            self.__refresh_all_files_list()
        return success

    def download_dir(self, full_path_on_drive: str, local_dir_name: str, tabs: int = 1) -> bool:
        """
        :param full_path_on_drive:
        :param local_dir_name:
        :param tabs:
        :return:
        see download_dir_test()
        """
        success = False
        if os.path.exists(local_dir_name):
            mt.exception_error(mt.EXISTS.format(local_dir_name), tabs=tabs, depth=2)
        else:
            _dir = self.__get_dir(full_path_on_drive)
            if _dir is not None:
                print('{}Creating local dirs and downloading files:'.format(tabs * '\t'))
                mt.create_dir(dir_path=local_dir_name, tabs=tabs + 1)
                # create all dirs locally
                for full_path, file_d in self.all_files_dict.items():
                    if full_path.startswith(full_path_on_drive) and full_path != full_path_on_drive:
                        if file_d['type'].endswith('folder'):
                            rel_path = full_path[len(full_path_on_drive) + 1:]
                            mt.create_dir('{}/{}'.format(local_dir_name, rel_path), tabs=tabs + 1)

                # download all file to existing dir structure
                for full_path, file_d in self.all_files_dict.items():
                    if full_path.startswith(full_path_on_drive) and full_path != full_path_on_drive:
                        if not file_d['type'].endswith('folder'):
                            rel_path = full_path[len(full_path_on_drive) + 1:]
                            save_path = '{}/{}'.format(local_dir_name, rel_path)
                            self.download_file(full_path_on_drive=full_path, local_save_path=save_path, tabs=tabs + 1)

                file_msg = '{}({})'.format(local_dir_name, mt.file_or_folder_size(local_dir_name))
                msg = '{}{}'.format(tabs * '\t', mt.DOWNLOADED.format(full_path_on_drive, file_msg))
                print(mt.add_color(msg, ops=mt.SUCCESS_C))
                success = True
        return success

    def upload_dir(self, dst_full_path_on_drive: str, local_dir_name: str, tabs: int = 1) -> bool:
        """
        :param dst_full_path_on_drive:
        :param local_dir_name:
        :param tabs:
        :return:
        see upload_dir_test()
        """
        success = False
        if not os.path.exists(local_dir_name):
            mt.exception_error(mt.NOT_FOUND.format(local_dir_name), tabs=tabs, depth=2)
        else:
            print('{}Creating dirs and uploading files:'.format(tabs * '\t'))
            success_dir = self.create_dir(dst_full_path_on_drive, tabs=tabs + 1)
            if success_dir:
                # create all dirs on drive
                empty_dirs = []
                for _dir_path, dir_names, file_names in os.walk(local_dir_name):
                    for d in dir_names:
                        empty_dirs.append(d)
                for ed in empty_dirs:
                    self.create_dir('{}/{}'.format(dst_full_path_on_drive, ed), tabs=tabs + 1)

                # upload all files
                files_full_path = {}
                for _dir_path, dir_names, file_names in os.walk(local_dir_name):
                    for f in file_names:
                        # local full path
                        fp_local = '{}/{}'.format(os.path.abspath(_dir_path).replace('\\', '/'), f)

                        # drive relative path
                        fp_relative_dir = os.path.abspath(_dir_path)[len(os.path.abspath(local_dir_name)) + 1:]
                        if fp_relative_dir == '':
                            fp_drive_rel = f
                        else:
                            fp_drive_rel = '{}/{}'.format(fp_relative_dir.replace('\\', '/'), f)
                        fp_on_drive = '{}/{}'.format(dst_full_path_on_drive, fp_drive_rel)  # add 'root/.../...'

                        files_full_path[fp_local] = fp_on_drive
                for fp_local_rel, fp_on_drive in files_full_path.items():
                    self.upload_file(dst_full_path_on_drive=fp_on_drive, local_file_path=fp_local_rel, tabs=tabs + 1)

                file_msg = '{}({})'.format(local_dir_name, mt.file_or_folder_size(local_dir_name))
                msg = '{}{}'.format(tabs * '\t', mt.UPLOADED.format(file_msg, dst_full_path_on_drive))
                print(mt.add_color(msg, ops=mt.SUCCESS_C))
                success = True
                self.__refresh_all_files_list()

        return success
