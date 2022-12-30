import numpy as np
import os
import abc
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.socket import socket_tools as st
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.pyplot import pyplot_tools as pyplt
from wizzi_utils.models import models_configs as cfg
# noinspection PyPackageRequirements
import cv2


class BaseModel:
    def __init__(self, save_load_dir: str, model_name: str):
        """
        :param save_load_dir: where the model is saved (or will be if not exists)
        :param model_name: valid name from models_config.py
        see:
        class_Models_test()
        """
        self.model_name = model_name
        self.local_path = save_load_dir
        if not os.path.exists(save_load_dir):
            mt.create_dir(save_load_dir, ack=False)
        return

    @staticmethod
    def model_name_valid(model_name: str) -> bool:
        if model_name not in cfg.MODELS_CONFIG:
            err = 'model {} not found in MODELS_CONFIG. options: {}'
            mt.exception_error(err.format(model_name, list(cfg.MODELS_CONFIG.keys())), real_exception=False)
            valid = False
        else:
            valid = True
        return valid

    @staticmethod
    def model_type_valid(model_type: str, model_type_needed: str) -> bool:
        if model_type not in model_type_needed:
            err = 'model type is {} but {} needed'.format(model_type, model_type_needed)
            mt.exception_error(err, real_exception=False)
            valid = False
        else:
            valid = True
        return valid

    @staticmethod
    def _download_if_needed(local_path: str, url_dict: dict) -> None:
        if not os.path.exists(local_path):
            download_style = url_dict['download_style']
            url = url_dict['url']
            if download_style == cfg.DownloadStyle.Direct.value:
                st.download_file(url=url, dst_path=local_path)
            elif download_style in [cfg.DownloadStyle.Tar.value, cfg.DownloadStyle.Zip.value]:
                root_d = os.path.dirname(local_path)
                comp_path = '{}/compressed.{}'.format(root_d, download_style)
                st.download_file(url=url, dst_path=comp_path)  # download compressed
                extracted_folder = '{}/ex'.format(root_d)
                if mt.is_windows():
                    extracted_folder = mt.full_path_no_limit(extracted_folder)

                mt.extract_file(src=comp_path, dst_folder=extracted_folder,
                                file_type=download_style)  # extract
                target_file_in_tar = url_dict['file_to_look']
                # find target file
                target_files = mt.find_files_in_folder(dir_path=extracted_folder, file_suffix=target_file_in_tar)
                if len(target_files) != 1:
                    err_msg = 'not found or found more than 1 target file in downloaded folder: {}'.format(target_files)
                    mt.exception_error(err_msg, real_exception=False)
                    exit(-1)
                mt.move_file(file_src=target_files[0], file_dst=local_path)  # move file

                mt.delete_file(file=comp_path)  # clean up tar\zip
                mt.delete_dir_with_files(dir_path=extracted_folder)  # clean up extracted_folder
        return

    @staticmethod
    def get_models_from_cfg(job: str, m_type: str = None, ack: bool = False, tabs: int = 1) -> list:
        """
        :param job: from cfg.Jobs
        :param m_type: Cv2 or Tfl or None
        :param ack:
        :param tabs:
        :return:
        """
        model_names = []
        count = 0
        for i, (m_name, m_dict) in enumerate(cfg.MODELS_CONFIG.items()):
            if m_dict['job'] == job:
                if m_type is None or (m_type is not None and m_dict['model_type'].startswith(m_type)):
                    model_names.append(m_name)
                    if ack:
                        count += 1
                        mt.dict_as_table(table=m_dict, title='{}){}'.format(count, m_name), tabs=tabs)

        return model_names

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def to_string(self, tabs: int = 1) -> str:
        print('abs method - needs implementation')
        exit(-1)
        return ''

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def detect_cv_img(
            self,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param cv_img: open cv image
        :param fp: float precision on the score percentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts. see extract_results()
        """
        print('abs method - needs implementation')
        exit(-1)
        return []


class OdBaseModel(BaseModel):
    DEFAULT_COLOR_DICT = {
        'label_bbox': 'black',  # if draw_labels - text bg color
        'text': 'white',  # if draw_labels - text bg color
        'sub_image': 'blue',  # if draw_sub_image - sub image bbox color
        'default_bbox': 'red',  # bbox over the detection
    }

    def __init__(
            self,
            save_load_dir: str,
            model_name: str,
            allowed_class: list = None,
    ):
        super().__init__(save_load_dir=save_load_dir, model_name=model_name)
        if not self.model_name_valid(model_name):
            exit(-1)
        self.model_cfg = cfg.MODELS_CONFIG[model_name]
        self.labels = self.model_cfg['labels_dict']['labels']
        self.allowed_classes = self.labels if allowed_class is None else allowed_class
        self.device = 'CPU'
        return

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def to_string(self, tabs: int = 1) -> str:
        print('abs method - needs implementation')
        exit(-1)
        return ''

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def detect_cv_img(
            self,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param cv_img: open cv image
        :param fp: float precision on the score percentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts. see extract_results()
        """
        print('abs method - needs implementation')
        exit(-1)
        return []

    @staticmethod
    def get_object_detection_models(m_type: str = None, ack: bool = False, tabs: int = 1) -> list:
        """
        :param m_type: Cv2 or Tfl or None
        :param ack:
        :param tabs:
        :return:
        """
        model_names = BaseModel.get_models_from_cfg(job=cfg.Jobs.OBJECT_DETECTION.value, m_type=m_type, ack=ack,
                                                    tabs=tabs)
        return model_names

    def draw_detections(
            self,
            detections: list,
            cv_img: np.array,
            colors_d: dict = None,
            draw_labels: bool = False,
            draw_tl_image: bool = False,
            draw_sub_image: bool = False,
            header_tl: dict = None,
            header_bl: dict = None,
            header_tr: dict = None,
            header_br: dict = None,
    ) -> None:
        """
        :param detections: output of self.classify_cv_img() on od models
        :param colors_d: colors in str form:
            bbox color
            label_bbox color
            text color
            if class_id_bbox exist - color the bbox of that class in the given color
            e.g. colors_d={
                    'label_bbox': 'black',  # if draw_labels - text bg color
                    'text': 'white',  # if draw_labels - text bg color
                    'sub_image': 'blue',  # if draw_sub_image - sub image bbox color
                    'default_bbox': 'red',  # bbox over the detection
                    'person_bbox': 'black', # custom color per class person
                    'dog_bbox': 'lime',  # custom color per class dog
                    'cat_bbox': 'magenta',  # custom color per class cat
                },
        :param cv_img: the same that was given input to self.classify_cv_img()
        :param draw_labels: draw label(label_bbox and text on it) on the bbox
        :param draw_tl_image: draw traffic light if exists - has colors in it
        :param draw_sub_image: draw sub image if exists
        :param header_tl: dict with args for cvt.add_header on top_left corner
        :param header_bl: dict with args for cvt.add_header on bottom_left corner
        :param header_tr: dict with args for cvt.add_header on top_right corner
        :param header_br: dict with args for cvt.add_header on bottom_right corner
        header e.g.
        header_tl={'text': 'wizzi_utils', 'x_offset': 0, 'text_color': 'aqua', 'with_rect': True, 'bg_color': 'black',
               'bg_font_scale': 1}
        :return:
        """
        if colors_d is None:
            colors_d = self.DEFAULT_COLOR_DICT
        for detection in detections:
            label = detection['label']
            score_percentage = detection['score_percentage']
            x0 = detection['bbox']['x0']
            y0 = detection['bbox']['y0']
            x1 = detection['bbox']['x1']
            y1 = detection['bbox']['y1']
            traffic_light_d = detection['traffic_light'] if 'traffic_light' in detection else {}
            bbox_sub_image_d = detection['bbox_sub_image'] if 'bbox_sub_image' in detection else None

            # DRAW BBOX
            k = '{}_bbox'.format(label)  # check custom color for this class
            color_bgr = pyplt.get_bgr_color(colors_d[k]) if k in colors_d else pyplt.get_bgr_color(
                colors_d['default_bbox'])
            cv2.rectangle(img=cv_img, pt1=(x0, y0), pt2=(x1, y1), color=color_bgr, thickness=2)

            if draw_tl_image:  # DRAW TL
                for loc, point_and_color in traffic_light_d.items():
                    cv2.circle(cv_img, center=tuple(point_and_color['point']), radius=3,
                               color=pyplt.get_bgr_color(point_and_color['color']), thickness=-1)

            if draw_sub_image and bbox_sub_image_d is not None:  # DRAW sub image
                pt1 = (bbox_sub_image_d['x0'], bbox_sub_image_d['y0'])
                pt2 = (bbox_sub_image_d['x1'], bbox_sub_image_d['y1'])
                cv2.rectangle(img=cv_img, pt1=pt1, pt2=pt2, color=pyplt.get_bgr_color(colors_d['sub_image']),
                              thickness=1)  # sub image

            if draw_labels:
                if 'custom_label' in detection:
                    label = detection['custom_label']
                label_conf = '{}({:.1f}%)'.format(label, score_percentage)  # DRAW labels
                cvt.add_text(cv_img, header=label_conf, pos=(x0, y1), text_color=colors_d['text'], with_rect=True,
                             bg_color=colors_d['label_bbox'], bg_font_scale=2)

        if header_tl is not None:
            h = header_tl
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.TOP_LEFT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        if header_bl is not None:
            h = header_bl
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.BOTTOM_LEFT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        if header_tr is not None:
            h = header_tr
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.TOP_RIGHT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        if header_br is not None:
            h = header_br
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.BOTTOM_RIGHT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        return

    @staticmethod
    def add_traffic_light_to_detections(
            detections: list,
            traffic_light_p: dict,
            ack: bool = False,
            tabs: int = 1
    ) -> list:
        """
        :param detections: detections from classify_cv_img()
        :param traffic_light_p: dict of loc to float percentage

            if not none get 3 2d points in a traffic_light form
            e.g. traffic_light={
                    # x of all traffic_light points is frame_width / 2
                    'up': 0.2,  # red light will be take from y = frame_height * 0.2
                    'mid': 0.3,  # yellow light will be take from y = frame_height * 0.3
                    'down': 0.4  # green light will be take from y = frame_height * 0.3
                }
        :param ack:
        :param tabs:
        :return: for each detection in detections: add entry 'traffic_light' with a dict with keys up, mid, down
                    each has location to dict of point and color:
            e.g.
                'traffic_light'= { # x_mid = frame_width / 2
                'up': {'point': [x_mid, y_up], 'color': 'red'},  # y_up = frame_height * traffic_light['up']
                'mid': {'point': [x_mid, y_mid], 'color': 'yellow'},
                'down': {'point': [x_mid, y_down], 'color': 'green'}
                }
        """
        if ack:
            print('{}traffic light of {} detections:'.format(tabs * '\t', len(detections)))
        for detection_d in detections:
            label = detection_d['label']
            # score = detection_d['score_percentage']
            x0 = detection_d['bbox']['x0']
            y0 = detection_d['bbox']['y0']
            x1 = detection_d['bbox']['x1']
            y1 = detection_d['bbox']['y1']
            y_dist = y0 - y1  # image is flipped on y axis
            x_dist = x1 - x0
            # prepare 'traffic_light' points
            # on x middle
            # on y 20% 30% 40% from the top
            x_mid = int(x0 + 0.5 * x_dist)
            y_up = int(y0 - traffic_light_p['up'] * y_dist)
            y_mid = int(y0 - traffic_light_p['mid'] * y_dist)
            y_down = int(y0 - traffic_light_p['down'] * y_dist)
            traffic_light_out = {
                'up': {'point': [x_mid, y_up], 'color': 'red'},
                'mid': {'point': [x_mid, y_mid], 'color': 'yellow'},
                'down': {'point': [x_mid, y_down], 'color': 'green'}
            }
            detection_d['traffic_light'] = traffic_light_out
            if ack:
                print('{}\ttraffic light of detection {}: {}'.format(tabs * '\t', label, traffic_light_out))

        return detections

    @staticmethod
    def add_sub_sub_image_to_detection(
            detections: list,
            cv_img: np.array,
            bbox_image_p: dict,
            ack: bool = False,
            tabs: int = 1
    ) -> list:
        """
        :param detections:
        :param cv_img:
        :param bbox_image_p: dict that specify how much from the bbox to save
                bbox_image_p = {  # all bbox
                    'x_left_start': 0,
                    'x_left_end': 1,
                    'y_top_start': 0,
                    'y_top_end': 1,
                }
                bbox_image_p = {  # sub bbox
                    'x_left_start': 0.2,  x left: start from 20% of the image
                    'x_left_end': 0.9, x right: end at 90%
                    'y_top_start': 0.2, y top: start at 20%
                    'y_top_end': 0.6, y bottom: end at 60%
                }
        :param ack: print sub image dict
        :param tabs:
        :return: for each detection in detections: add entry 'bbox_sub_image' with a dict with keys image,x0,x1,y0,y1
                    if you imshow - you will get the bbox of the detection (according to bbox_image_p)
        """
        if ack:
            print('{}bbox_sub_image of {} detections:'.format(tabs * '\t', len(detections)))
        for detection_d in detections:
            label = detection_d['label']
            # confidence = detection_d['score_percentage']
            bbox = detection_d['bbox']
            x0, x1 = bbox['x0'], bbox['x1']
            y0, y1 = bbox['y0'], bbox['y1']

            y_dist = y1 - y0
            x_dist = x1 - x0
            # prepare sub sub image (a part of the bbox)
            # the origin is left - top. (x0,y0) is top left. (x1,y1) is bottom right
            sub_y0 = int(y0 + bbox_image_p['y_top_start'] * y_dist)
            sub_y1 = int(y0 + bbox_image_p['y_top_end'] * y_dist)
            sub_x0 = int(x0 + bbox_image_p['x_left_start'] * x_dist)
            sub_x1 = int(x0 + bbox_image_p['x_left_end'] * x_dist)

            if 0 <= sub_x0 < sub_x1 and 0 <= sub_y0 < sub_y1:
                detection_d['bbox_sub_image'] = {
                    'image': cv_img[sub_y0:sub_y1, sub_x0:sub_x1].tolist(),
                    'x0': sub_x0,
                    'y0': sub_y0,
                    'x1': sub_x1,
                    'y1': sub_y1,
                }
                # cvt.display_open_cv_image(detection_d['bbox_sub_image']['image'])
            else:
                detection_d['bbox_sub_image'] = None

            if ack:
                string = '{}\tbbox_sub_image of detection {}:'.format(tabs * '\t', label)
                bbox_sub = detection_d['bbox_sub_image']
                if bbox_sub is not None:
                    string += ' x0={} y0={} x1={} y1={} '.format(bbox_sub['x0'], bbox_sub['y0'], bbox_sub['x1'],
                                                                 bbox_sub['y1'])
                    string += '{}'.format(mt.to_str(np.array(bbox_sub['image']), title='sub_img'))
                    print(string)
                else:
                    print('{} is None'.format(string))
        return detections


class PdBaseModel(BaseModel):
    DEFAULT_COLOR_DICT = {
        'bbox_c': 'blue',
        'sub_image_c': 'darkorange',
        'text_c': 'white',
    }
    NOT_FOUND_VALUE = -1
    NOT_FOUND_PAIR = [NOT_FOUND_VALUE, NOT_FOUND_VALUE]

    def __init__(
            self,
            save_load_dir: str,
            model_name: str,
            allowed_joint_names: list = None,
            min_p_valid_joints: float = 0.3,
    ):
        super().__init__(save_load_dir=save_load_dir, model_name=model_name)
        if not self.model_name_valid(model_name):
            exit(-1)
        self.model_cfg = cfg.MODELS_CONFIG[model_name]
        self.joint_names = self.model_cfg['joint_names']
        self.pairs_indices = self.model_cfg['pairs_indices']
        # grab default colors and convert to bgr
        self.pairs_indices_colors = [pyplt.get_bgr_color(c) for c in self.model_cfg['pairs_indices_colors']]
        self.joint_colors = [pyplt.get_bgr_color(c) for c in self.model_cfg['joint_colors']]
        self.allowed_joint_names = list(
            self.joint_names.values()) if allowed_joint_names is None else allowed_joint_names
        self.min_p_valid_joints = min_p_valid_joints
        self.device = 'CPU'
        return

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def to_string(self, tabs: int = 1) -> str:
        print('abs method - needs implementation')
        exit(-1)
        return ''

    @staticmethod
    def get_pose_detection_models(m_type: str = None, ack: bool = False, tabs: int = 1) -> list:
        """
        :param m_type: Cv2 or Tfl or None
        :param ack:
        :param tabs:
        :return:
        """
        model_names = BaseModel.get_models_from_cfg(job=cfg.Jobs.POSE_DETECTION.value, m_type=m_type, ack=ack,
                                                    tabs=tabs)
        return model_names

    # noinspection PyUnusedLocal
    @abc.abstractmethod
    def detect_cv_img(
            self,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param cv_img: open cv image
        :param fp: float precision on the score percentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts. see extract_results()
        """
        print('abs method - needs implementation')
        exit(-1)
        return []

    def draw_detections(
            self,
            detections: list,
            cv_img: np.array,
            colors_d: dict,
            draw_joints: bool = True,
            draw_labels: bool = False,
            draw_edges: bool = False,
            draw_bbox: bool = False,
            draw_sub_image: bool = False,
            header_tl: dict = None,
            header_bl: dict = None,
            header_tr: dict = None,
            header_br: dict = None,
    ) -> None:
        """
        :param detections: output of self.classify_cv_img()
        :param colors_d: colors in str form:
            bbox_c: bbox color
            sub_image_c: sub image color
            text_c: text color
            joints_c: str or list of size self.joint_names. if None - has default in cfg
            edge_c: str or list of size self.pairs_indices. if None - has default in cfg
            e.g. colors_d={
                    'bbox_c': 'blue'
                    'sub_image_c': 'black'
                    'text_c': 'black',
                    'joints_c': str or list of size self.joint_names. if None - has default in cfg
                    'edge_c': str or list of size self.pairs_indices. if None - has default in cfg
                },
        :param cv_img: the same that was given input to self.classify_cv_img()
        :param draw_joints: draw joints
        :param draw_labels: draw joint ids (valid if draw_joints True)
        :param draw_edges: draw edges between joints
        :param draw_bbox: draw bbox around poses
        :param draw_sub_image: draw sub image from the bbox around the poses
        :param header_tl: dict with args for cvt.add_header on top_left corner
        :param header_bl: dict with args for cvt.add_header on bottom_left corner
        :param header_tr: dict with args for cvt.add_header on top_right corner
        :param header_br: dict with args for cvt.add_header on bottom_right corner
        header e.g.
        header_tl={'text': 'wizzi_utils', 'x_offset': 0, 'text_color': 'aqua', 'with_rect': True, 'bg_color': 'black',
               'bg_font_scale': 1}
        :return:
        """
        if colors_d is None:
            colors_d = self.DEFAULT_COLOR_DICT

        for detection in detections:  # each detection is a pose
            names = detection['joint_names_list']
            ids = detection['joint_ids_list']
            xys = detection['joint_x_y_list']
            # zs = detection['joint_z_list'] if 'joint_z_list' in detection else [self.NOT_FOUND_VALUE] * len(names)
            scores = detection['score_percentages_list']
            pose_bbox = detection['bbox']
            bbox_sub_image_d = detection['bbox_sub_image'] if 'bbox_sub_image' in detection else None

            if draw_bbox:  # DRAW BBOX
                color_bgr = pyplt.get_bgr_color(colors_d['bbox_c'])
                pt1 = (pose_bbox['x0'], pose_bbox['y0'])
                pt2 = (pose_bbox['x1'], pose_bbox['y1'])
                cv2.rectangle(img=cv_img, pt1=pt1, pt2=pt2, color=color_bgr, thickness=2)

            if draw_sub_image and bbox_sub_image_d is not None:  # DRAW sub image
                color_bgr = pyplt.get_bgr_color(colors_d['sub_image_c'])
                pt1 = (bbox_sub_image_d['x0'], bbox_sub_image_d['y0'])
                pt2 = (bbox_sub_image_d['x1'], bbox_sub_image_d['y1'])
                cv2.rectangle(img=cv_img, pt1=pt1, pt2=pt2, color=color_bgr, thickness=1)  # sub image

            # DRAW EDGES: lines connecting the edges
            if draw_edges:
                if 'edge_c' in colors_d:
                    edges_colors = colors_d['edge_c']
                    if mt.is_str(edges_colors):  # 1 color for all
                        edges_colors = [pyplt.get_bgr_color(edges_colors)] * len(self.pairs_indices)
                    else:  # list of colors - must be in len(self.joint_names)
                        edges_colors = [pyplt.get_bgr_color(c) for c in edges_colors]
                else:  # take default colors
                    edges_colors = self.pairs_indices_colors
                for pair, bgr_c in zip(self.pairs_indices, edges_colors):
                    partA, partB = pair
                    if xys[partA] != self.NOT_FOUND_PAIR and xys[partB] != self.NOT_FOUND_PAIR:
                        cv2.line(cv_img, pt1=tuple(xys[partA]), pt2=tuple(xys[partB]), color=bgr_c, thickness=2)

            # DRAW JOINTS: circle and joint id if asked
            if draw_joints:
                text_color_str = colors_d['text_c']
                if 'joint_colors' in colors_d:
                    joints_colors = colors_d['joint_colors']
                    if mt.is_str(joints_colors):  # 1 color for all
                        joints_colors = [pyplt.get_bgr_color(joints_colors)] * len(self.joint_names)
                    else:  # list of colors - must be in len(self.joint_names)
                        joints_colors = [pyplt.get_bgr_color(c) for c in joints_colors]
                else:  # take default colors
                    joints_colors = self.joint_colors

                for name, jid, xy, score, bgr_c in zip(names, ids, xys, scores, joints_colors):
                    if xy != self.NOT_FOUND_PAIR:
                        cv2.circle(cv_img, center=tuple(xy), radius=5, color=bgr_c, thickness=-1)
                        if draw_labels:
                            label_xy = (xy[0] - 5, xy[1])
                            label = '{}'.format(jid)
                            cvt.add_text(cv_img, header=label, pos=label_xy, text_color=text_color_str, with_rect=False,
                                         bg_font_scale=2)
        if header_tl is not None:
            h = header_tl
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.TOP_LEFT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        if header_bl is not None:
            h = header_bl
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.BOTTOM_LEFT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        if header_tr is not None:
            h = header_tr
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.TOP_RIGHT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        if header_br is not None:
            h = header_br
            cvt.add_header(cv_img, header=h['text'], loc=pyplt.Location.BOTTOM_RIGHT.value, x_offset=h['x_offset'],
                           text_color=h['text_color'], with_rect=h['with_rect'], bg_color=h['bg_color'],
                           bg_font_scale=h['bg_font_scale'])
        return

    @staticmethod
    def get_bbox_dict(joint_x_y_list: list) -> dict:
        x0, y0, x1, y1 = np.inf, np.inf, -1, -1
        for x_y in joint_x_y_list:
            if x_y != PdBaseModel.NOT_FOUND_PAIR:
                x, y = x_y
                if x0 > x:
                    x0 = x
                elif x1 < x:
                    x1 = x
                if y0 > y:
                    y0 = y
                elif y1 < y:
                    y1 = y

        # check bbox valid:
        if 0 <= x0 < x1 and 0 <= y0 < y1:
            bbox = {
                #  pt1 = (x0, y0)  # obj frame top left corner
                #  pt2 = (x1, y1)  # obj frame bottom right corner
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
            }
        else:
            bbox = None
        return bbox

    @staticmethod
    def add_sub_sub_image_to_detection(
            detections: list,
            cv_img: np.array,
            bbox_image_p: dict,
            ack: bool = False,
            tabs: int = 1
    ) -> list:
        """
        :param detections:
        :param cv_img:
        :param bbox_image_p: dict that specify how much from the bbox to save
                bbox_image_p = {  # all bbox
                    'x_left_start': 0,
                    'x_left_end': 1,
                    'y_top_start': 0,
                    'y_top_end': 1,
                }
                bbox_image_p = {  # sub bbox
                    'x_left_start': 0.2,  x left: start from 20% of the image
                    'x_left_end': 0.9, x right: end at 90%
                    'y_top_start': 0.2, y top: start at 20%
                    'y_top_end': 0.6, y bottom: end at 60%
                }
        :param ack: print sub image dict
        :param tabs:
        :return: for each detection in detections: add entry 'bbox_sub_image' with a dict with keys image,x0,x1,y0,y1
                    if you imshow - you will get the bbox of the detection (according to bbox_image_p)
        """
        if ack:
            print('{}bbox_sub_image of {} detections:'.format(tabs * '\t', len(detections)))
        for pose_i, detection_d in enumerate(detections):
            # label = detection_d['label']
            # confidence = detection_d['score_percentage']
            bbox = detection_d['bbox']
            x0, x1 = bbox['x0'], bbox['x1']
            y0, y1 = bbox['y0'], bbox['y1']

            y_dist = y1 - y0
            x_dist = x1 - x0
            # prepare sub sub image (a part of the bbox)
            # the origin is left - top. (x0,y0) is top left. (x1,y1) is bottom right
            sub_y0 = int(y0 + bbox_image_p['y_top_start'] * y_dist)
            sub_y1 = int(y0 + bbox_image_p['y_top_end'] * y_dist)
            sub_x0 = int(x0 + bbox_image_p['x_left_start'] * x_dist)
            sub_x1 = int(x0 + bbox_image_p['x_left_end'] * x_dist)

            if 0 <= sub_x0 < sub_x1 and 0 <= sub_y0 < sub_y1:
                detection_d['bbox_sub_image'] = {
                    'image': cv_img[sub_y0:sub_y1, sub_x0:sub_x1].tolist(),
                    'x0': sub_x0,
                    'y0': sub_y0,
                    'x1': sub_x1,
                    'y1': sub_y1,
                }
                # cvt.display_open_cv_image(detection_d['bbox_sub_image']['image'])
            else:
                detection_d['bbox_sub_image'] = None

            if ack:
                string = '{}\tbbox_sub_image of pose {}:'.format(tabs * '\t', pose_i)
                bbox_sub = detection_d['bbox_sub_image']
                if bbox_sub is not None:
                    string += ' x0={} y0={} x1={} y1={} '.format(bbox_sub['x0'], bbox_sub['y0'], bbox_sub['x1'],
                                                                 bbox_sub['y1'])
                    string += '{}'.format(mt.to_str(np.array(bbox_sub['image']), title='sub_img'))
                    print(string)
                else:
                    print('{} is None'.format(string))
        return detections
