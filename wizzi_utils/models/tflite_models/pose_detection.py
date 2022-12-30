from tflite_runtime.interpreter import Interpreter
import numpy as np
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.models import models_configs as cfg
from wizzi_utils.models.base_models import PdBaseModel
# noinspection PyPackageRequirements
import cv2


class TflPdModel(PdBaseModel):
    def __init__(
            self,
            save_load_dir: str,
            model_name: str,
            threshold: float = None,
            allowed_joint_names: list = None,
            min_p_valid_joints: float = 0.3,
            check_type: bool = True
    ):
        """
        :param save_load_dir: where the model is saved (or will be if not exists)
        :param model_name: valid name in MODEL_CONF.keys()
        :param threshold: only detection above this threshold will be pass first filter
        :param allowed_joint_names: joint_names to track from the joint_names of the model
        :param min_p_valid_joints:
            let x=min_p_valid_joints*len(allowed_joint_names)
            to keep a pose: we need max(x,2) joints found
        :param check_type:
        example:
        see:
        """
        super().__init__(save_load_dir=save_load_dir, model_name=model_name, allowed_joint_names=allowed_joint_names,
                         min_p_valid_joints=min_p_valid_joints)
        if check_type and not self.model_type_valid(self.model_cfg['model_type'], cfg.ModelType.PdTflNormal.value):
            exit(-1)
        if threshold is not None:
            self.model_cfg['threshold'] = threshold
        self.pairs_indices = self.model_cfg['pairs_indices']

        model_fp = "{}/{}.tflite".format(self.local_path, self.model_name)
        self._download_if_needed(local_path=model_fp, url_dict=self.model_cfg['tflite'])
        self.model_size = mt.file_or_folder_size(model_fp)
        # print('Loading {}(size {}, {} classes)'.format(self.model_name, self.model_size, len(self.labels)))
        self.interpreter = Interpreter(model_path=model_fp, num_threads=4)
        # allocate input output placeholders
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, input_height, input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.model_cfg['in_dims'] = (input_width, input_height)
        self.model_cfg['input_type'] = self.interpreter.get_input_details()[0]['dtype']
        quantization = self.interpreter.get_input_details()[0]['quantization']
        # need to normalize if model doesn't do quantization (then quantization == (0.0, 0))
        # else e.g. quantization == (0.0078125, 128) - no need to normalize
        self.model_cfg['normalize_RGB'] = True if quantization == (0.0, 0) else False

        # # you can print this to get more details on the model
        # mt.dict_as_table(self.interpreter.get_input_details()[0], title='input')
        # mt.dict_as_table(self.interpreter.get_output_details()[0], title='output')
        # mt.dict_as_table(self.interpreter.get_tensor_details()[0], title='tensor')
        return

    def to_string(self, tabs: int = 1) -> str:
        tabs_s = tabs * '\t'
        string = '{}{}'.format(tabs_s, mt.add_color(string='TflPdModel:', ops='underlined'))
        string += '\n\t{}name= {} (size {})'.format(tabs_s, self.model_name, self.model_size)
        string += '\n\t{}local_path= {}'.format(tabs_s, self.local_path)
        string += '\n\t{}{}'.format(tabs_s, mt.to_str(self.allowed_joint_names, 'allowed_joint_names'))
        string += '\n{}'.format(mt.dict_as_table(self.model_cfg, title='cfg', fp=6, ack=False, tabs=tabs + 1))
        return string

    def prepare_input(self, cv_img: np.array) -> np.array:
        """
        :param cv_img:
        resize and change dtype to predefined params
        :return:
        """
        img_RGB = cvt.bgr_to_rgb(cv_img)

        if self.model_cfg['normalize_RGB']:
            center = 127.5
            img_RGB = (img_RGB / center) - 1  # normalize image
        img = cv2.resize(img_RGB, self.model_cfg['in_dims'])  # size of this model input
        img_processed = np.expand_dims(img, axis=0).astype(self.model_cfg['input_type'])  # a,b,c -> 1,a,b,c
        return img_processed

    def run_network(self, img_preprocessed: np.array) -> None:
        self.interpreter.set_tensor(self.input_details[0]['index'], img_preprocessed)  # set input tensor
        self.interpreter.invoke()  # run
        return

    def extract_results(
            self,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param cv_img: cv image
        :param fp: float precision on the score percentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts:
        dict is a detection of pose
            each has items:
                'joint_names_list':  - all joint names like in the config
                'joint_ids_list': - all joint ids like in the config
                'joint_x_y_list': - if joint found: it's x,y values
                'joint_z_list': - if joint found: it's z value
                'score_percentages_list': - if joint found: it's confidence in 0-100%,
        """
        # get results
        depth = 5  # each points has x,y,z,visibility,presence
        # full pose -> 195/5=39. but i think there are only 33
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # points numpy

        if ack:
            title_suffix = '' if img_title is None else '{} '.format(img_title)
            title = '{} detection on image {}{}:'.format(self.model_name, title_suffix, cv_img.shape)
            print('{}{}'.format(tabs * '\t', title))
            print('{}Meta_data:'.format(tabs * '\t'))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(outputs, 'outputs')))
            print('{}Detections:'.format(tabs * '\t'))

        detections = []  # each detection is a set of joint
        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        net_w, net_h = self.model_cfg['in_dims']
        for pose_id in range(1):  # currently supports 1 pose
            detection_d = {
                'joint_names_list': [],
                'joint_ids_list': [],
                'joint_x_y_list': [],
                'joint_z_list': [],
                'score_percentages_list': [],
                'bbox': {}
            }
            valid_joints_found = 0  # max is len(self.allowed_joint_names)
            for joint_id, joint_name in self.joint_names.items():
                detection_d['joint_names_list'].append(joint_name)
                detection_d['joint_ids_list'].append(joint_id)
                x = int(outputs[joint_id * depth + 0] * img_w / net_w)
                y = int(outputs[joint_id * depth + 1] * img_h / net_h)
                z = int(outputs[joint_id * depth + 2])

                visibility = outputs[joint_id * depth + 3]
                visibility = 1 / (1 + np.exp(visibility))  # reverse sigmoid
                presence = outputs[joint_id * depth + 4]
                presence = 1 / (1 + np.exp(presence))  # reverse sigmoid
                score_frac = 1 - max(visibility, presence)  # change from err to acc: acc = 1-err
                score_percentage = round(score_frac * 100, fp)

                if self.model_cfg['threshold'] <= score_frac <= 1.0 and joint_name in self.allowed_joint_names:
                    detection_d['joint_x_y_list'].append([x, y])
                    detection_d['joint_z_list'].append(z)
                    detection_d['score_percentages_list'].append(score_percentage)
                    valid_joints_found += 1
                else:
                    detection_d['joint_x_y_list'].append(self.NOT_FOUND_PAIR)
                    detection_d['joint_z_list'].append(self.NOT_FOUND_VALUE)
                    detection_d['score_percentages_list'].append(self.NOT_FOUND_VALUE)

                if ack:
                    d_msg = '{}\tpose {}: {}({}): xy={}, z={}, score=({}%)'
                    print(d_msg.format(tabs * '\t', pose_id, joint_name, joint_id, detection_d['joint_x_y_list'][-1],
                                       detection_d['joint_z_list'][-1], detection_d['score_percentages_list'][-1]))
            min_joints_found_ok = valid_joints_found >= self.min_p_valid_joints * len(self.allowed_joint_names)
            pose_valid = min_joints_found_ok and valid_joints_found >= 2
            if pose_valid:  # minimum 2 regardless to self.min_p_valid_joints(which could be zero)
                detection_d['bbox'] = PdBaseModel.get_bbox_dict(detection_d['joint_x_y_list'])
                if detection_d['bbox'] is not None:
                    if ack:
                        print('{}\tbbox:{}'.format(tabs * '\t', detection_d['bbox']))
                    detections.append(detection_d)
                else:
                    if ack:
                        print('{}\tPose {} was dumped. bbox is None'.format(tabs * '\t', pose_id))
            else:
                if ack:
                    min_needed = max(self.min_p_valid_joints * len(self.allowed_joint_names), 2)
                    msg = '{}\tPose {} was dumped. It had valid_joints_found={} (min needed {})'
                    print(msg.format(tabs * '\t', pose_id, valid_joints_found, min_needed))
        return detections

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
        :param fp: float precision on the score precentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts. see extract_results()
        """

        img_preprocessed = self.prepare_input(cv_img)
        self.run_network(img_preprocessed)
        detections = self.extract_results(
            cv_img=cv_img,
            fp=fp,
            ack=ack,
            tabs=tabs,
            img_title=img_title
        )
        return detections


class TflPdModelPoseNet(TflPdModel):
    def __init__(
            self,
            save_load_dir: str,
            model_name: str,
            threshold: float = None,
            allowed_joint_names: list = None,
            min_p_valid_joints: float = 0.3,
            check_type: bool = True
    ):
        super().__init__(save_load_dir=save_load_dir, model_name=model_name, threshold=threshold,
                         allowed_joint_names=allowed_joint_names, min_p_valid_joints=min_p_valid_joints,
                         check_type=False)
        if check_type and not self.model_type_valid(self.model_cfg['model_type'], cfg.ModelType.PdTflPoseNet.value):
            exit(-1)
        return

    @staticmethod
    def mod(a: np.array, b: int) -> np.array:
        """ find a % b """
        floored = np.floor_divide(a, b)
        return np.subtract(a, np.multiply(floored, b))

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        """ apply sigmoid activation to numpy array """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_and_argmax2d(heatmap: np.array, threshold: float) -> np.array:
        """
        :param heatmap: 9x9x17 (17 joints)
        :param threshold:
        :return: y,x coordinates 17x2
        """
        # v1 is 9x9x17 heatmap
        # v1 = interpreter.get_tensor(output_details[0]['index'])[0]
        height, width, depth = heatmap.shape
        reshaped = np.reshape(heatmap, [height * width, depth])
        reshaped = TflPdModelPoseNet.sigmoid(reshaped)
        # apply threshold
        reshaped = (reshaped > threshold) * reshaped
        coords = np.argmax(reshaped, axis=0)
        yCoords = np.round(np.expand_dims(np.divide(coords, width), 1))
        xCoords = np.expand_dims(TflPdModelPoseNet.mod(coords, width), 1)
        ret = np.concatenate([yCoords, xCoords], 1)
        return ret

    @staticmethod
    def get_offsets(offsets: np.array, coords: np.array, num_key_points: int = 17) -> np.array:
        """
        :param offsets: 9x9x34 - probably yx heatmap per joint(17*2)
        :param coords: 17x2
        :param num_key_points: number of joints
        :return: get offset vectors from all coordinates
        """
        # offsets = interpreter.get_tensor(output_details[1]['index'])[0]
        offset_vectors = []
        for i, (heatmap_y, heatmap_x) in enumerate(coords):
            # print(i, y, x)
            heatmap_y = int(heatmap_y)
            heatmap_x = int(heatmap_x)
            # print(heatmap_y, heatmap_x)
            # make sure indices aren't out of range
            heatmap_y = min(8, heatmap_y)
            heatmap_x = min(8, heatmap_x)
            y_off = offsets[heatmap_y, heatmap_x, i]
            x_off = offsets[heatmap_y, heatmap_x, i + num_key_points]
            # ov = get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)
            offset_vectors.append([y_off, x_off])
        offset_vectors = np.array(offset_vectors)
        return offset_vectors

    def extract_results(
            self,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param cv_img: cv image
        :param fp: float precision on the score percentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts:
        dict is a detection of pose
            each has items:
                'joint_names_list':  - all joint names like in the config
                'joint_ids_list': - all joint ids like in the config
                'joint_x_y_list': - if joint found: it's x,y values
                'joint_z_list': - if joint found: it's z value
                'score_percentages_list': - if joint found: it's confidence in 0-100%,
        """
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        offsets = self.interpreter.get_tensor(self.output_details[1]['index'])[0]

        if ack:
            title_suffix = '' if img_title is None else '{} '.format(img_title)
            title = '{} detection on image {}{}:'.format(self.model_name, title_suffix, cv_img.shape)
            print('{}{}'.format(tabs * '\t', title))
            print('{}Meta_data:'.format(tabs * '\t'))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(outputs, 'outputs')))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(offsets, 'offsets')))
            print('{}Detections:'.format(tabs * '\t'))

        # get y,x positions from heat map
        yx = TflPdModelPoseNet.sigmoid_and_argmax2d(outputs, threshold=self.model_cfg['threshold'])
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(yx, 'yx')))

        # points below threshold (value is [0, 0])
        drop_pts = list(np.unique(np.where(yx == 0)[0]))
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(drop_pts, 'drop_pts')))

        # get offsets from positions
        offset_vectors = TflPdModelPoseNet.get_offsets(offsets, yx)
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(offset_vectors, 'offset_vectors')))

        # use stride to get coordinates in image coordinates
        output_stride = 32
        yx_values = yx * output_stride + offset_vectors
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(yx_values, 'yx_values')))

        for bad_joint_ind in drop_pts:
            yx_values[bad_joint_ind] = self.NOT_FOUND_PAIR
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(yx_values, 'yx_values')))

        detections = []  # each detection is a set of joint
        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        net_w, net_h = self.model_cfg['in_dims']
        for pose_id in range(1):  # currently supports 1 pose
            detection_d = {
                'joint_names_list': [],
                'joint_ids_list': [],
                'joint_x_y_list': [],
                'score_percentages_list': [],
                'bbox': {}
            }
            valid_joints_found = 0  # max is len(self.allowed_joint_names)
            for joint_id, joint_name in self.joint_names.items():
                detection_d['joint_names_list'].append(joint_name)
                detection_d['joint_ids_list'].append(joint_id)

                if yx_values[joint_id].tolist() == self.NOT_FOUND_PAIR:
                    score_frac = 0
                    x, y, score_percentage = None, None, None
                else:
                    y = int(yx_values[joint_id][0] * img_h / net_h)
                    x = int(yx_values[joint_id][1] * img_w / net_w)
                    score_frac = 0.99  # for now till i will extract the confidence from the detection
                    # score_percentage = round(score_frac * 100, fp)
                    score_percentage = 'TODO ???'

                if self.model_cfg['threshold'] <= score_frac <= 1.0 and joint_name in self.allowed_joint_names:
                    detection_d['joint_x_y_list'].append([x, y])
                    detection_d['score_percentages_list'].append(score_percentage)
                    valid_joints_found += 1
                else:
                    detection_d['joint_x_y_list'].append(self.NOT_FOUND_PAIR)
                    detection_d['score_percentages_list'].append(self.NOT_FOUND_VALUE)

                if ack:
                    d_msg = '{}\tpose {}: {}({}): xy={}, score=({}%)'
                    print(d_msg.format(tabs * '\t', pose_id, joint_name, joint_id, detection_d['joint_x_y_list'][-1],
                                       detection_d['score_percentages_list'][-1]))
            min_joints_found_ok = valid_joints_found >= self.min_p_valid_joints * len(self.allowed_joint_names)
            pose_valid = min_joints_found_ok and valid_joints_found >= 2
            if pose_valid:  # minimum 2 regardless to self.min_p_valid_joints(which could be zero)
                detection_d['bbox'] = PdBaseModel.get_bbox_dict(detection_d['joint_x_y_list'])
                if detection_d['bbox'] is not None:
                    if ack:
                        print('{}\tbbox:{}'.format(tabs * '\t', detection_d['bbox']))
                    detections.append(detection_d)
                else:
                    if ack:
                        print('{}\tPose {} was dumped. bbox is None'.format(tabs * '\t', pose_id))
            else:
                if ack:
                    min_needed = max(self.min_p_valid_joints * len(self.allowed_joint_names), 2)
                    msg = '{}\tPose {} was dumped. It had valid_joints_found={} (min needed {})'
                    print(msg.format(tabs * '\t', pose_id, valid_joints_found, min_needed))
        return detections
