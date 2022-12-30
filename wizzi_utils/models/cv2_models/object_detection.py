import numpy as np
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.open_cv import open_cv_tools as cvt
from wizzi_utils.models import models_configs as cfg
from wizzi_utils.models.base_models import OdBaseModel
# noinspection PyPackageRequirements
import cv2
# import cv3


class Cv2OdModel(OdBaseModel):
    def __init__(self,
                 save_load_dir: str,
                 model_name: str,
                 allowed_class: list = None,
                 threshold: float = None,
                 nms_threshold: float = None,
                 in_dims: tuple = None,
                 scalefactor: float = None,
                 mean: tuple = None,
                 swapRB: bool = None,
                 crop: bool = None,
                 check_type: bool = True,
                 device: str = 'cpu'
                 ):
        """
        :param save_load_dir: where the model is saved (or will be if not exists)
        :param model_name: valid name in MODEL_CONF.keys()
        :param allowed_class: ignore rest of class. list of strings
            if None - all classes allowed
        for all the following - if is None: take default value from MODELS_DNN_OBJECT_DETECTION_META_DATA['model_name']
        :param threshold: only detection above this threshold will be returned
        :param nms_threshold: non maximum suppression threshold
        :param in_dims:
        :param scalefactor:
        :param mean:
        :param swapRB:
        :param crop:
        :param check_type:
        :param device:
        example:
        model = od.Cv2OdModel(
            save_load_dir=m_save_dir,
            model_name=model_name,
            allowed_class=['dog', 'cat'],
            threshold=0.1,
            nms_threshold=0.3,
            in_dims=(416, 416),
            scalefactor=1 / 127.5,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        see:
        best_model_images_test()
        best_model_video_test()
        best_model_cam_test()
        models_compare_images_test()
        models_compare_video_test()
        models_compare_cam_test()
        """
        super().__init__(save_load_dir=save_load_dir, model_name=model_name, allowed_class=allowed_class)
        if check_type and not self.model_type_valid(self.model_cfg['model_type'], cfg.ModelType.OdCvNormal.value):
            exit(-1)
        net = None
        if self.model_cfg['family'] == cfg.DnnFamily.Caffe.value:
            model_prototxt = "{}/{}.prototxt".format(self.local_path, self.model_name)
            model_caffe = "{}/{}.caffemodel".format(self.local_path, self.model_name)
            self._download_if_needed(local_path=model_prototxt, url_dict=self.model_cfg['prototxt'])
            self._download_if_needed(local_path=model_caffe, url_dict=self.model_cfg['caffemodel'])
            self.model_resources_sizes = [mt.file_or_folder_size(model_prototxt), mt.file_or_folder_size(model_caffe)]
            net = cv2.dnn.readNetFromCaffe(prototxt=model_prototxt, caffeModel=model_caffe)

        elif self.model_cfg['family'] == cfg.DnnFamily.Darknet.value:
            model_cfg = "{}/{}.cfg".format(self.local_path, self.model_name)
            model_weights = "{}/{}.weights".format(self.local_path, self.model_name)
            self._download_if_needed(local_path=model_cfg, url_dict=self.model_cfg['cfg'])
            self._download_if_needed(local_path=model_weights, url_dict=self.model_cfg['weights'])
            self.model_resources_sizes = [mt.file_or_folder_size(model_cfg), mt.file_or_folder_size(model_weights)]
            net = cv2.dnn.readNetFromDarknet(cfgFile=model_cfg, darknetModel=model_weights)

        elif self.model_cfg['family'] == cfg.DnnFamily.TF.value:
            model_pbtxt = "{}/{}.pbtxt".format(self.local_path, self.model_name)
            model_pb = "{}/{}.pb".format(self.local_path, self.model_name)
            self._download_if_needed(local_path=model_pbtxt, url_dict=self.model_cfg['pbtxt'])
            self._download_if_needed(local_path=model_pb, url_dict=self.model_cfg['pb'])
            self.model_resources_sizes = [mt.file_or_folder_size(model_pbtxt), mt.file_or_folder_size(model_pb)]
            net = cv2.dnn.readNetFromTensorflow(model=model_pb, config=model_pbtxt)

        if net is None:
            mt.exception_error('Failed to create network', real_exception=False)
            exit(-1)

        self.device = self.set_device(net, device)

        self.model = cv2.dnn_DetectionModel(net)

        if threshold is not None:
            self.model_cfg['threshold'] = threshold
        if nms_threshold is not None:
            self.model_cfg['nms_threshold'] = nms_threshold
        if in_dims is not None:
            self.model_cfg['in_dims'] = in_dims
        if scalefactor is not None:
            self.model_cfg['scalefactor'] = scalefactor
        if mean is not None:
            self.model_cfg['mean'] = mean
        if swapRB is not None:
            self.model_cfg['swapRB'] = swapRB
        if crop is not None:
            self.model_cfg['crop'] = crop

        # self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
        self.model.setInputParams(
            scale=self.model_cfg['scalefactor'],
            size=self.model_cfg['in_dims'],
            mean=self.model_cfg['mean'],
            swapRB=self.model_cfg['swapRB'],
            crop=self.model_cfg['crop']
        )

        return

    def to_string(self, tabs: int = 1) -> str:
        tabs_s = tabs * '\t'
        string = '{}{}'.format(tabs_s, mt.add_color(string='Cv2OdModel:', ops='underlined'))
        string += '\n\t{}name= {} (resources size: {})'.format(tabs_s, self.model_name, self.model_resources_sizes)
        string += '\n\t{}Running on {}'.format(tabs_s, self.device)
        string += '\n\t{}local_path={}'.format(tabs_s, self.local_path)
        string += '\n\t{}{}'.format(tabs_s, mt.to_str(self.allowed_classes, 'allowed_class'))
        string += '\n{}'.format(mt.dict_as_table(self.model_cfg, title='conf', fp=6, ack=False, tabs=tabs + 1))
        return string

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
        classes, scores, boxes = self.model.detect(cv_img, self.model_cfg['threshold'], self.model_cfg['nms_threshold'])
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(classes, 'classes')))
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(scores, 'scores')))
        # print('{}\t{}'.format(tabs * '\t', mt.to_str(boxes, 'boxes')))
        if len(classes) > 0 and len(scores) > 0:
            classes = classes.flatten()
            scores = scores.flatten()

        detections = self.extract_results(
            classes=classes,
            scores=scores,
            boxes=boxes,
            cv_img=cv_img,
            fp=fp,
            ack=ack,
            tabs=tabs,
            img_title=img_title
        )

        return detections

    def extract_results(
            self,
            classes: np.array,
            scores: np.array,
            boxes: np.array,
            cv_img: np.array,
            fp: int = 2,
            ack: bool = False,
            tabs: int = 1,
            img_title: str = None
    ) -> list:
        """
        :param classes: result of classes, scores, boxes = self.model.detect(cv_img, self.threshold, self.nms_threshold)
        :param scores: result of classes, scores, boxes = self.model.detect(cv_img, self.threshold, self.nms_threshold)
        :param boxes: result of classes, scores, boxes = self.model.detect(cv_img, self.threshold, self.nms_threshold)
        :param cv_img: cv image
        :param fp: float precision on the score precentage. e.g. fp=2: 0.1231231352353 -> 12.31%
        :param ack: if True: print meta data and detections
        :param tabs: if ack True: tabs for print
        :param img_title: if ack True: title for print
        :return: list of dicts:
        dict is a detection of an object above threshold.
            has items:
                label:str e.g. 'person'
                score_percentage: float e.g. 12.31
                bbox: dict with keys x0,y0,x1,y1
                #  pt1 = (x0, y0)  # obj frame top left corner
                #  pt2 = (x1, y1)  # obj frame bottom right corner
        """

        if ack:
            title_suffix = '' if img_title is None else '{} '.format(img_title)
            title = '{} detections({}) on image {}{}:'.format(self.model_name, len(classes), title_suffix, cv_img.shape)

            print('{}{}'.format(tabs * '\t', title))
            print('{}Meta_data:'.format(tabs * '\t'))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(classes, 'classes')))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(scores, 'scores')))
            print('{}\t{}'.format(tabs * '\t', mt.to_str(boxes, 'boxes')))
            print('{}Detections:'.format(tabs * '\t'))

        img_h, img_w = cv_img.shape[0], cv_img.shape[1]
        detections = []
        for i, (class_id, score_frac, bbox) in enumerate(zip(classes, scores, boxes)):
            label = self.labels[class_id]
            if label in self.allowed_classes:
                score_percentage = round(score_frac * 100, fp)

                (x0, y0) = (bbox[0], bbox[1])
                (w, h) = (bbox[2], bbox[3])

                x1 = min(x0 + w, img_w)  # dont exceed frame width
                y1 = min(y0 + h, img_h)  # dont exceed frame height

                if 'need_normalize' in self.model_cfg:
                    in_dims_w, in_dims_h = self.model_cfg['in_dims']
                    x0 = int(x0 * (img_w / in_dims_w))
                    x1 = int(x1 * (img_w / in_dims_w))
                    y0 = int(y0 * (img_h / in_dims_h))
                    y1 = int(y1 * (img_h / in_dims_h))

                detection_d = {
                    'label': label,
                    'score_percentage': score_percentage,
                    'bbox': {
                        #  pt1 = (x0, y0)  # obj frame top left corner
                        #  pt2 = (x1, y1)  # obj frame bottom right corner
                        'x0': int(x0),
                        'y0': int(y0),
                        'x1': int(x1),
                        'y1': int(y1),
                    },
                }

                detections.append(detection_d)
                if ack:
                    d_msg = '{}\t{})Detected {}({}%) in top left=({}), bottom right=({})'
                    print(d_msg.format(tabs * '\t', i, label, score_percentage, (x0, y0), (x1, y1)))

        return detections

    @staticmethod
    def set_device(net: cv2.dnn, device: str = 'cpu') -> str:
        string = 'CPU'
        if device == "gpu":
            if cvt.cuda_on():
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                string = 'GPU'
            else:
                mt.exception_error('GPU requested but open_cv can\'t find cuda device')
                net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        else:  # device == "cpu":
            net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        return string
