from enum import Enum
from wizzi_utils.models.labels_bank import *


class ModelType(Enum):
    OdCvNormal = 'Cv2OdModel'
    OdTflNormal = 'TflOdModel'
    PdCvNormal = 'Cv2PdModel'
    PdTflNormal = 'TflPdModel'
    PdTflPoseNet = 'TflPdModelPoseNet'
    PdCvCocoMulti = 'Cv2PdModelCocoMultiPoses'


class Jobs(Enum):
    OBJECT_DETECTION = 'object_detection'
    SEGMENTATION = 'segmentation'  # future
    POSE_DETECTION = 'pose_detection'
    CLASSIFICATION = 'classification'  # future


class DnnFamily(Enum):
    Caffe = 'Caffe'
    Darknet = 'Darknet'
    TF = 'TensorFlow'


class DownloadStyle(Enum):
    Direct = 'Direct'
    Tar = 'tar'
    Zip = 'zip'


MODELS_CONFIG = {
    # object detection TFL and cv2
    # TFL
    'coco_ssd_mobilenet_v1_1_0_quant_2018_06_29': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,
        'tflite': {
            'url': 'http://storage.googleapis.com/download.tensorflow.org/models/tflite/' +
                   'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip',
            'download_style': DownloadStyle.Zip.value,
            'file_to_look': 'detect.tflite',
        },
        'extra': {
            'info': 'https://gist.github.com/iwatake2222/e4c48567b1013cf31de1cea36c4c061c',
            'size': 'tflite(3.99 MB)',
            'fps': '5.10',
        },
    },
    'ssd_mobilenet_v3_small_coco_2020_01_14': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,  # TODO labels are not correct
        'tflite': {
            'url': 'http://download.tensorflow.org/models/object_detection/' +
                   'ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'model.tflite',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v3_small_coco_2020_01_14',
            'size': 'tflite(6.86 MB)',
            'fps': '21.96 FPS',
            'info': 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/'
                    + 'tf1_detection_zoo.md#mobile-models'
        },
    },
    'ssd_mobilenet_v3_large_coco_2020_01_14': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,
        'tflite': {
            'url': 'http://download.tensorflow.org/models/object_detection/' +
                   'ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'model.tflite',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v3_large_coco_2020_01_14',
            'size': 'tflite(12.42 MB)',
            'fps': '12.94 FPS',
            'info': 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/'
                    + 'tf1_detection_zoo.md#mobile-models',
        },
    },
    'ssd_mobilenet_v2_mnasfpn': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,
        'tflite': {
            'url': 'http://download.tensorflow.org/models/object_detection/' +
                   'ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'model.tflite',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v2_mnasfpn',
            'size': 'tflite(9.68 MB)',
            'fps': '5.65 FPS',
            'info': 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/'
                    + 'tf1_detection_zoo.md#mobile-models'
        },
    },
    'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,
        'tflite': {
            'url': 'http://download.tensorflow.org/models/object_detection/' +
                   'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'model.tflite',
        },
        'extra': {
            'web_name': 'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19',
            'size': 'tflite(15.97 MB)',
            'fps': '11.01 FPS',
            'info': 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/'
                    + 'tf1_detection_zoo.md#mobile-models'
        },
    },
    'ssd_mobilenet_v1_1_metadata_1': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,
        'tflite': {
            'url': 'https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite',
            'download_style': DownloadStyle.Direct.value,
        },
        'URL': {
            'tflite': 'https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite',
            'tflite_download_style': DownloadStyle.Direct.value,

        },
        'extra': {
            'web_name': 'TODO',
            'size': 'tflite(3.99 MB)',
            'fps': '4.43 FPS',
            'info': 'https://www.tensorflow.org/lite/examples/object_detection/overview'
        },
    },
    'ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess': {
        'model_type': ModelType.OdTflNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'threshold': 0.2,
        'nms': {
            'score_threshold': 0.4,
            'nms_threshold': 0.4,
        },
        'labels_dict': COCO_80_RAW,
        'tflite': {
            'url': 'https://drive.google.com/' +
                   'uc?export=download&confirm=${CODE}&id=1LjTqn5nChAVKhXgwBUp00XIKXoZrs9sB',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess',
            'size': 'tflite(5.09 MB)',
            'fps': '0.63 FPS',
            'info': 'https://github.com/PINTO0309/PINTO_model_zoo/tree/main/006_mobilenetv2-ssdlite/01_coco/'
                    + '03_integer_quantization'
        },
    },
    # CV
    'yolov4_tiny': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (608, 608),
        'scalefactor': 1 / 127.5,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_YOLO_80_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/yolov4-tiny.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolov4-tiny',
            'size': 'weights(23.13 MB), cfg(2.96 KB)',
            'fps': '11.33',
        },
    },
    'yolov4': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (608, 608),
        'scalefactor': 1 / 127.5,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_YOLO_80_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/yolov4.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolov4',
            'size': 'weights(245.78 MB), cfg(11.94 KB)',
            'fps': '1.44 FPS'
        },
    },
    'yolov3': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (416, 416),
        'scalefactor': 1 / 255,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_YOLO_80_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/yolov3.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://pjreddie.com/media/files/yolov3.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolov3',
            'size': 'weights(236.52 MB), cfg(8.22 KB)',
            'fps': '3.24 FPS',
        },
    },
    'yolov3_ssp': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (608, 608),
        'scalefactor': 1 / 127.5,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_YOLO_80_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-spp.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolov3-spp',
            'size': 'weights(240.53 MB), cfg(8.4 KB)',
            'fps': '1.67 FPS',
            'info': 'TODO',
        },
    },
    'yolo_voc': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (416, 416),
        'scalefactor': 1 / 255,
        'mean': (127.5, 127.5, 127.5),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': PASCAL_VOC_2012_20_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/yolo-voc.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://pjreddie.com/media/files/yolo-voc.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolo-voc.weights',
            'size': 'weights(193.31 MB), cfg(2.66 KB)',
            'fps': '7.48 FPS',
        },
    },
    'MobileNetSSD_deploy': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'in_dims': (800, 600),
        'scalefactor': 1 / 127.5,
        'mean': (127.5, 127.5, 127.5),
        'swapRB': False,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': PASCAL_VOC_2012_21_LABELS,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/' +
                   'daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'MobileNetSSD_deploy',
            'size': 'caffemodel(22.08 MB), prototxt(28.67 KB)',
            'fps': '13.28 FPS',
            'info': 'https://www.programmersought.com/article/569030384/',
        },
    },
    'yolov3_tiny': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (512, 512),
        'scalefactor': 1 / 127.5,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_YOLO_80_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://github.com/ultralytics/yolov3/releases/download/v8/yolov3-tiny.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolov3-tiny.weights',
            'size': 'weights(33.79 MB), cfg(1.87 KB)',
            'fps': '18.56 FPS',
        },
    },
    'yolov2_tiny_voc': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Darknet.value,
        # YOLO FAMILY ALLOWED IN_DIMS: (320, 320), (416, 416), (512, 512), (608, 608)
        'in_dims': (416, 416),
        'scalefactor': 1 / 127.5,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': PASCAL_VOC_2012_20_LABELS,
        'cfg': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/tiny-yolo-voc.cfg',
            'download_style': DownloadStyle.Direct.value,
        },
        'weights': {
            'url': 'https://pjreddie.com/media/files/yolov2-tiny-voc.weights',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'yolov2-tiny-voc.weights',
            'size': 'weights(60.53 MB), cfg(1.38 KB)',
            'fps': '24.61 FPS',
        },
    },
    'VGG_ILSVRC2016_SSD_300x300_deploy': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'in_dims': (300, 300),
        'scalefactor': 1,
        'mean': (127, 127, 127),
        'swapRB': False,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': ILSVRC2016_201_LABELS,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_vgg16.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'https://www.dropbox.com/s/8apyk3uzk2vl522/' +
                   'VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel?dl=1',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel',
            'size': 'caffemodel(192.06 MB), prototxt(23.93 KB)',
            'fps': '2.58',
        },
    },
    'VGG_ILSVRC_16_layers': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (102.9801, 115.9465, 122.7717),
        'swapRB': False,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': PASCAL_VOC_2012_21_LABELS,
        'need_normalize': 'normalize output needed',
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'faster_rcnn_vgg16.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'VGG16_faster_rcnn_final.caffemodel',
        },
        'extra': {
            'web_name': 'VGG16_faster_rcnn_final.caffemodel',
            'size': 'caffemodel(522.92 MB), prototxt(8.45 KB)',
            'fps': '0.21 FPS',
        },
    },
    'rfcn_pascal_voc_resnet50': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (102.9801, 115.9465, 122.7717),
        'swapRB': False,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': PASCAL_VOC_2012_21_LABELS,
        'need_normalize': 'normalize output needed',
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'rfcn_pascal_voc_resnet50.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'https://onedrive.live.com/download?' +
                   'cid=10B28C0E28BF7B83&resid=10B28C0E28BF7B83%215317&authkey=%21AIeljruhoLuail8',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'resnet50_rfcn_final.caffemodel',
        },
        'extra': {
            'web_name': 'resnet50_rfcn_final.caffemodel',
            'size': 'caffemodel(121.58 MB), prototxt(62.8 KB)',
            'fps': '1.28 FPS',
        },
    },
    'faster_rcnn_zf': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (102.9801, 115.9465, 122.7717),
        'swapRB': False,
        'crop': False,
        'threshold': 0.4,
        'nms_threshold': 0.0,
        'labels_dict': PASCAL_VOC_2012_21_LABELS,
        'need_normalize': 'normalize output needed',
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'faster_rcnn_zf.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'https://dl.dropboxusercontent.com/s/o6ii098bu51d139/' +
                   'faster_rcnn_models.tgz?dl=0',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'ZF_faster_rcnn_final.caffemodel',
        },
        'extra': {
            'web_name': 'ZF_faster_rcnn_final',
            'size': 'caffemodel(226.19 MB), prototxt(7.1 KB)',
            'fps': '0.84 FPS',
        },
    },
    'ssd_inception_v2_coco_2017_11_17': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_183_LABELS,
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'ssd_inception_v2_coco_2017_11_17.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'ssd_inception_v2_coco_2017_11_17',
            'size': 'pb(97.26 MB), pbtxt(114.77 KB)',
            'fps': '3.78 FPS',
        },
    },
    'ssd_mobilenet_v1_coco': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (416, 416),
        'scalefactor': 1 / 127.5,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.4,
        'nms_threshold': 0.4,
        'labels_dict': COCO_183_LABELS,
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'ssd_mobilenet_v1_coco.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v1_coco_11_06_2017',
            'size': 'pb(27.76 MB), pbtxt(62.08 KB)',
            'fps': '27.13 FPS',
        },
    },
    'faster_rcnn_inception_v2_coco_2018_01_28': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.4,
        'nms_threshold': 0.4,
        'labels_dict': COCO_182_LABELS,
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/' +
                   'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
            'size': 'pb(54.51 MB), pbtxt(112.92 KB)',
            'fps': '2.51 FPS',
        },
    },
    'faster_rcnn_resnet50_coco_2018_01_28': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_182_LABELS,
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'faster_rcnn_resnet50_coco_2018_01_28.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'faster_rcnn_resnet50_coco_2018_01_28',
            'size': 'pb(114.97 MB), pbtxt(88.76 KB)',
            'fps': '0.84 FPS',
        },
    },
    'ssd_mobilenet_v1_coco_2017_11_17': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (300, 300),
        'scalefactor': 1,
        'mean': (0, 0, 0),
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_183_LABELS,
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'ssd_mobilenet_v1_coco_2017_11_17.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v1_coco_2017_11_17',
            'size': 'pb(27.76 MB), pbtxt(62.08 KB)',
            'fps': '46.28 FPS',
        },
    },
    'ssd_mobilenet_v1_ppn_coco': {  # checked
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (300, 300),
        'scalefactor': 1,
        'mean': (127.5, 127.5, 127.5),
        'swapRB': False,
        'crop': False,
        'threshold': 0.4,
        'nms_threshold': 0.4,
        'labels_dict': COCO_183_LABELS,  # TODO labels could be wrong
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'ssd_mobilenet_v1_ppn_coco.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/' +
                   'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
            'size': 'pb(10.29 MB), pbtxt(67.4 KB)',
            'fps': '38.72 FPS',
        },
    },
    'ssd_mobilenet_v2_coco_2018_03_29': {
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.TF.value,
        'in_dims': (800, 600),
        'scalefactor': 1,
        'mean': (0, 0, 0), 'file_to_look': 'frozen_inference_graph.caffemodel',
        'swapRB': True,
        'crop': False,
        'threshold': 0.2,
        'nms_threshold': 0.4,
        'labels_dict': COCO_183_LABELS,
        'pbtxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'pb': {
            'url': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
            'download_style': DownloadStyle.Tar.value,
            'file_to_look': 'frozen_inference_graph.pb',
        },
        'extra': {
            'web_name': 'ssd_mobilenet_v2_coco_2018_03_29',
            'size': 'pb(66.46 MB), pbtxt(112.89 KB)',
            'fps': '6.57 FPS',
        },
    },
    'opencv_face_detector': {
        'model_type': ModelType.OdCvNormal.value,
        'job': Jobs.OBJECT_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'in_dims': (520, 520),
        'scalefactor': 1,
        'mean': (104, 177, 123),
        'swapRB': False,
        'crop': False,
        'threshold': 0.5,
        'nms_threshold': None,
        'labels_dict': {
            'desc': 'TODO',
            'num_classes': 1,
            'labels': ['None', 'face']
        },
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/' +
                   'opencv_face_detector.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/' +
                   'res10_300x300_ssd_iter_140000.caffemodel',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'res10_300x300_ssd_iter_140000',
            'size': 'caffemodel(10.17 MB), prototxt(27.45 KB)',
            'info1': 'https://www.programmersought.com/article/16544476883/',
            'info2': 'https://www.programmersought.com/article/16544476883/',
            'info3': 'https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/',
        },
    },
    # pose detection TFL and cv2
    # TFL
    'pose_landmark_full': {
        'model_type': ModelType.PdTflNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'threshold': 0.2,
        'joint_names': {
            0: 'nose', 1: 'leftEyeInside', 2: 'leftEye', 3: 'leftEyeOutside', 4: 'rightEyeInside',
            5: 'rightEye', 6: 'rightEyeOutside', 7: 'leftEar', 8: 'rightEar', 9: 'leftMouth',
            10: 'rightMouth', 11: 'leftShoulder', 12: 'rightShoulder', 13: 'leftElbow', 14: 'rightElbow',
            15: 'leftWrist', 16: 'rightWrist', 17: 'leftPalm', 18: 'rightPalm', 19: 'leftIndex',
            20: 'rightIndex', 21: 'leftPinky', 22: 'rightPinky', 23: 'leftHip', 24: 'rightHip',
            25: 'leftKnee', 26: 'rightKnee', 27: 'leftAnkle', 28: 'rightAnkle', 29: 'leftHeel',
            30: 'rightHeel', 31: 'leftFoot', 32: 'rightFoot'
            # don't think those are supported
            # 33: 'midHip', 34: 'forehead',35: 'leftThumb',36: 'leftHand',37: 'rightThumb',38: 'rightHand',
        },
        'joint_colors': [
            'red', 'magenta', 'lime', 'magenta', 'magenta',
            'lime', 'magenta', 'lime', 'lime', 'lime',
            'lime', 'darkorange', 'darkorange', 'yellow', 'yellow',
            'darkorange', 'darkorange', 'aqua', 'aqua', 'lime',
            'lime', 'yellow', 'yellow', 'aqua', 'aqua',
            'magenta', 'magenta', 'aqua', 'aqua', 'magenta',
            'magenta', 'aqua', 'aqua',
        ],
        'pairs_indices': [
            [7, 3], [3, 2], [2, 1], [1, 0], [0, 9], [9, 10], [10, 0], [0, 4], [4, 5], [5, 6], [6, 8],
            [17, 15], [15, 13], [13, 11], [18, 16], [16, 14], [14, 12], [23, 25], [25, 27], [27, 29],
            [29, 31], [24, 26], [26, 28], [28, 30], [30, 32], [23, 24], [24, 12], [12, 11], [11, 23],
        ],
        'pairs_indices_colors': [
            'lime', 'lime', 'lime', 'lime', 'lime',
            'lime', 'lime', 'lime', 'lime', 'lime',
            'lime', 'yellow', 'darkorange', 'yellow', 'yellow',
            'darkorange', 'yellow', 'aqua', 'magenta', 'aqua',
            'magenta', 'aqua', 'magenta', 'aqua', 'magenta',
            'aqua', 'lime', 'darkorange', 'lime'
        ],
        'tflite': {
            'url': 'https://github.com/google/mediapipe/raw/master/mediapipe/modules/pose_landmark/' +
                   'pose_landmark_full.tflite',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_landmark_full',
            'size': 'tflite(6.13 MB)',
            'fps': 'TODO',
            'info': 'https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_detection',
            'info2': 'https://github.com/vladmandic/blazepose/tree/fe647445507e37469d96da6fde5c8b0980f745bc'
        },
    },
    'pose_landmark_lite': {
        'model_type': ModelType.PdTflNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'threshold': 0.2,
        'joint_names': {
            0: 'nose', 1: 'leftEyeInside', 2: 'leftEye', 3: 'leftEyeOutside', 4: 'rightEyeInside',
            5: 'rightEye', 6: 'rightEyeOutside', 7: 'leftEar', 8: 'rightEar', 9: 'leftMouth',
            10: 'rightMouth', 11: 'leftShoulder', 12: 'rightShoulder', 13: 'leftElbow', 14: 'rightElbow',
            15: 'leftWrist', 16: 'rightWrist', 17: 'leftPalm', 18: 'rightPalm', 19: 'leftIndex',
            20: 'rightIndex', 21: 'leftPinky', 22: 'rightPinky', 23: 'leftHip', 24: 'rightHip',
            25: 'leftKnee', 26: 'rightKnee', 27: 'leftAnkle', 28: 'rightAnkle', 29: 'leftHeel',
            30: 'rightHeel', 31: 'leftFoot', 32: 'rightFoot'
            # don't think those are supported
            # 33: 'midHip', 34: 'forehead',35: 'leftThumb',36: 'leftHand',37: 'rightThumb',38: 'rightHand',
        },
        'joint_colors': [
            'red', 'magenta', 'lime', 'magenta', 'magenta',
            'lime', 'magenta', 'lime', 'lime', 'lime',
            'lime', 'darkorange', 'darkorange', 'yellow', 'yellow',
            'darkorange', 'darkorange', 'aqua', 'aqua', 'lime',
            'lime', 'yellow', 'yellow', 'aqua', 'aqua',
            'magenta', 'magenta', 'aqua', 'aqua', 'magenta',
            'magenta', 'aqua', 'aqua',
        ],
        'pairs_indices_colors': [
            'lime', 'lime', 'lime', 'lime', 'lime',
            'lime', 'lime', 'lime', 'lime', 'lime',
            'lime', 'yellow', 'darkorange', 'yellow', 'yellow',
            'darkorange', 'yellow', 'aqua', 'magenta', 'aqua',
            'magenta', 'aqua', 'magenta', 'aqua', 'magenta',
            'aqua', 'lime', 'darkorange', 'lime'
        ],
        'pairs_indices': [
            [7, 3], [3, 2], [2, 1], [1, 0], [0, 9],
            [9, 10], [10, 0], [0, 4], [4, 5], [5, 6],
            [6, 8], [17, 15], [15, 13], [13, 11], [18, 16],
            [16, 14], [14, 12], [23, 25], [25, 27], [27, 29],
            [29, 31], [24, 26], [26, 28], [28, 30], [30, 32],
            [23, 24], [24, 12], [12, 11], [11, 23],
        ],
        'tflite': {
            'url': 'https://github.com/google/mediapipe/raw/master/mediapipe/modules/pose_landmark/' +
                   'pose_landmark_lite.tflite',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'TODO',
            'size': 'tflite(2.68 MB)',
            'fps': 'TODO',
            'info': 'https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark',
            'info2': 'https://github.com/vladmandic/blazepose/tree/fe647445507e37469d96da6fde5c8b0980f745bc'
        },
    },
    'pose_landmark_heavy': {
        'model_type': ModelType.PdTflNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'threshold': 0.2,
        'joint_names': {
            0: 'nose', 1: 'leftEyeInside', 2: 'leftEye', 3: 'leftEyeOutside', 4: 'rightEyeInside',
            5: 'rightEye', 6: 'rightEyeOutside', 7: 'leftEar', 8: 'rightEar', 9: 'leftMouth',
            10: 'rightMouth', 11: 'leftShoulder', 12: 'rightShoulder', 13: 'leftElbow', 14: 'rightElbow',
            15: 'leftWrist', 16: 'rightWrist', 17: 'leftPalm', 18: 'rightPalm', 19: 'leftIndex',
            20: 'rightIndex', 21: 'leftPinky', 22: 'rightPinky', 23: 'leftHip', 24: 'rightHip',
            25: 'leftKnee', 26: 'rightKnee', 27: 'leftAnkle', 28: 'rightAnkle', 29: 'leftHeel',
            30: 'rightHeel', 31: 'leftFoot', 32: 'rightFoot'
            # don't think those are supported
            # 33: 'midHip', 34: 'forehead',35: 'leftThumb',36: 'leftHand',37: 'rightThumb',38: 'rightHand',
        },
        'joint_colors': [
            'red', 'magenta', 'lime', 'magenta', 'magenta',
            'lime', 'magenta', 'lime', 'lime', 'lime',
            'lime', 'darkorange', 'darkorange', 'yellow', 'yellow',
            'darkorange', 'darkorange', 'aqua', 'aqua', 'lime',
            'lime', 'yellow', 'yellow', 'aqua', 'aqua',
            'magenta', 'magenta', 'aqua', 'aqua', 'magenta',
            'magenta', 'aqua', 'aqua',
        ],
        'pairs_indices': [
            [7, 3], [3, 2], [2, 1], [1, 0], [0, 9], [9, 10], [10, 0], [0, 4], [4, 5], [5, 6], [6, 8],
            [17, 15], [15, 13], [13, 11], [18, 16], [16, 14], [14, 12], [23, 25], [25, 27], [27, 29],
            [29, 31], [24, 26], [26, 28], [28, 30], [30, 32], [23, 24], [24, 12], [12, 11], [11, 23],
        ],
        'pairs_indices_colors': [
            'lime', 'lime', 'lime', 'lime', 'lime',
            'lime', 'lime', 'lime', 'lime', 'lime',
            'lime', 'yellow', 'darkorange', 'yellow', 'yellow',
            'darkorange', 'yellow', 'aqua', 'magenta', 'aqua',
            'magenta', 'aqua', 'magenta', 'aqua', 'magenta',
            'aqua', 'lime', 'darkorange', 'lime'
        ],
        'tflite': {
            'url': 'https://github.com/google/mediapipe/raw/master/mediapipe/modules/pose_landmark/' +
                   'pose_landmark_heavy.tflite',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_landmark_heavy',
            'size': 'tflite(26.42 MB)',
            'fps': 'TODO',
            'info': 'https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_detection',
            'info2': 'https://github.com/vladmandic/blazepose/tree/fe647445507e37469d96da6fde5c8b0980f745bc'
        },
    },
    'posenet': {
        'model_type': ModelType.PdTflPoseNet.value,
        'job': Jobs.POSE_DETECTION.value,
        'threshold': 0.2,
        'joint_names': {
            0: 'nose', 1: 'leftEye', 2: 'rightEye', 3: 'leftEar', 4: 'rightEar',
            5: 'leftShoulder', 6: 'rightShoulder', 7: 'leftElbow', 8: 'rightElbow', 9: 'leftWrist',
            10: 'rightWrist', 11: 'leftHip', 12: 'rightHip', 13: 'leftKnee', 14: 'rightKnee',
            15: 'leftAnkle', 16: 'rightAnkle',
        },
        'joint_colors': [
            'red', 'magenta', 'magenta', 'lime', 'lime',
            'darkorange', 'darkorange', 'yellow', 'yellow', 'darkorange',
            'darkorange', 'aqua', 'aqua', 'magenta', 'magenta',
            'aqua', 'aqua'
        ],
        'pairs_indices': [
            [5, 6], [5, 7], [7, 9], [5, 11], [6, 8],
            [8, 10], [6, 12], [11, 12], [11, 13], [13, 15],
            [12, 14], [14, 16]
        ],
        'pairs_indices_colors': [
            'darkorange', 'yellow', 'darkorange', 'lime', 'yellow',
            'darkorange', 'lime', 'aqua', 'aqua', 'magenta',
            'aqua', 'magenta'
        ],
        'tflite': {
            'url': 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/' +
                   'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'posenet',
            'size': 'tflite(12.65 MB)',
            'fps': 'TODO',
            'info': 'https://github.com/ecd1012/rpi_pose_estimation/blob/main/run_pose_estimation.py',
            'info2': 'https://www.tensorflow.org/lite/examples/pose_estimation/overview',
        },
    },
    # cv2
    'openpose_pose_mpi': {
        'model_type': ModelType.PdCvNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'joint_names': {
            0: 'Head', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
            5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'RHip', 9: 'RKnee',
            10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'Chest',
            15: 'Background'
        },
        'joint_colors': [
            'darkorange', 'red', 'darkorange', 'yellow', 'darkorange',
            'darkorange', 'yellow', 'darkorange', 'lime', 'aqua',
            'magenta', 'lime', 'aqua', 'magenta', 'aqua',
            'black'
        ],
        'pairs_indices': [
            [0, 1], [1, 2], [2, 3], [3, 4], [1, 5],
            [5, 6], [6, 7], [1, 14], [14, 8], [8, 9],
            [9, 10], [14, 11], [11, 12], [12, 13]
        ],
        'pairs_indices_colors': [
            'red', 'darkorange', 'yellow', 'darkorange', 'darkorange',
            'yellow', 'darkorange', 'aqua', 'lime', 'aqua',
            'magenta', 'lime', 'aqua', 'magenta'
        ],
        'in_dims': (368, 368),
        'scalefactor': 1 / 255,
        'mean': (0, 0, 0),
        'swapRB': False,
        'crop': False,
        'threshold': 0.1,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/' +
                   'pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_iter_160000',
            'size': 'caffemodel(196.41 MB), prototxt(45.83 KB)',
            'fps': 'TODO',
            'info': 'https://www.programmersought.com/article/3282857837/',
        },
    },
    'openpose_pose_coco': {
        'model_type': ModelType.PdCvNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'joint_names': {
            0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
            5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'RHip', 9: 'RKnee',
            10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye',
            15: 'LEye', 16: 'REar', 17: 'LEar', 18: 'Background',
        },
        'joint_colors': [
            'red', 'red', 'darkorange', 'yellow', 'darkorange',
            'darkorange', 'yellow', 'darkorange', 'lime', 'aqua',
            'magenta', 'lime', 'aqua', 'magenta', 'blue',
            'blue', 'aqua', 'aqua', 'black'
        ],
        'pairs_indices': [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
            [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
            [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17]
        ],
        'pairs_indices_colors': [
            'darkorange', 'darkorange', 'yellow', 'darkorange', 'yellow',
            'darkorange', 'lime', 'aqua', 'magenta', 'lime',
            'aqua', 'magenta', 'red', 'blue', 'aqua',
            'blue', 'aqua'
        ],
        'in_dims': (368, 368),
        'scalefactor': 1 / 255,
        'mean': (0, 0, 0),
        'swapRB': False,
        'crop': False,
        'threshold': 0.1,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/' +
                   'coco/pose_deploy_linevec.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_iter_440000',
            'size': 'caffemodel(199.58 MB), prototxt(45.78 KB)',
            'fps': 'TODO',
            'info1': 'https://github.com/CMU-Perceptual-Computing-Lab/openpose',
            'info2': 'https://www.programmersought.com/article/3282857837/',
        },
    },
    'openpose_pose_coco_multi': {
        'model_type': ModelType.PdCvCocoMulti.value,
        'job': Jobs.POSE_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'joint_names': {
            0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
            5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'RHip', 9: 'RKnee',
            10: 'RAnkle', 11: 'LHip', 12: 'LKnee', 13: 'LAnkle', 14: 'REye',
            15: 'LEye', 16: 'REar', 17: 'LEar'
        },
        'joint_colors': [
            'red', 'red', 'darkorange', 'yellow', 'darkorange',
            'darkorange', 'yellow', 'darkorange', 'lime', 'aqua',
            'magenta', 'lime', 'aqua', 'magenta', 'blue',
            'blue', 'aqua', 'aqua'
        ],
        'pairs_indices': [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
            [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
            [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17]
        ],
        'pairs_indices_colors': [
            'darkorange', 'darkorange', 'yellow', 'darkorange', 'yellow',
            'darkorange', 'lime', 'aqua', 'magenta', 'lime',
            'aqua', 'magenta', 'red', 'blue', 'aqua',
            'blue', 'aqua'
        ],
        'pafs_indices': [
            # index of pafs corresponding to the POSE_PAIRS
            # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output,
            # Similarly, (1,5) -> (39,40) and so on.
            [31, 32], [39, 40], [33, 34], [35, 36], [41, 42],
            [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
            [27, 28], [29, 30], [47, 48], [49, 50], [53, 54],
            [51, 52], [55, 56]
        ],
        'in_dims': (368, 368),
        'scalefactor': 1 / 255,
        'mean': (0, 0, 0),
        'swapRB': False,
        'crop': False,
        'threshold': 0.1,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/' +
                   'coco/pose_deploy_linevec.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_iter_440000',
            'size': 'caffemodel(199.58 MB), prototxt(45.78 KB)',
            'fps': 'TODO',
            'info1': 'https://github.com/CMU-Perceptual-Computing-Lab/openpose',
            'info2': 'https://www.programmersought.com/article/3282857837/',
            'info3': 'https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/' +
                     'multi-person-openpose.py',
        },
    },
    'body25': {
        'model_type': ModelType.PdCvNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'joint_names': {
            0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist',
            5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip',
            10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle',
            15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe',
            20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel',
            25: 'Background',
        },
        'joint_colors': [  # TODO fix on a clearer picture
            'red', 'red', 'darkorange', 'yellow', 'darkorange',
            'darkorange', 'yellow', 'darkorange', 'lime', 'aqua',
            'magenta', 'aqua', 'aqua', 'magenta', 'aqua',
            'lime', 'lime', 'magenta', 'magenta', 'blue',
            'magenta', 'lime', 'aqua', 'magenta', 'blue',
            'blue',
        ],
        'pairs_indices': [
            [1, 0], [1, 2], [1, 5], [2, 3], [3, 4],
            [5, 6], [6, 7], [0, 15], [15, 17], [0, 16],
            [16, 18], [1, 8], [8, 9], [9, 10], [10, 11],
            [11, 22], [22, 23], [11, 24], [8, 12], [12, 13],
            [13, 14], [14, 19], [19, 20], [14, 21]
        ],
        'pairs_indices_colors': [
            'red', 'darkorange', 'darkorange', 'yellow', 'darkorange',
            'yellow', 'darkorange', 'lime', 'lime', 'lime',
            'lime', 'lime', 'lime', 'aqua', 'magenta',
            'aqua', 'lime', 'darkorange', 'lime', 'aqua',
            'magenta', 'aqua', 'lime', 'darkorange'
        ],
        'in_dims': (368, 368),
        'scalefactor': 1 / 255,
        'mean': (0, 0, 0),
        'swapRB': False,
        'crop': False,
        'threshold': 0.1,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/' +
                   'openpose/master/models/pose/body_25/pose_deploy.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/' +
                   'pose_iter_584000.caffemodel',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_iter_584000.caffemodel',
            'size': 'caffemodel(99.86 MB), prototxt(41.34 KB)',
            'fps': 'TODO',
            'info1': 'https://github.com/CMU-Perceptual-Computing-Lab/openpose',
            'info2': 'https://www.programmersought.com/article/3282857837/',
            'info3': 'https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html',
        },
    },
    'hand_pose': {
        'model_type': ModelType.PdCvNormal.value,
        'job': Jobs.POSE_DETECTION.value,
        'family': DnnFamily.Caffe.value,
        'joint_names': {
            0: 'handRoot',
            1: 'thumbBase', 2: 'thumbMid1', 3: 'thumbMid2', 4: 'thumbNail',
            5: 'indexBase', 6: 'indexMid1', 7: 'indexMid2', 8: 'indexNail',
            9: 'middleBase', 10: 'middleMid1', 11: 'middleMid2', 12: 'middleNail',
            13: 'ringBase', 14: 'ringMid1', 15: 'ringMid2', 16: 'ringNail',
            17: 'pinkyBase', 18: 'pinkyMid1', 19: 'pinkyMid2', 20: 'pinkyNail',
        },
        'joint_colors': [
            'red',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
        ],
        'pairs_indices': [
            [0, 1], [1, 2], [2, 3], [3, 4],  # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  # index
            [0, 9], [9, 10], [10, 11], [11, 12],  # middle
            [0, 13], [13, 14], [14, 15], [15, 16],  # ring
            [0, 17], [17, 18], [18, 19], [19, 20]  # pinky
        ],
        'pairs_indices_colors': [
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow',
            'darkorange', 'aqua', 'lime', 'yellow'
        ],
        'in_dims': (368, 368),
        'scalefactor': 1 / 255,
        'mean': (0, 0, 0),
        'swapRB': False,
        'crop': False,
        'threshold': 0.1,
        'prototxt': {
            'url': 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/' +
                   'hand/pose_deploy.prototxt',
            'download_style': DownloadStyle.Direct.value,
        },
        'caffemodel': {
            'url': 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel',
            'download_style': DownloadStyle.Direct.value,
        },
        'extra': {
            'web_name': 'pose_iter_102000.caffemodel',
            'size': 'caffemodel(140.52 MB), prototxt(25.83 KB)',
            'fps': 'TODO',
            'info1': 'https://www.programmersought.com/article/1536716931/',
        },
    },
}
