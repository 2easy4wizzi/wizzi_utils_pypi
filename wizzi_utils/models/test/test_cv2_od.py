from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt
from wizzi_utils.open_cv.test import test_open_cv_tools as cvtt
from wizzi_utils.models.cv2_models import object_detection as od
from wizzi_utils.models.test.shared_code_for_tests import od_or_pd_Model_image_test
from wizzi_utils.models.test.shared_code_for_tests import od_or_pd_Model_video_test
from wizzi_utils.models.test.shared_code_for_tests import od_or_pd_Model_cam_test
from wizzi_utils.models.test.shared_code_for_tests import CAM_FRAMES, IMAGE_MS, VID_X_FRAMES_CV
from wizzi_utils.models.test.shared_code_for_tests import OD_OP, OD_IM_TEST, OD_CAM_TEST, OD_VID_TEST


def __get_od_dict(op: int = 1):
    """
    cv2 od models:
    [
        'yolov4_tiny',
        'yolov4',
        'yolov3',
        'yolov3_ssp',
        'yolo_voc',
        'MobileNetSSD_deploy',
        'yolov3_tiny',
        'yolov2_tiny_voc',
        'VGG_ILSVRC2016_SSD_300x300_deploy',
        'VGG_ILSVRC_16_layers',
        'rfcn_pascal_voc_resnet50',
        'faster_rcnn_zf',
        'ssd_inception_v2_coco_2017_11_17',
        'ssd_mobilenet_v1_coco',
        'faster_rcnn_inception_v2_coco_2018_01_28',
        'faster_rcnn_resnet50_coco_2018_01_28',
        'ssd_mobilenet_v1_coco_2017_11_17',
        'ssd_mobilenet_v1_ppn_coco',
        'ssd_mobilenet_v2_coco_2018_03_29',
        'opencv_face_detector'
    ]
    :param op:
    :return:
    """
    if op == 1:
        od_solo = {
            'model_names': ['yolov4'],
            'dis_size': (640, 480),
            'grid': (1, 1),
        }
        od_meta_dict = od_solo
    elif op == 2:
        od_dual = {
            'model_names': [
                'yolov3',
                'yolov4'
            ],
            'dis_size': (640 * 2, 480),
            'grid': (1, 2),
        }
        od_meta_dict = od_dual
    elif op == 3:
        od_all_models = {
            'model_names': od.OdBaseModel.get_object_detection_models(m_type='Cv2', ack=False),
            'dis_size': 'fs',
            'grid': (4, 5),
        }
        od_meta_dict = od_all_models
    else:
        od_meta_dict = None
    return od_meta_dict


def _get_models(model_names: list) -> list:
    models = []
    for model_name in model_names:
        m_save_dir = '{}/object_detection/{}'.format(mtt.MODELS, model_name)
        model = od.Cv2OdModel(
            save_load_dir=m_save_dir,
            model_name=model_name,
            device='gpu',
        )
        print(model.to_string(tabs=1))
        models.append(model)
    return models


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    od_meta_dict = __get_od_dict(op=OD_OP)
    models = _get_models(od_meta_dict['model_names'])
    if OD_IM_TEST:
        cv_img = cvtt.load_img_from_web(mtt.DOG, ack=False)
        od_or_pd_Model_image_test(
            cv_img=cv_img,
            models=models,
            ms=IMAGE_MS,
            dis_size=od_meta_dict['dis_size'],
            grid=od_meta_dict['grid'],
        )
    if OD_CAM_TEST:
        od_or_pd_Model_cam_test(
            models=models,
            max_frames=CAM_FRAMES,
            dis_size=od_meta_dict['dis_size'],
            grid=od_meta_dict['grid'],
        )
    if OD_VID_TEST:
        # vid_path = cvtt.get_vid_from_web(name=mtt.DOG1)
        vid_path = cvtt.get_vid_from_web(name=mtt.WOMAN_YOGA)
        od_or_pd_Model_video_test(
            models=models,
            vid_path=vid_path,
            work_every_x_frames=VID_X_FRAMES_CV,
            dis_size=od_meta_dict['dis_size'],
            grid=od_meta_dict['grid'],
        )
    print('{}'.format('-' * 20))
    return
