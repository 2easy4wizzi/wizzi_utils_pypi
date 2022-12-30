from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt
from wizzi_utils.open_cv.test import test_open_cv_tools as cvtt
from wizzi_utils.models.tflite_models import object_detection as od
from wizzi_utils.models.test.shared_code_for_tests import od_or_pd_Model_image_test
from wizzi_utils.models.test.shared_code_for_tests import od_or_pd_Model_video_test
from wizzi_utils.models.test.shared_code_for_tests import od_or_pd_Model_cam_test
from wizzi_utils.models.test.shared_code_for_tests import CAM_FRAMES, IMAGE_MS, VID_X_FRAMES_TFL
from wizzi_utils.models.test.shared_code_for_tests import OD_OP, OD_IM_TEST, OD_CAM_TEST, OD_VID_TEST


def __get_od_dict(op: int = 1):
    """
    Tfl od models
    [
        'coco_ssd_mobilenet_v1_1_0_quant_2018_06_29',
        'ssd_mobilenet_v3_small_coco_2020_01_14',
        'ssd_mobilenet_v3_large_coco_2020_01_14',
        'ssd_mobilenet_v2_mnasfpn',
        'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19',
        'ssd_mobilenet_v1_1_metadata_1',
        'ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess'
    ]
    :param op:
    :return:
    """
    if op == 1:
        od_solo = {
            'model_names': ['ssd_mobilenet_v2_mnasfpn'],
            'dis_size': (640, 480),
            'grid': (1, 1),
        }
        od_meta_dict = od_solo
    elif op == 2:
        od_dual = {
            'model_names': [
                'coco_ssd_mobilenet_v1_1_0_quant_2018_06_29',
                'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19'
            ],
            'dis_size': (640 * 2, 480),
            'grid': (1, 2),
        }
        od_meta_dict = od_dual
    elif op == 3:
        od_all_models = {
            'model_names': od.OdBaseModel.get_object_detection_models(m_type='Tfl', ack=False),
            'dis_size': 'fs',
            'grid': (2, 4),
        }
        od_meta_dict = od_all_models
    else:
        od_meta_dict = None
    return od_meta_dict


def _get_models(model_names: list) -> list:
    models = []
    for model_name in model_names:
        m_save_dir = '{}/object_detection/{}'.format(mtt.MODELS, model_name)
        model = od.TflOdModel(
            save_load_dir=m_save_dir,
            model_name=model_name,
            allowed_class=None,
            threshold=0.5,
            nms={'score_threshold': 0.4, 'nms_threshold': 0.4}
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
            work_every_x_frames=VID_X_FRAMES_TFL,
            dis_size=od_meta_dict['dis_size'],
            grid=od_meta_dict['grid'],
        )
    print('{}'.format('-' * 20))
    return
