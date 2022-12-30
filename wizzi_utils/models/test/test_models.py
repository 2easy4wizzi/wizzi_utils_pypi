from wizzi_utils.misc import misc_tools as mt


def test_cv2_object_detection_models():
    try:
        from wizzi_utils.models.test import test_cv2_od

        test_cv2_od.test_all()
    except (ModuleNotFoundError, AttributeError) as e:
        mt.exception_error(e)
        print('\tIt\'s possible you dont have open cv')
    return


def test_tflite_object_detection_models():
    try:
        from wizzi_utils.models.test import test_tflite_od

        test_tflite_od.test_all()
    except (ModuleNotFoundError, AttributeError) as e:
        mt.exception_error(e)
        print('\tIt\'s possible you dont have tflite')
    return


def test_cv2_pose_detection_models():
    try:
        from wizzi_utils.models.test import test_cv2_pd

        test_cv2_pd.test_all()
    except (ModuleNotFoundError, AttributeError) as e:
        mt.exception_error(e)
        print('\tIt\'s possible you dont have open cv')
    return


def test_tflite_pose_detection_models():
    try:
        from wizzi_utils.models.test import test_tflite_pd

        test_tflite_pd.test_all()
    except (ModuleNotFoundError, AttributeError) as e:
        mt.exception_error(e)
        print('\tIt\'s possible you dont have tflite')
    return


def test_cv2_tracking_models():
    try:
        from wizzi_utils.models.test import test_cv2_tracking

        test_cv2_tracking.test_all()
    except (ModuleNotFoundError, AttributeError) as e:
        mt.exception_error(e)
        print('\tIt\'s possible you dont have open cv contrib')
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    test_cv2_object_detection_models()
    test_tflite_object_detection_models()
    test_cv2_pose_detection_models()
    test_tflite_pose_detection_models()
    test_cv2_tracking_models()
    print('{}'.format('-' * 20))
    return
