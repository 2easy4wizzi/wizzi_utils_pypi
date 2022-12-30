try:
    from wizzi_utils.models import models_configs
    from wizzi_utils.models import base_models
    from wizzi_utils.models import cv2_models
    from wizzi_utils.models import tflite_models
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass

from wizzi_utils.models import test
