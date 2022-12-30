try:
    from wizzi_utils.models.tflite_models import object_detection as od
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass

try:
    from wizzi_utils.models.tflite_models import pose_detection as pd
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass
