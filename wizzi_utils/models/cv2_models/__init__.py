try:
    from wizzi_utils.models.cv2_models import object_detection as od
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass

try:
    from wizzi_utils.models.cv2_models import pose_detection as pd
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass

try:
    from wizzi_utils.models.cv2_models import tracking as tr
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass
