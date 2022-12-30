try:
    from wizzi_utils.models.test.test_models import *
    from wizzi_utils.models.test.shared_code_for_tests import *
except (ModuleNotFoundError, AttributeError, ImportError) as e:
    pass
