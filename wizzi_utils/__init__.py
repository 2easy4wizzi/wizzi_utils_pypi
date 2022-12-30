"""
# package wizzi utils:
"""

# default package - available without extra namespace
from wizzi_utils.misc import *

silence_cv_warn()
__version__ = version()

# extra packages - available with extra namespace - requires extra modules
# from wizzi_utils import google as got  # noqa: E402
from wizzi_utils import json as jt  # noqa: E402
from wizzi_utils import open_cv as cvt  # noqa: E402
from wizzi_utils import pyplot as pyplt  # noqa: E402
from wizzi_utils import socket as st  # noqa: E402
from wizzi_utils import torch as tt  # noqa: E402
from wizzi_utils import tflite as tflt  # noqa: E402
from wizzi_utils import tts  # noqa: E402
from wizzi_utils import models  # noqa: E402


def test_all_modules():
    # misc package
    test.test_all()

    # need to make couple of steps before
    # try:
    #     # google package
    #     got.test.test_all()
    # except AttributeError as err:
    #     exception_error(err, real_exception=True)

    try:
        # json package
        jt.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # open_cv package
        cvt.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # pyplot package
        pyplt.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # socket package
        st.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # torch package
        tt.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # tflite package
        tflt.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # tts package
        tts.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)

    try:
        # models package
        models.test.test_all()
    except AttributeError as err:
        exception_error(err, real_exception=True)
    return
