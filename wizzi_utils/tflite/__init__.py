"""
for tflite_runtime:
pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime # windows 10
"""
try:
    from wizzi_utils.tflite.tflite_tools import *
except ModuleNotFoundError as e:
    pass

from wizzi_utils.tflite import test
