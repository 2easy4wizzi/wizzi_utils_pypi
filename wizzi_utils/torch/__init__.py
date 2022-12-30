"""
for torch 181 gpu cuda 101:
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
for torch 182 gpu cuda 102:
pip install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
"""
try:
    from wizzi_utils.torch.torch_tools import *
except ModuleNotFoundError as e:
    pass

from wizzi_utils.torch import test
