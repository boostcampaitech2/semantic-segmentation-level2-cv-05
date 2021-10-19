from .unet import UNet
from .base import fcn_resnet50

# https://github.com/qubvel/segmentation_models.pytorch/tree/35d79c1aa5fb26ba0b2c1ec67084c66d43687220
# !pip install git+https://github.com/qubvel/segmentation_models.pytorch
from segmentation_models_pytorch import (Unet, UnetPlusPlus, MAnet, Linknet,
                                         FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN)