import torch.nn as nn
from torchvision import models

class fcn_resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=pretrained)
        self.fcn_resnet50.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    
    def forward(self, x):
        return self.fcn_resnet50(x)['out']