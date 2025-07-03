import torch
import torchvision
from torch import Tensor
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
class ResNext50(nn.Module):
    def __init__(self,
                 model1=torchvision.models.resnext50_32x4d(),
                 ):
        super(ResNext50, self).__init__()
        model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, 4)
        self.predict1 = model1
    def forward(self, x1):
        x = self.predict1(x1)
        return x