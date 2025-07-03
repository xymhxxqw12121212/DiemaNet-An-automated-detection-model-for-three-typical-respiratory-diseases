import torch
import torchvision
from torch import Tensor
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
class ResNet18(nn.Module):
    def __init__(self,num_classes,
                 model1=torchvision.models.resnet18(pretrained=False),
                 ):
        super(ResNet18, self).__init__()
        model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, num_classes)
        self.predict1 = model1
    def forward(self, x1):
        x = self.predict1(x1)
        return x

class ResNet50(nn.Module):
    def __init__(self,
                 model1=torchvision.models.resnet50(pretrained=False),
                 ):
        super(ResNet50, self).__init__()
        model1.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model1.fc.in_features
        model1.fc = nn.Linear(num_ftrs, 4)
        self.predict1 = model1
    def forward(self, x1):
        x = self.predict1(x1)
        return x

class fusion_resnet18(nn.Module):
    def __init__(self,num_classes):
        super(fusion_resnet18, self).__init__()
        self.predict1 = ResNet18(num_classes=256)
        self.predict2 = ResNet18(num_classes=256)
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=num_classes)
    def forward(self, x1, x2):
        out1 = self.predict1(x1)
        out2 = self.predict2(x2)
        fusion_out = out1+out2
        # fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        return x


if __name__ == '__main__':
    x1 = torch.randn(32,1,6,512)
    x2 =torch.randn(32,1,49,768)
    # model = GoogleNet_openl(num_classes=4, aux_logits=True, init_weights=True)
    model = ResNet18(num_classes=4)
    model = fusion_resnet18(num_classes=4)
    y=model(x1,x2)
    print(y.shape)