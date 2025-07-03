import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os


class Res_sampleCNN(nn.Module):
    def __init__(self,inplanes=1,numclass = 4):
        super(Res_sampleCNN,self).__init__()
        self.numclass = numclass
        self.inplanes =inplanes
        self.conv1 = nn.Conv1d(self.inplanes,64, kernel_size= 3, stride=3, padding=0)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.layers1 = Res_Conv1D(64,128)
        self.layers2 = Res_Conv1D(128, 256)
        self.layers3 = Res_Conv1D(256, 512)
        self.layers4 = Res_Conv1D(512, 1024)
        self.layers5 = Res_Conv1D(1024, 2048,pool=False)
        self.fc1 = nn.Linear(2048, 256)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, self.numclass)
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.layers4(out)
        out = self.layers5(out)
        a,b,_ = out.shape
        out = out.view(a,b)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
# 定义SEblock模型
class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.glob_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Res_Conv1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes,pool = True):
        """定义BasicBlock残差块类
        参数：
            inplanes (int): 输入的Feature Map的通道数
            planes (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是planes*expansion
        """
        super(Res_Conv1D, self).__init__()
        self.pool = pool
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size= 3, stride=3, padding=0)
        self.drop1 = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size = 3, stride=1, padding=2, dilation=2)
        self.se = SE_Block(channel=planes)
        self.downsample = nn.Conv1d(inplanes, planes, kernel_size = 3, stride=3, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        if self.pool == True:
            self.lastpool = nn.MaxPool1d(kernel_size =1,stride=2)


    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.drop1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.se(out)
        out2 = self.downsample(identity)
        out += out2
        out = self.relu3(out)
        if self.pool == True:
            out = self.lastpool(out)
        return out

if __name__ == '__main__':
    a = torch.randn(5,1,16000)
    model = Res_sampleCNN(numclass=4)
    print(model(a))