import torch.nn as nn
import torch
import torch.nn.functional as F
from RGA.rga_modules import RGA_Module

class GoogleNet_RGA(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_RGA, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # ceil_mode=true 得到的小数向上取整 ceil_mode=false 向下取整

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)  # 第一个64是输入特征矩阵深度，第二个64是卷积核的个数
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  # 第一个参数是输入特征矩阵深度，后面的参数都是按照表格中的参数
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
        self.rga_att5a = RGA_Module(832, (3 * 2))
        self.rga_att5b = RGA_Module(1024, (3 * 2))
    def forward(self, x):  # 网络的正向传播过程
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # if self.training and self.aux_logits:  # eval model lose this layer
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        # if self.training and self.aux_logits:  # eval model lose this layer
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        x = self.rga_att5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        x = self.rga_att5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        # if self.training and self.aux_logits:  # eval model lose this layer
        # return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class GoogleNet_openl_RGA(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_openl_RGA, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=(1, 2), padding=1,
                                     ceil_mode=False)  # ceil_mode=true 得到的小数向上取整 ceil_mode=false 向下取整

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)  # 第一个64是输入特征矩阵深度，第二个64是卷积核的个数
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=(1, 2), padding=1, ceil_mode=False)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  # 第一个参数是输入特征矩阵深度，后面的参数都是按照表格中的参数
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=(1, 2), padding=1, ceil_mode=False)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
        self.rga_att5a = RGA_Module(832, (3 * 32))
        self.rga_att5b = RGA_Module(1024, (3 * 32))
    def forward(self, x):  # 网络的正向传播过程
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # if self.training and self.aux_logits:  # eval model lose this layer
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        # if self.training and self.aux_logits:  # eval model lose this layer
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        print(x.shape)
        x = self.rga_att5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        print(x.shape)
        x = self.rga_att5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        # if self.training and self.aux_logits:  # eval model lose this layer
        # return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
class Inception(nn.Module):  # Inception模板
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(  # 传入非关键字的参数
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出特征矩阵大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )


    def forward(self, x):  # 正向传播过程
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]  # 将4个分支的输出放入到一个列表当中
        return torch.cat(outputs, 1)  # 通过cat函数将这4个分支进行合并，在第一个维度也就是channel深度进行合并



class BasicConv2d(nn.Module):  # 卷积模板文件
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # 正向传播过程
        x = self.conv(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    a = torch.randn(32,1,100,64)
    b = torch.randn(32,1,6,512)
    # model = GoogleNet_RGA(num_classes=4, aux_logits=True, init_weights=True)
    model = GoogleNet_openl_RGA(num_classes=4, aux_logits=True, init_weights=True)
    y = model(b)
    print(y.shape)