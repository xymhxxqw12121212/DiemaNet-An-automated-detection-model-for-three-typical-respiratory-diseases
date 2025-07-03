import torch.nn as nn
import torch
import torch.nn.functional as F

class GoogleNet_SE(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_SE, self).__init__()
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
        self.se = SE_Block(1024, reduction=16)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):  # 网络的正向传播过程
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x) #torch.Size([64, 256, 12, 8])
        x = self.inception3b(x) #torch.Size([64, 480, 12, 8])
        x = self.maxpool3(x)
        x = self.inception4a(x) #torch.Size([64, 512, 6, 4])
        x = self.inception4b(x) #torch.Size([64, 512, 6, 4])
        x = self.inception4c(x) #torch.Size([64, 512, 6, 4])
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x= self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GoogleNet_openl_SE(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_openl_SE, self).__init__()
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
        self.se = SE_Block(1024, reduction=16)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):  # 网络的正向传播过程
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


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
        outputs = torch.cat(outputs, 1)
        return outputs # 通过cat函数将这4个分支进行合并，在第一个维度也就是channel深度进行合并

class BasicConv2d(nn.Module):  # 卷积模板文件
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # 正向传播过程
        x = self.conv(x)
        x = self.relu(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上



if __name__ == '__main__':
    a = torch.randn(32,1,100,64)
    b = torch.randn(32,1,6,512)
    model = GoogleNet_SE(num_classes=4, aux_logits=True, init_weights=True)
    y = model(a)
    print(y.shape)