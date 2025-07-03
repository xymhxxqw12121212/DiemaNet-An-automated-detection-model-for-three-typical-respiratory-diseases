import torch.nn as nn
import torch
import torch.nn.functional as F
from .attention.EMA import EMA

class GoogleNet_EMA(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_EMA, self).__init__()
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

        self.inception5a = Inception_EMA(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_EMA(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)  # 4a的输出
            self.aux2 = InceptionAux(528, num_classes)  # 4d的输出

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

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
        aux1 = self.aux1(x)
        x = self.inception4b(x) #torch.Size([64, 512, 6, 4])
        x = self.inception4c(x) #torch.Size([64, 512, 6, 4])
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux2, aux1
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class GoogleNet_openl_EMA(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_openl_EMA, self).__init__()
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
        self.inception5a = Inception_EMA(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_EMA(832, 384, 192, 384, 48, 128, 128)
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)  # 4a的输出
            self.aux2 = InceptionAux(528, num_classes)  # 4d的输出
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

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
        aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux2, aux1
        # return x
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
        outputs = torch.cat(outputs, 1)
        return outputs # 通过cat函数将这4个分支进行合并，在第一个维度也就是channel深度进行合并
class Inception_EMA(nn.Module):  # Inception_ema模板
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception_EMA, self).__init__()

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
        if ((ch1x1 + ch3x3 + ch5x5 + pool_proj)%32 == 0 ):
            self.atten = EMA(ch1x1 + ch3x3 + ch5x5 + pool_proj)
        else:
            self.atten = EMA(ch1x1 + ch3x3 + ch5x5 + pool_proj,factor=16)
    def forward(self, x):  # 正向传播过程
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]  # 将4个分支的输出放入到一个列表当中
        outputs = torch.cat(outputs, 1)
        outputs = self.atten(outputs)
        return outputs # 通过cat函数将这4个分支进行合并，在第一个维度也就是channel深度进行合并


class InceptionAux(nn.Module):  # 定义辅助分类器模板
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 2, 3]

        self.fc1_1 = nn.Linear(768, 384)  # 768是mfcc和gfcc展平后的节点个数128*3*2
        self.fc1_2 = nn.Linear(12288, 384)  # 768是openl展平后的节点个数128*3*64
        self.fc1_3 = nn.Linear(3072, 384)  # 3072是wav2vec展平后的节点个数128*1*24
        self.fc2 = nn.Linear(384, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14 输入特征矩阵的维度
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)  # 当我们实例化一个模型model后，可以通过model.train()和model.eval()来控制模型的状态，
        # 在model.train()模式下self.training=True，在model.eval()模式下self.training=False
        # N x 2048
        if x.shape[1]==768:
            x = F.relu(self.fc1_1(x), inplace=True)
        if x.shape[1]==12288:
            x = F.relu(self.fc1_2(x), inplace=True)
        if x.shape[1]==3072:
            x = F.relu(self.fc1_3(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x



class BasicConv2d(nn.Module):  # 卷积模板文件
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # 正向传播过程
        x = self.conv(x)
        x = self.relu(x)
        return x


class GoogleNet_EMA2(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_EMA2, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # ceil_mode=true 得到的小数向上取整 ceil_mode=false 向下取整
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
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
        self.inception5a = Inception_EMA(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_EMA(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
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

class GoogleNet_openl_EMA2(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet_openl_EMA2, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=(1, 2), padding=1,
                                     ceil_mode=False)  # ceil_mode=true 得到的小数向上取整 ceil_mode=false 向下取整
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
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
        self.inception5a = Inception_EMA(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_EMA(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1))  # 自适应平均池化下采样操作（1，1）是输出特征矩阵的高和宽，好处就是无论输入特征矩阵的高和宽是什么样的大小，我们都能够我们所指定的一个特征矩阵的高和宽
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
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

if __name__ == '__main__':
    a = torch.randn(32,1,100,64)
    b = torch.randn(32,1,6,512)
    model = GoogleNet_openl_EMA2(num_classes=4, aux_logits=True, init_weights=True)
    y = model(a)
    print(y.shape)