"""
AIO -- All Model in One
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .pydiffres import DiffRes
from .GoogleNet import GoogleNet,GoogleNet_openl
from .GoogleNet_SE import GoogleNet_SE, GoogleNet_openl_SE
from .GoogleNet_EMA import GoogleNet_EMA, GoogleNet_openl_EMA,GoogleNet_EMA2, GoogleNet_openl_EMA2
# from .GoogleNet_RGA import GoogleNet_RGA, GoogleNet_openl_RGA
class ch2_model_noatt(nn.Module):
    def __init__(self):
        super(ch2_model_noatt, self).__init__()

        # googlenet for MFCC
        self.googlenet_model = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        # mfccfeture to weights
        self.post_mfcc_att_dropout = nn.Dropout(p=0.1)
        # openl3
        self.post_wav_layer1 = nn.AvgPool2d(kernel_size=(6,1))
        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer2 = nn.Linear(512, 128)  # 512 for 1 and 768 for 2

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)

    def forward(self,audio_mfcc, audio_openl3):
        # audio_mfcc: [batch, 1,100,64]
        # audio_wav: [batch,1,6, 512]

        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=3)

        # mfcc-googlenet
        audio_mfcc_p, _, _ = self.googlenet_model(audio_mfcc)  # [batch, 128]


        # openl3
        # audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1, 512)  # [batch,6, 512]
        audio_openl3 = self.post_wav_layer1(audio_openl3) # [batch,1, 512]
        audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1)  # [batch, 512]
        audio_openl3_d = self.post_wav_dropout(audio_openl3)  # [batch,512]
        audio_openl3_p = F.relu(self.post_wav_layer2(audio_openl3_d), inplace=False)  # [batch, 128]

        ## combine()
        audio_att = torch.cat([audio_mfcc_p, audio_openl3_p], dim=-1)  # [batch, 256]
        audio_att_d_1 = self.post_att_dropout(audio_att)  # [batch, 256]
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)  # [batch, 128]
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128]
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 4]
        return output_att

class ch2_mfcc_wav2vec(nn.Module):
    def __init__(self):
        super(ch2_mfcc_wav2vec, self).__init__()

        # googlenet for MFCC
        self.googlenet_model = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        # mfccfeture to weights
        self.post_mfcc_att_dropout = nn.Dropout(p=0.1)
        # wav2vec
        self.post_wav_layer1 = nn.AvgPool2d(kernel_size=(49,1))
        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer2 = nn.Linear(768, 128)  # 512 for 1 and 768 for 2

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)

    def forward(self,audio_mfcc, audio_openl3):
        # audio_mfcc: [batch, 1,100,64]
        # audio_wav: [batch,1,6, 512]
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=3)
        # mfcc-googlenet
        audio_mfcc_p, _, _ = self.googlenet_model(audio_mfcc)  # [batch, 128]
        # openl3
        # audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1, 512)  # [batch,6, 512]
        audio_openl3 = self.post_wav_layer1(audio_openl3) # [batch,1, 512]
        audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1)  # [batch, 512]
        audio_openl3_d = self.post_wav_dropout(audio_openl3)  # [batch,512]
        audio_openl3_p = F.relu(self.post_wav_layer2(audio_openl3_d), inplace=False)  # [batch, 128]

        ## combine()
        audio_att = torch.cat([audio_mfcc_p, audio_openl3_p], dim=-1)  # [batch, 256]
        audio_att_d_1 = self.post_att_dropout(audio_att)  # [batch, 256]
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)  # [batch, 128]
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128]
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 4]
        return output_att

class ch2_gfcc_wav2vec(nn.Module):
    def __init__(self):
        super(ch2_gfcc_wav2vec, self).__init__()

        # googlenet for MFCC
        self.googlenet_model = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        # mfccfeture to weights
        self.post_mfcc_att_dropout = nn.Dropout(p=0.1)
        # wav2vec
        self.post_wav_layer1 = nn.AvgPool2d(kernel_size=(49,1))
        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer2 = nn.Linear(768, 128)  # 512 for 1 and 768 for 2

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)

    def forward(self,audio_mfcc, audio_openl3):
        # audio_mfcc: [batch, 1,100,64]
        # audio_wav: [batch,1,6, 512]
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=3)
        # mfcc-googlenet
        audio_mfcc_p, _, _ = self.googlenet_model(audio_mfcc)  # [batch, 128]
        # openl3
        # audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1, 512)  # [batch,6, 512]
        audio_openl3 = self.post_wav_layer1(audio_openl3) # [batch,1, 512]
        audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1)  # [batch, 512]
        audio_openl3_d = self.post_wav_dropout(audio_openl3)  # [batch,512]
        audio_openl3_p = F.relu(self.post_wav_layer2(audio_openl3_d), inplace=False)  # [batch, 128]

        ## combine()
        audio_att = torch.cat([audio_mfcc_p, audio_openl3_p], dim=-1)  # [batch, 256]
        audio_att_d_1 = self.post_att_dropout(audio_att)  # [batch, 256]
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)  # [batch, 128]
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128]
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 4]
        return output_att

class ch2_gfcc_openl(nn.Module):
    def __init__(self):
        super(ch2_gfcc_openl, self).__init__()

        # googlenet for GFCC
        self.googlenet_model = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        # mfccfeture to weights
        self.post_mfcc_att_dropout = nn.Dropout(p=0.1)
        # openl
        self.post_wav_layer1 = nn.AvgPool2d(kernel_size=(6,1))
        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer2 = nn.Linear(512, 128)  # 512 for 1 and 768 for 2

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(256, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, 4)

    def forward(self,audio_mfcc, audio_openl3):
        # audio_mfcc: [batch, 1,100,64]
        # audio_wav: [batch,1,6, 512]
        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=3)
        # mfcc-googlenet
        audio_mfcc_p, _, _ = self.googlenet_model(audio_mfcc)  # [batch, 128]
        # openl3
        # audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1, 512)  # [batch,6, 512]
        audio_openl3 = self.post_wav_layer1(audio_openl3) # [batch,1, 512]
        audio_openl3 = audio_openl3.reshape(audio_openl3.shape[0], -1)  # [batch, 512]
        audio_openl3_d = self.post_wav_dropout(audio_openl3)  # [batch,512]
        audio_openl3_p = F.relu(self.post_wav_layer2(audio_openl3_d), inplace=False)  # [batch, 128]

        ## combine()
        audio_att = torch.cat([audio_mfcc_p, audio_openl3_p], dim=-1)  # [batch, 256]
        audio_att_d_1 = self.post_att_dropout(audio_att)  # [batch, 256]
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)  # [batch, 128]
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128]
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 4]
        return output_att



class fusion_googlenet_googlenet(nn.Module):
    def __init__(self,):
        super(fusion_googlenet_googlenet, self).__init__()
        self.googlenet_model1 = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        self.googlenet_model2 = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x1, x2):
        out1 = self.googlenet_model1(x1)
        out2 = self.googlenet_model2(x2)
        fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        return x

class fusion_googlenet_googlenet_openl(nn.Module):
    def __init__(self,):
        super(fusion_googlenet_googlenet_openl, self).__init__()
        self.googlenet_model1 = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        self.googlenet_model2 = GoogleNet_openl(num_classes=128, aux_logits=True, init_weights=True)
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x1, x2):
        out1 = self.googlenet_model1(x1)
        out2 = self.googlenet_model2(x2)
        fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        return x


class TCCNN_SE(nn.Module):
    def __init__(self,):
        super(TCCNN_SE, self).__init__()
        self.googlenet_model1 = GoogleNet_SE(num_classes=128, aux_logits=True, init_weights=True)
        self.googlenet_model2 = GoogleNet_openl_SE(num_classes=128, aux_logits=True, init_weights=True)
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x1, x2):
        out1 = self.googlenet_model1(x1)
        out2 = self.googlenet_model2(x2)
        fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        return x

class TCCNN_E(nn.Module):
    def __init__(self,):
        super(TCCNN_E, self).__init__()
        self.googlenet_model1 = GoogleNet_EMA2(num_classes=128, aux_logits=True, init_weights=True)
        self.googlenet_model2 = GoogleNet_openl_EMA2(num_classes=128, aux_logits=True, init_weights=True)
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x1, x2):
        out1 = self.googlenet_model1(x1)
        out2 = self.googlenet_model2(x2)
        fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        return x

class DiemaNet(nn.Module):
    def __init__(self,):
        super(DiemaNet, self).__init__()
        self.googlenet_model1 = GoogleNet_EMA2(num_classes=128, aux_logits=True, init_weights=True)
        self.get_ret1 = DiffRes(
            in_t_dim=167,  #使用6ms步长为167，降维率
            in_f_dim=64,
            dimension_reduction_rate=0.4,
            # 1.5s:0.333,2s:0.5, 2.5s:0.6, 3s:0.666, 3.5s:0.714,4s:0.75
            learn_pos_emb=False  # If you like to make the resolution encoding learnable
        )
        self.googlenet_model2 = GoogleNet_openl_EMA2(num_classes=128, aux_logits=True, init_weights=True)
        self.get_ret2 = DiffRes(
            in_t_dim=10,  # 2.5s:21 3s:26 3.5s:31 4s:36 4.5s:41: 5s:46 7.5s:71
            in_f_dim=512,  # The frequency dimension of your spectrogram
            dimension_reduction_rate=0.4,
            # 1.5s:0.454,2s:0.625,2.5s:0.714,3s:0.769,3.5s:0.806,4s:0.833
            learn_pos_emb=False  # If you like to make the resolution encoding learnable
        )
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x1, x2):
        ret1 = self.get_ret1(x1)
        out1 = self.googlenet_model1(ret1['feature'])
        loss1 = ret1['guide_loss']
        ret2 = self.get_ret2(x2)
        out2 = self.googlenet_model2(ret2['feature'])
        loss2 = ret2['guide_loss']
        fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        loss = loss1+loss2
        return x, loss
    def _init_weights(self):
        # He初始化适用于ReLU激活函数
        init.kaiming_normal_(self.dn1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dn1.bias, 0)
        init.kaiming_normal_(self.dn2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dn2.bias, 0)


class TCCNN_D(nn.Module):
    def __init__(self,):
        super(TCCNN_D, self).__init__()
        self.googlenet_model1 = GoogleNet(num_classes=128, aux_logits=True, init_weights=True)
        self.get_ret1 = DiffRes(
            in_t_dim=167,  #使用6ms步长为167，降维率
            in_f_dim=64,
            dimension_reduction_rate=0.4,
            # 1.5s:0.333,2s:0.5, 2.5s:0.6, 3s:0.666, 3.5s:0.714,4s:0.75
            learn_pos_emb=False  # If you like to make the resolution encoding learnable
        )
        self.googlenet_model2 = GoogleNet_openl(num_classes=128, aux_logits=True, init_weights=True)
        self.get_ret2 = DiffRes(
            in_t_dim=10,  # 2.5s:21 3s:26 3.5s:31 4s:36 4.5s:41: 5s:46 7.5s:71
            in_f_dim=512,  # The frequency dimension of your spectrogram
            dimension_reduction_rate=0.4,
            # 1.5s:0.454,2s:0.625,2.5s:0.714,3s:0.769,3.5s:0.806,4s:0.833
            learn_pos_emb=False  # If you like to make the resolution encoding learnable
        )
        self.dn1 = nn.Linear(in_features=256, out_features=128)
        self.drop1 = nn.Dropout(p=0.3)
        self.dn2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x1, x2):
        ret1 = self.get_ret1(x1)
        out1 = self.googlenet_model1(ret1['feature'])
        loss1 = ret1['guide_loss']
        ret2 = self.get_ret2(x2)
        out2 = self.googlenet_model2(ret2['feature'])
        loss2 = ret2['guide_loss']
        fusion_out = torch.cat((out1, out2), 1)
        x = F.relu(self.dn1(F.relu(fusion_out)))
        x = self.drop1(x)
        x = self.dn2(x)
        loss = loss1+loss2
        return x, loss
    def _init_weights(self):
        # He初始化适用于ReLU激活函数
        init.kaiming_normal_(self.dn1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dn1.bias, 0)
        init.kaiming_normal_(self.dn2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dn2.bias, 0)

if __name__ == '__main__':
    x1 = torch.randn(32,1,100,64)#1s的mfcc特征
    x2 = torch.randn(32,1,6,512)#1s的openl3特征
    x3 = torch.randn(32,1,49,768)#wav2vec特征
    x4 = torch.randn(32, 1, 98, 64)  # 1s的gfcc特征
    # model = ch2_model_noatt()
    # model = ch2_gfcc_wav2vec()
    # model = fusion_googlenet_resnet18()
    # model = fusion_googlenet_googlenet()
    # model = fusion_googlenet_googlenet_openl()
    # model = ch2_gfcc_openl()
    # model = ch2model_rga()
    # model = fusion_googlenet_googlenet_openl_CA()
    # model = DiemaNet()
    model = TCCNN_SE()
    # model = fusion_googlenet_googlenet_openl_RGA()
    y= model(x1,x2)
    print(y.shape)
    # print(loss1)