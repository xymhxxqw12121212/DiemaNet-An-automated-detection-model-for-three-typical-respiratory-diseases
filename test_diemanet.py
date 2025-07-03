import itertools

import soundfile as sf
import torch
import torch.nn as nn
import librosa
import pandas as pd
import numpy as np
import json
import os
import librosa
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,precision_recall_curve,auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import logging
from pynvml import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from model.GoogleNet import GoogleNet,GoogleNet_openl
from model.GoogleNet_EMA import GoogleNet_EMA
# from model.GoogleNet_RGA import GoogleNet_RGA
# from model.resnet import ResNet18,ResNet50, fusion_resnet18
# from model.resnext import ResNext50
# from model.efficientnet import effnetv2_s
# from model.efficientnet import effnetv2_m
# from model.xception import Xception
from model.CH2_model import *
import warnings
from utils import reshape_numpy



# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

# GPU 的数量
gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)
#指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有没有GPU
#模型移动到gpu
#model.to(device)

print('*******************')
print("测试 mfcc+openl+TCCNN_D_1s-6mss_batch64")
print('*******************')

sample_rate=16000
window_size = int(sample_rate * 25 // 1000 + 1)
window_stride = int(sample_rate * 10 // 1000)

# npz_path ='./feature/test'#本地
npz_path ='/root/autodl-tmp/feature1s_6ms/test'#远程服务器
lable_npz = []
mfcc_npz = []
openl_npz = []
wav2vec_npz = []
gfcc_npz = []
npzs=os.listdir(npz_path)
for npz in npzs:
    if 'mfcc' in npz :
        mfcc_npz.append(npz)
    if 'openl' in npz :
        openl_npz.append(npz)
    if 'lable' in npz :
        lable_npz.append(npz)
X1_test = []
X2_test = []
y_test = []
for i in range(len(lable_npz)):
    # ls = np.load(r'./feature/test/{}'.format(lable_npz[i]))
    ls = np.load(r'/root/autodl-tmp/feature1s_6ms/test/{}'.format(lable_npz[i]))
    keys = ls.files
    for key in keys:
        y_test.append(ls[key])
    ill= lable_npz[i].split('_')[0]
    if len(mfcc_npz) > 0:
        # ms = np.load(r'./feature/test/{}_mfcc.npz'.format(ill))
        ms = np.load(r'/root/autodl-tmp/feature1s_6ms/test/{}_mfcc.npz'.format(ill))
        for key in keys:
            X1_test.append(ms[key])
    if len(openl_npz) > 0:
        # ops = np.load(r'./feature/test/{}_openl3.npz'.format(ill))
        ops = np.load(r'/root/autodl-tmp/feature1s_6ms/test/{}_openl3.npz'.format(ill))
        for key in keys:
            X2_test.append(ops[key])


y_test= np.array(y_test)
X1_test = np.array(X1_test)
X2_test = np.array(X2_test)

for i in range(0,1):
    fold_dict = dict()
    # model_path = r'./modelsave/4class_mfcc_resnet18.pth'
    # model_path = r'./modelsave/4class_mfcc_parallel.pth'
    # model_path = r'./modelsave_balanced_dataset/best_network.pth'
    # model_path = r'./modelsave_balanced_dataset/4class_mfcc_googlenet.pth'
    model_path = r'./modelsave_balanced_dataset/best_network.pth'
    if len(X2_test) > 0:
        # model = fusion_resnet18(num_classes=4)
        #model = ch2_model()
        # model = ch2_model_noatt()
        # model = ch2_mfcc_wav2vec()
        # model = ch2_gfcc_wav2vec()
        # model = ch2_gfcc_openl()
        # model = fusion_googlenet_resnet18()  # 用于mfcc/gfcc+openl
        # model = fusion_googlenet_googlenet()  # 用于mfcc、gfcc、wav2vec两两组合
        # model = TCCNN_D()
        model = DiemaNet()  # 用于mfcc、openl两两组合
        # model = fusion_temp_resnet18()  # 用于mfcc、openl两两组合
        # model = model = fusion_googlenet_googlenet_openl()
        # model = GoogleNet(num_classes=4, aux_logits=True, init_weights=True)# wav2vec+googlenet
        # model = ResNet18()  # resnet18模型
        # model = GoogleNet_openl(num_classes=4, aux_logits=True, init_weights=True)
    else:
        model = ResNet18(num_classes=4)  # resnet18模型
        # model = ResNet50()  # resnet50模型
        # model = ResNext50()
        # model = GoogleNet(num_classes=4, aux_logits=True, init_weights=True) #注意，该模型返回三个loss值，train函数需要修改loss
        # model = GoogleNet_openl(num_classes=4, aux_logits=True, init_weights=True)  # 注意，该模型返回三个loss值，train函数需要修改loss
        # model = GoogleNet_EMA(num_classes=4, aux_logits=True, init_weights=True)
        # model = GoogleNet_RGA(num_classes=4, aux_logits=True, init_weights=True)
        # model = Xception(num_classes=4)
        # model = resnet50()
        # model = effnetv2_s(num_classes = 4)
        # model = effnetv2_m(num_classes = 4)
    #模型放到GPU
    model.load_state_dict(torch.load(model_path))
    model=model.to(device)
    if len(X2_test)>0:
        testdataset = TensorDataset(
            torch.tensor(X1_test, dtype=torch.float32),
            torch.tensor(X2_test, dtype=torch.float32),
            torch.tensor(y_test))
    else:
        testdataset = TensorDataset(
            torch.tensor(X1_test, dtype=torch.float32),
            torch.tensor(y_test))
    testdataloader = DataLoader(testdataset, batch_size=64, shuffle=False)
    total_time = 0
    model.eval()
    with torch.no_grad():
        pred_label= np.array([])
        for batch_idx, dataset in enumerate(testdataloader):
            if len(dataset) == 2:
                data1, labelOrg = dataset  # 仅用mfcc特征
                data1 = data1.to(device)
                predict = model(data1)
                #googlenet
                # predict, aux_logit2, aux_logit1 = model(data1)
            elif len(dataset) == 3:
                data1, data2, labelOrg = dataset  # 两种特征
                data1 = data1.to(device)
                data2 = data2.to(device)
                start_time = time.time()
                predict,_ = model(data1, data2)
                cost_time = time.time() - start_time
                total_time += cost_time
                #googlenet
                # predict, aux_logit2, aux_logit1 = model(data1,data2)
            predict=predict.cpu()
            pred_label=np.concatenate((pred_label,torch.argmax(predict,dim=1)))

    print('********************')
    print("total_time={}".format(total_time))
    pred_label=np.array(pred_label)

    # 将预测结果保存为csv
    # csvdata = dict()
    # csv_path=r"./results1_3s.csv"
    # csvdata={"name":N,"true_label":y_test[i],"pred_label":pred_label}
    # df=pd.DataFrame(csvdata)
    # df.to_csv(csv_path,sep=',',index=False,header=True)

    y_test=np.array(y_test)
    y_pred = np.array(pred_label)
    print("-" * 10)

# 定义类别标签
classes = ['healthy', 'covid', 'asthma', 'copd']
cm = confusion_matrix(y_test, y_pred)
"Pre=TP/(TP+FP)"
"Recall= TP/(TP+FN)"
"F1= 2*Pre*Recall/(Pre+Recall)"
health_TP= cm[0,0];health_FP=sum(cm[[1,2,3],0]);health_FN=sum(cm[0,[1,2,3]])
covid_TP= cm[1,1];covid_FP=sum(cm[[0,2,3],1]);covid_FN=sum(cm[1,[0,2,3]])
asthma_TP= cm[2,2];asthma_FP=sum(cm[[0,1,3],2]);asthma_FN=sum(cm[2,[0,1,3]])
copd_TP= cm[3,3];copd_FP=sum(cm[[0,1,2],3]);copd_FN=sum(cm[3,[0,1,2]])
def cal_prerecf1(tp,fp,fn):
    pre= tp/(tp+fp)
    rec= tp/(tp+fn)
    f1= 2*pre*rec/(pre+rec)
    return pre,rec,f1
def cal_avg(x1, x2, x3, x4):
    s = 0.25
    avg_x = s * x1+ s * x2 + s * x3 + s * x4
    return avg_x
def log_result(state,pre,rec,f1):
    print("********************")
    print(state)
    print(pre)
    print(rec)
    print(f1)
    print("********************")
acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/np.sum(cm)
health_pre, health_rec, health_f1= cal_prerecf1(health_TP, health_FP, health_FN)
covid_pre, covid_rec, covid_f1= cal_prerecf1(covid_TP, covid_FP, covid_FN)
asthma_pre, asthma_rec, asthma_f1= cal_prerecf1(asthma_TP, asthma_FP, asthma_FN)
copd_pre, copd_rec, copd_f1= cal_prerecf1(copd_TP, copd_FP, copd_FN)
avg_pre= cal_avg(health_pre, covid_pre, asthma_pre, copd_pre)
avg_rec= cal_avg(health_rec, covid_rec, asthma_rec, copd_rec)
avg_f1= cal_avg(health_f1, covid_f1, asthma_f1, copd_f1)
log_result(state="health",pre=health_pre, rec=health_rec, f1=health_f1)
log_result(state="covid",pre=covid_pre, rec=covid_rec, f1=covid_f1)
log_result(state="asthma",pre=asthma_pre, rec=asthma_rec, f1=asthma_f1)
log_result(state="copd",pre=copd_pre, rec=copd_rec, f1=copd_f1)

print("avg_pre:",avg_pre)
print("avg_rec:",avg_rec)
print("avg_f1",avg_f1)
print("acc:",acc)

###将实验结果保存为csv文件
csv_data = [
    ["type", "pre", "rec", "f1", "acc"],
    ["health", health_pre, health_rec, health_f1,None],
    ["covid", covid_pre, covid_rec, covid_f1,None],
    ["asthma", asthma_pre, asthma_rec, asthma_f1,None],
    ["copd", copd_pre, copd_rec, copd_f1, None],
    ["avg", avg_pre, avg_rec, avg_f1, acc]
]
df = pd.DataFrame(csv_data)
filename = "results.csv"
df.to_csv(filename, index = False)
# 绘制混淆矩阵
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(set(y_test)))  # length of classes
plt.xticks(tick_marks,classes)
plt.yticks(tick_marks,classes)

# 在混淆矩阵中添加文本说明
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    tex = cm[i, j]
    # tex = cm[i,j]/sum(cm[i,:])
    # tex = round(tex,2)
    plt.text(j, i, tex,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix2')
plt.show()


