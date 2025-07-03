import numpy as np
import os
import torch
import pandas as pd
from torch import nn
from utils import Countalpha,plot_loss_acc, get_feature, reshape_numpy
from pynvml import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from model.GoogleNet import GoogleNet, GoogleNet_openl
from model.GoogleNet_EMA import GoogleNet_EMA
# from model.GoogleNet_RGA import GoogleNet_RGA
from model.resnet import ResNet18, ResNet50, fusion_resnet18
# from model.resnext import ResNext50
# from model.efficientnet import effnetv2_s, effnetv2_m
from model.transformer import TransformerModel
from model.CH2_model import *
from early_stopping import EarlyStopping
# from model.xception import Xception
from torch.optim.lr_scheduler import StepLR
import warnings
warnings.filterwarnings("ignore")
# GPU 的数量
# gpu_count = torch.cuda.device_count()
# print("gpu_count=",gpu_count)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有没有GPU
print(device)
print('*******************')
print("训练 mfcc+openl+tccnn_e_batch64")
print('*******************')

train_acc_graph = []
train_loss_graph = []
test_acc_graph = []
test_loss_graph = []
#训练和测试模型
def train_and_test(model, dataLoader,testdataLoader, lossFunc, n_epoch,device='cuda'):
    total_step = len(dataloader)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.00001)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(n_epoch):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        for batch_idx, dataset in enumerate(dataLoader):
            optimizer.zero_grad()
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            if len(dataset)== 2:
                data1, labelOrg = dataset#仅用mfcc特征
                data1 = data1.to(device)
                predict = model(data1)
                ##googlenet
                # predict,aux_logit2,aux_logit1 = model(data1)
            elif len(dataset)== 3:
                data1,data2,labelOrg = dataset#两种特征
                data1 = data1.to(device)
                data2 = data2.to(device)
                predict = model(data1, data2)
                ##googlenet
                # predict, aux_logit2, aux_logit1 = model(data1,data2)
            label = labelOrg.to(device)
            loss0 = lossFunc(predict,label.long())
            # loss1 = lossFunc(aux_logit1, label.long())
            # loss2 = lossFunc(aux_logit2, label.long())
            # loss=loss0 + loss1*0.3 + loss2*0.3
            loss = loss0
            loss.backward()
            optimizer.step()
            # Tensor.item() 类型转换，返回一个数
            sum_loss += loss.item()
            _, pred = torch.max(predict, 1)
            correct += pred.eq(label.data).sum().item()
            total += label.size(0)


        #模型测试
        eval_loss = 0
        test_correct=0
        test_total=0
        model.eval()
        with torch.no_grad():
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
                    predict = model(data1, data2)
                    ##googlenet
                    # predict, aux_logit2, aux_logit1 = model(data1,data2)
                label = labelOrg.to(device)# float32
                loss0 = lossFunc(predict, label.long())
                # loss1 = lossFunc(aux_logit1, label.long())
                # loss2 = lossFunc(aux_logit2, label.long())
                # loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                loss = loss0
                # 记录误差和acc
                eval_loss += loss.item()
                _, pred = torch.max(predict, 1)
                test_correct += pred.eq(label.data).sum().item()
                test_total += label.size(0)

        scheduler.step()  # 更新学习率
        train_loss.append(sum_loss/total_step)
        train_acc.append(100. * correct / total)
        test_loss.append(eval_loss/ len(testdataLoader))
        test_acc.append(100. * test_correct / test_total)
        print('Epoch [{}/{}], train_loss: {:.6f}, train_Acc:{:.4f}, test_loss: {:.6f}, test_Acc:{:.4f}'.format(epoch + 1, n_epoch, sum_loss / total_step,
                                                                             100. * correct / total,eval_loss/len(testdataLoader),100. * test_correct / test_total))
        # 早停止
        train_loss_graph.append(sum_loss / total_step)
        train_acc_graph.append(100. * correct / total)
        test_loss_graph.append(eval_loss/len(testdataLoader))
        test_acc_graph.append(100. * test_correct / test_total)
        early_stopping(test_correct / test_total, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
    plot_loss_acc(train_loss, test_loss, train_acc, test_acc)

npz_path ='/root/autodl-tmp/feature/train' #根据需要修改路劲
lable_npz=[]
mfcc_npz=[]
openl_npz=[]
wav2vec_npz=[]
gfcc_npz=[]

npzs=os.listdir(npz_path)
for npz in npzs:
    if 'mfcc' in npz :
        mfcc_npz.append(npz)
    if 'openl' in npz :
        openl_npz.append(npz)
    # if 'wav2vec' in npz :
    #     wav2vec_npz.append(npz)
    # if 'gfcc' in npz :
    #     gfcc_npz.append(npz)
    if 'lable' in npz :
        lable_npz.append(npz)
##读取训练集的特征和标签
X1_train= []
X2_train= []
y_train= []
for i in range(len(lable_npz)):
    ls = np.load(r'/root/autodl-tmp/feature/train/{}'.format(lable_npz[i]))
    keys = ls.files
    for key in keys:
        y_train.append(ls[key])
    ill= lable_npz[i].split('_')[0]
    if len(mfcc_npz) > 0:
        ms = np.load(r'/root/autodl-tmp/feature/train/{}_mfcc.npz'.format(ill))
        for key in keys:
            X1_train.append(ms[key])
    if len(openl_npz) >0:
        ops = np.load(r'/root/autodl-tmp/feature/train/{}_openl3.npz'.format(ill))
        for key in keys:
            X2_train.append(ops[key])
    if len(wav2vec_npz) >0:
        ops = np.load(r'/root/autodl-tmp/feature/train/{}_wav2vec.npz'.format(ill))
        for key in keys:
            X2_train.append(ops[key])
    if len(gfcc_npz) >0:
        ops = np.load(r'/root/autodl-tmp/feature/train/{}_gfcc.npz'.format(ill))
        for key in keys:
            X2_train.append(ops[key])

##读取测试集的特征和标签
npz_path2 ='/root/autodl-tmp/feature/test'
lable2_npz=[]
openl2_npz=[]
mfcc2_npz=[]
wav2vec2_npz=[]
gfcc2_npz=[]
npzs2=os.listdir(npz_path2)
for npz in npzs2:
    if 'mfcc' in npz :
        mfcc2_npz.append(npz)
    if 'openl' in npz :
        openl2_npz.append(npz)
    # if 'wav2vec' in npz :
    #     wav2vec2_npz.append(npz)
    # if 'gfcc' in npz :
    #     gfcc2_npz.append(npz)
    if 'lable' in npz :
        lable2_npz.append(npz)

X1_test = []
X2_test = []
y_test = []
for i in range(len(lable_npz)):
    ls = np.load(r'/root/autodl-tmp/feature/test/{}'.format(lable2_npz[i]))
    keys = ls.files
    for key in keys:
        y_test.append(ls[key])
    ill= lable2_npz[i].split('_')[0]
    if len(mfcc2_npz) > 0:
        ms = np.load(r'/root/autodl-tmp/feature/test/{}_mfcc.npz'.format(ill))
        for key in keys:
            X1_test.append(ms[key])
    if len(openl2_npz) >0:
        ops = np.load(r'/root/autodl-tmp/feature/test/{}_openl3.npz'.format(ill))
        for key in keys:
            X2_test.append(ops[key])
    if len(wav2vec2_npz) >0:
        ops = np.load(r'/root/autodl-tmp/feature/test/{}_wav2vec.npz'.format(ill))
        for key in keys:
            X2_test.append(ops[key])
    if len(gfcc2_npz) >0:
        ops = np.load(r'/root/autodl-tmp/feature/test/{}_gfcc.npz'.format(ill))
        for key in keys:
            X2_test.append(ops[key])
print("length of X2_test:",len(X2_test))
y_train= np.array(y_train)
y_test= np.array(y_test)
X1_train = reshape_numpy(X1_train)
X1_test = reshape_numpy(X1_test)
X2_train = reshape_numpy(X2_train,transpose=False)
X2_test = reshape_numpy(X2_test,transpose=False)
# if len(openl2_npz) or len(wav2vec_npz) >0:
#     X2_train = reshape_numpy(X2_train,transpose=False)
#     X2_test = reshape_numpy(X2_test,transpose=False)


# print("shape of X1_test:",(X1_test.shape))
pred_folds_list = []
y_pred=[]
y_preds=[]
avscores = []
avauc=[]
for i in range(0,1):
    # save_path1 = r'./modelsave_balanced_dataset/TCCNN_1s.pth'  # mfcc+openl_dualgooglenet 2s模型
    # save_path1 = r'./modelsave_balanced_dataset/TCCNN_E.pth'  # mfcc+openl_TCCNN_E模型
    # save_path1 = r'./modelsave_balanced_dataset/gfcc_googlenet.pth'  # gfcc+goolglenet模型
    # save_path1 = r'./modelsave_balanced_dataset/mfcc_googlenet.pth'  # openl+goolglenet模型
    # save_path1 = r'./modelsave_balanced_dataset/mfcc+wav2vec_tccnn.pth'  # mfcc+openl_tccnn模型
    # save_path1 = r'./modelsave_balanced_dataset/mfcc_gfcc_tccnn.pth'  # mfcc+gfcc+双通道goolglenet模型
    save_path1 = r'./modelsave_balanced_dataset/mfcc_openl_tccnn_e.pth'  # mfcc+openl+双通道goolglenet模型
    # save_path1 = r'./modelsave_balanced_dataset/wav2vec2_googlenet.pth'  #wav2vec2+goolglenet模型
    save_path = r'./modelsave_balanced_dataset'  # 使用早停
    early_stopping = EarlyStopping(save_path,patience=10)
    if len(X2_test)>0:
        # model = fusion_resnet18(num_classes=4)
        # model = ch2_model_noatt()#可用于mfcc/gfcc+openl的组合
        # model = ch2_mfcc_wav2vec()
        # model = ch2_gfcc_wav2vec()
        # model = fusion_googlenet_resnet18()#用于mfcc/gfcc+openl
        # model = fusion_googlenet_googlenet()  # 用于mfcc、gfcc、wav2vec两两组合
        model = TCCNN_E()
        # model = TCCNN_SE()
        # model = fusion_googlenet_googlenet_openl()  # 用于mfcc、openl两两组合
        # model = fusion_googlenet_googlenet_openl_EMA2()
        # model = ResNet18()
        # model = GoogleNet(num_classes=4, aux_logits=True, init_weights=True)  # 输入wav2vec
        # model = GoogleNet_openl(num_classes=4, aux_logits=True, init_weights=True)  # 注意，该模型仅用于输入（1，6,512）维度的openl特征
    else:
        # model = ResNet18(num_classes=4) #resnet模型
        # model = ResNet50()
        # model = ResNext50()
        # model = parallel_all_you_want(4) #parallel_all_you_want
        model = GoogleNet(num_classes=4, aux_logits=True, init_weights=True) #注意，该模型返回三个loss值，train函数需要修改loss
        # model = GoogleNet_openl(num_classes=4, aux_logits=True, init_weights=True) #注意，该模型仅用于输入（1，6,512）维度的openl特征
        # model = GoogleNet_EMA(num_classes=4, aux_logits=True, init_weights=True)
        # model = GoogleNet_RGA(num_classes=4, aux_logits=True, init_weights=True)
        # model = Xception(num_classes=4)
        # model = effnetv2_s(num_classes=4)
        # model = effnetv2_m(num_classes=4)
        # model = TransformerModel(input_size=64 , output_size= 4,hidden_size=64)
    # model.load_state_dict(torch.load(load_path))
    model.to("cuda")
    criterion = nn.CrossEntropyLoss() # 定义自己的需要的损失函数，此处为交叉熵损失函数
    num_epochs= 1000
    if len(X2_train)>0:
        dataset = TensorDataset(
            torch.tensor(X1_train, dtype=torch.float32),
            torch.tensor(X2_train, dtype=torch.float32),
            torch.tensor(y_train))
        testdataset = TensorDataset(
            torch.tensor(X1_test, dtype=torch.float32),
            torch.tensor(X2_test, dtype=torch.float32),
            torch.tensor(y_test))
    else:
        dataset = TensorDataset(
            torch.tensor(X1_train, dtype=torch.float32),
            torch.tensor(y_train))
        testdataset = TensorDataset(
            torch.tensor(X1_test, dtype=torch.float32),
            torch.tensor(y_test))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    testdataloader = DataLoader(testdataset, batch_size=64, shuffle=True)

    #train model
    train_and_test(model,dataLoader=dataloader,testdataLoader=testdataloader,lossFunc=criterion,n_epoch=num_epochs,device=device)
    torch.save(model.state_dict(), save_path1)
#将列表合并为一个字典
data = {
    'Train Loss': train_loss_graph,
    'Train Accuracy': train_acc_graph,
    'Test Loss': test_loss_graph,
    'Test Accuracy': test_acc_graph
}




