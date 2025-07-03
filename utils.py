import matplotlib.pyplot as plt
import librosa
# import openl3
import numpy as np
import os
def get_feature(X,feature='mfcc',window_size = int(16000 * 25 // 1000 + 1),window_stride = int(16000 * 10 // 1000)):
    if feature =='mfcc':
        MFCC = []
        for j in range(len(X)):
            mfcc = librosa.feature.mfcc(X[j], n_mfcc=64, n_fft=window_size, hop_length=window_stride)
            mfcc = mfcc.T
            MFCC.append(mfcc.tolist())
        X1 = np.array(MFCC)
        return X1


def Countalpha(label):
    n_num=0
    p_num=0
    for x in label:
        if x==0:
            n_num=n_num+1
        if x==1:
            p_num=p_num+1
    return n_num,p_num

def create_datafile (path):
    if not os.path.exists(path):
        os.makedirs(path)

#给数组增加一个维度
def reshape_numpy(X,transpose = False):
    X = np.array(X)
    if transpose:
      B, W, H = X.shape[0], X.shape[1], X.shape[2]
    else:
      B, W, H = X.shape[0], X.shape[2], X.shape[1]
    X = X.reshape(B, 1, H, W)
    return X
def plot_loss_acc(train_loss,test_loss,train_acc,test_acc):
    epoch=range(len(train_loss))
    plt.subplot(121)
    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('loss')
    plt.plot(epoch, train_loss,color='red')
    plt.plot(epoch, test_loss, label="test loss", color='blue')
    plt.legend(["train loss","test loss"])
    plt.subplot(122)
    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('acc')
    plt.plot(epoch, train_acc, label="train acc", color='red')
    plt.plot(epoch, test_acc, label="test acc", color='blue')
    plt.legend(["train acc", "test acc"])
    plt.savefig('train_test_loss_acc.png')
    plt.show()