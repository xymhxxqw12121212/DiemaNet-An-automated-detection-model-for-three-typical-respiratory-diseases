"此文件用于提取train_test_data中音频的音频特征"
import os
import json
import librosa
import openl3
import numpy as np
import warnings
from utils import *
# from transformers import Wav2Vec2Model , Wav2Vec2Processor
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

sample_rate=16000
window_size = int(sample_rate * 25 // 1000 + 1)
window_stride = int(sample_rate * 6 // 1000)
#加载openl特征提取网络
model_openl3 = openl3.models.load_audio_embedding_model(input_repr="linear", content_type="env", embedding_size=512)
#加在wav2vec2特征提取网络
# wav2vec_path = "./wav2vec_model"
# processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
# model_wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_path)    # 用于提取通用特征，768维
dic= {'healthy': 0, 'covid':1, 'asthma':2, 'copd':3}
def extfeature(audio_path, save_path, ill, feature= 'mfcc' , win_length= 16000 ):
    #feature=['mfcc']或['mfcc',[openl]]
    ts= []
    feas1_dict = {}
    feas2_dict = {}
    feas3_dict = {}
    feas4_dict = {}
    lab_dict = {}
    audios= os.listdir(audio_path)
    num = 0
    for audio in audios:
        x, sr= librosa.load(os.path.join(audio_path, audio), sample_rate)
        t= len(x)/sr
        ts.append(t)
        x = librosa.util.fix_length(x, win_length)
        #标签
        lab_dict[audio] = dic[ill]
        np.savez(save_path + "//" + ill + "_" + 'lables', **lab_dict)
        #mfcc
        if feature== 'mfcc':
            fea = librosa.feature.mfcc(x, n_mfcc=64, n_fft=window_size, hop_length=window_stride).T  # (100.64)
            feas1_dict[audio] = fea
            np.savez(save_path + "//" + ill + "_" + 'mfcc', **feas1_dict)
        if feature == 'openl':
            emb, _ = openl3.get_audio_embedding(x, 16000, model=model_openl3, verbose=1, center=True,hop_size=0.06)  # (6,512)
            feas2_dict[audio] = emb
            np.savez(save_path + "//" + ill + "_" + 'openl3', **feas2_dict)
        #wav2vec
        if feature== 'wav2vec':
            input_values = processor(x, sampling_rate=16000, return_tensors="pt").input_values
            wav2vec = model_wav2vec(input_values)['last_hidden_state']  # [1,124,768]
            wav2vec = wav2vec.detach().numpy()
            wav2vec = wav2vec.squeeze(0)
            feas3_dict[audio] = wav2vec
            np.savez(save_path + "//" + ill + "_" + 'wav2vec', **feas3_dict)
        #gfcc
        if feature == 'gfcc':#(98,64)
            gfccs = gfcc(x, fs=16000, num_ceps=64, nfilts=128)
            feas4_dict[audio] = gfccs
            np.savez(save_path + "//" + ill + "_" + 'gfcc', **feas4_dict)
            num = num + 1
            print("extract num:",num)





#提取训练集特征
#创建一个存储特征的文件夹
create_datafile(r"/root/autodl-tmp/feature1s-6ms/train")
path1= r'/root/autodl-tmp/train_test_data/train'#autodl中的数据盘tmp的路径，可自行更改
ills1= os.listdir(path1)
for ill in ills1:
    audio_path= os.path.join(path1, ill)
    save_path=r"/root/autodl-tmp/feature1s-6ms/train"  #autodl中的数据盘tmp的路径，可自行更改
    extfeature(audio_path, save_path, ill, feature='mfcc', win_length=16000)
    extfeature(audio_path, save_path, ill, feature='openl', win_length=16000)
    # extfeature(audio_path, save_path, ill, feature='wav2vec', win_length=16000)
    # extfeature(audio_path, save_path, ill, feature='gfcc',win_length=16000)

#提取测试集特征
#创建一个存储特征的文件夹
create_datafile(r"/root/autodl-tmp/feature1s-6ms/test")
path2= r'/root/autodl-tmp/train_test_data/test'
ills2= os.listdir(path2)
for ill in ills2:
    audio_path= os.path.join(path2, ill)
    save_path=r"/root/autodl-tmp/feature1s-6ms/test"
    extfeature(audio_path, save_path, ill, feature='mfcc', win_length=16000)
    extfeature(audio_path, save_path, ill, feature='openl', win_length=16000)
    # extfeature(audio_path, save_path, ill, feature='wav2vec', win_length=16000)
    # extfeature(audio_path, save_path, ill, feature='gfcc', win_length=16000)

#读取npz文件
# a=np.load(r'./feature/test/asthma_mfcc.npz')
# keys=a.files
# print(len(keys))
# for key in keys: print(key,a[key].shape)



