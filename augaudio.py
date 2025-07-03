"此文件用于将datasetSegment目录中的音频文件中符合时间长度条件的音频复制到dateseAugment中，并将其中COPD进行4倍扩充"

import os
import sys
import librosa
import shutil
import librosa.display
import random
from tqdm import tqdm
from scipy.io.wavfile import write
import numpy as np
from utils import create_datafile
from audiomentations import Compose, TimeStretch, PitchShift, Shift, Trim, Gain, PolarityInversion,AddGaussianNoise,AddGaussianSNR,AddShortNoises
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
create_datafile(r'./datasetSegAug/asthma')
create_datafile(r'./datasetSegAug/copd')
create_datafile(r'./datasetSegAug/covid')
create_datafile(r'./datasetSegAug/healthy')


al_path = r'./datasetSegment'
al_path1 = ['asthma', 'copd', 'covid', 'healthy']
SR = 16000
for path1 in al_path1:
    min_num = 0
    wav_list = os.listdir(r'./{}/{}'.format(al_path, path1))
    for wav in wav_list:
        wave_path = r'./{}/{}/{}'.format(al_path,path1,wav)
        save_path = r'./datasetSegAug/{}/{}'.format(path1,wav)
        w, s = librosa.load(wave_path, sr=SR)
        t = len(w)/s
        if t > 0.5 and t < 5:
            shutil.copy(wave_path, save_path)
# augment1 = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,p=0.5)])
# augment2 = Compose([TimeStretch(min_rate=0.8, max_rate=1.06,p=0.5)])
# augment3 = Compose([TimeStretch(min_rate=0.94, max_rate=1.24,p=0.5)])
# augment4 = Compose([PitchShift(min_semitones=-1, max_semitones=1,p=0.5)])
# augment5 = Compose([PitchShift(min_semitones=-2, max_semitones=2,p=0.5)])
# augment6 = Compose([PitchShift(min_semitones=-2.5, max_semitones=2.5,p=0.5)])
# augment7 = Compose([PitchShift(min_semitones=-3.5, max_semitones=3.5,p=0.5)])

augment1 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.06,p=0.5),
    PitchShift(min_semitones=-1, max_semitones=1,p=0.5),
    Gain(p=0.5)
])
augment2 = Compose([
    AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=35, p=0.5),
    TimeStretch(min_rate=0.94, max_rate=1.24,p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2,p=0.5),
    Gain(p=0.5)
])
augment3 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.06,p=0.5),
    PitchShift(min_semitones=-2.5, max_semitones=2.5,p=0.5),
    Gain(p=0.5)
])
augment4 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.06,p=0.5),
    PitchShift(min_semitones=-3.5, max_semitones=3.5,p=0.5),
    Gain(p=0.5)
])
augment5 = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,p=0.5),
    TimeStretch(min_rate=0.94, max_rate=1.24,p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2,p=0.5),
    Gain(p=0.5)
])
augment6 = Compose([
    AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=35, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.06,p=0.5),
    PitchShift(min_semitones=-3, max_semitones=3,p=0.5),
    Gain(p=0.5)
])
augment7 = Compose([
    AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=35, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.06,p=0.5),
    PitchShift(min_semitones=-3.5, max_semitones=3.5,p=0.5),
    Gain(p=0.5)
])
#对copd增强,增强20次
dataset_path=r'./datasetSegment/copd'
w_path=r'./datasetSegAug/copd'
dirnames=os.listdir(dataset_path)
for dir in tqdm(dirnames):
    name = dir.split(".")[0]
    dir_path = os.path.join(dataset_path, dir)
    data, _ = librosa.load(dir_path, SR)
    t = len(data) / SR
    if t > 0.5 and t < 5:
        for i in range(5):
            np.random.seed(i)
            data_aug1 = augment1(data, SR)
            data_aug2 = augment2(data, SR)
            data_aug3 = augment3(data, SR)
            data_aug4 = augment4(data, SR)
            augments = [data_aug1, data_aug2, data_aug3, data_aug4]
            for j in range(len(augments)):
                write(w_path + '\\' + name + '_aug' + str(i*4+j) + '.wav', 16000, augments[j])

#对covid增强,增强5次
dataset_path=r'./datasetSegment/covid'
w_path=r'./datasetSegAug/covid'
dirnames=os.listdir(dataset_path)
for dir in tqdm(dirnames):
    name = dir.split(".")[0]
    dir_path = os.path.join(dataset_path, dir)
    data, _ = librosa.load(dir_path, SR)
    t = len(data) / SR
    if t > 0.5 and t < 5:
        np.random.seed(11)
        data_aug1 = augment1(data, SR)
        data_aug2 = augment2(data, SR)
        data_aug3 = augment3(data, SR)
        data_aug4 = augment4(data, SR)
        data_aug5 = augment5(data, SR)
        augments = [data_aug1, data_aug2, data_aug3, data_aug4, data_aug5]
        for j in range(len(augments)):
            write(w_path + '\\' + name + '_aug' + str(j) + '.wav', 16000, augments[j])

#对哮喘增强,增强5次
dataset_path=r'./datasetSegment/asthma'
w_path=r'./datasetSegAug/asthma'
dirnames=os.listdir(dataset_path)
for dir in tqdm(dirnames):
    name = dir.split(".")[0]
    dir_path = os.path.join(dataset_path, dir)
    data, _ = librosa.load(dir_path, SR)
    t = len(data) / SR
    if t > 0.5 and t < 5:
        np.random.seed(11)
        data_aug1 = augment1(data, SR)
        data_aug2 = augment2(data, SR)
        data_aug3 = augment3(data, SR)
        data_aug4 = augment4(data, SR)
        data_aug5 = augment5(data, SR)
        augments = [data_aug1, data_aug2, data_aug3, data_aug4, data_aug5]
        for j in range(len(augments)):
            write(w_path + '\\' + name + '_aug' + str(j) + '.wav', 16000, augments[j])