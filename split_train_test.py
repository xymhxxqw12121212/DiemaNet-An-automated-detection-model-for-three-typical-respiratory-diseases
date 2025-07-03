#此文件用于划分训练测试集并创建train_test_data文件夹

import os
import shutil
from utils import create_datafile
import numpy as np
from sklearn.model_selection import train_test_split

al_path = r'./datasetSegAug'
al_path1 = ['asthma','copd','covid','healthy']
dic= {'healthy': 0, 'covid':1, 'asthma':2, 'copd':3}

for path1 in al_path1:
    create_datafile(r'./train_test_data/train/{}'.format(path1))
    create_datafile(r'./train_test_data/test/{}'.format(path1))
    wav_list = os.listdir(r'./{}/{}'.format(al_path,path1))
    label_list = np.ones(len(wav_list)) * dic[path1]
    X_train, X_test, y_train, y_test = train_test_split(wav_list, label_list, test_size=0.2,random_state=0)
    print("{} len X_train:".format(path1),len(X_train))
    print("{} len X_test:".format(path1), len(X_test))
    for x in X_train:
        wave_path = r'./{}/{}/{}'.format(al_path,path1,x)
        save_path = r'./train_test_data/train/{}/{}'.format(path1,x)
        shutil.copy(wave_path, save_path)
    for x in X_test:
        wave_path = r'./{}/{}/{}'.format(al_path,path1,x)
        save_path = r'./train_test_data/test/{}/{}'.format(path1,x)
        shutil.copy(wave_path, save_path)
