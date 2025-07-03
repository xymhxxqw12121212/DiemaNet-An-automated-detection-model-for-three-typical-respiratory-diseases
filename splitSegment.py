"此文件用于去静音,分割成多个咳嗽片段"
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
from utils import create_datafile
import random
SR = 16000  # sample rate
al_path1 = r'./dataset'
al_path2 = ['asthma','copd','covid','healthy']

for path2 in al_path2:
        count = 0
        audio_list = os.listdir(r'./dataset/{}'.format(path2))
        for audio in audio_list:
            audio_path = r'./dataset/{}/{}'.format(path2,audio)
            soundwav = AudioSegment.from_file(audio_path,format='wav')
            # 使用split_on_silence()分割音频，它将音频分割为多个部分，静音部分和非静音部分
            chunks = split_on_silence(soundwav,
                              # must be silent for at least half a second,沉默半秒
                              min_silence_len=400,
                              # consider it silent if quieter than -16 dBFS
                              silence_thresh=-40,
                              keep_silence=200
                              )
            count += len(chunks)
            create_datafile(r'./datasetSegment/{}'.format(path2))
            if len(chunks) == 0:
                print(path2,audio)
            else:
                for i,chunk in enumerate(chunks):
                    chunk.export(r'./datasetSegment/{}/Seg{}_{}'.format(path2,i,audio),format='wav')
        print(path2,count)
