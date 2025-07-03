"此文件用于统计去静音后的音频片段的时间统计量,以及波形图"

import os
import shutil
import librosa
from utils import create_datafile
import numpy as np
from sklearn.model_selection import train_test_split


# path= r"./datasetSegAug/copd/Seg0_cough_1dxqFcveTM_1607984976893.wav"
# w,s = librosa.load(path,sr=16000)
# t = len(w)/s
# print(t)
# os._exit(0)
# al_path = r'./datasetSegment'
# al_path1 = ['asthma', 'copd', 'covid', 'healthy']
# ts = []
# ts_02 = []
# ts_03 = []
# ts_04 = []
# ts_05 = []
# ts_1 = []
# ts_2 = []
# ts_3 = []
# ts_4 = []
# ts_5 = []
# ts_6 = []
# ts_7 = []
# ts_8 = []
# ts_9 = []
# for path1 in al_path1:
#     min_num = 0
#     wav_list = os.listdir(r'./{}/{}'.format(al_path, path1))
#     for wav in wav_list:
#         wave_path = r'./{}/{}/{}'.format(al_path,path1,wav)
#         w, sr = librosa.load(wave_path, sr=16000)
#         t = len(w)/sr
#         # if t >0.4 and t<5:
#         #     ts.append(t)
#         ts.append(t)
#         if t < 0.2:
#             ts_02.append(t)
#         if t < 0.3:
#             ts_03.append(t)
#         if t < 0.4:
#             ts_04.append(t)
#         if t < 0.5:
#             ts_05.append(t)
#         if t > 1:
#             ts_1.append(t)
#         if t > 2:
#             ts_2.append(t)
#         if t > 3:
#             ts_3.append(t)
#         if t > 4:
#             ts_4.append(t)
#         if t > 5:
#             # print("大于5秒的音频：\n", path1, wav)
#             ts_5.append(t)
#         if t > 6:
#             # print("大于6秒的音频：\n", path1, wav)
#             ts_6.append(t)
#         if t > 7:
#             # print("大于7秒的音频：\n", path1, wav)
#             ts_7.append(t)
#         if t > 8:
#             print("大于8秒的音频：\n", path1, wav)
#             ts_8.append(t)
#         if t > 9:
#             print("大于9秒的音频：\n",path1,wav)
#             ts_9.append(t)
# print('*'*20)
# print(len(ts))
# print("ts_02",len(ts_02))
# print("ts_03",len(ts_03))
# print("ts_04",len(ts_04))
# print("ts_05",len(ts_05))
# print("ts_1",len(ts_1))
# print("ts_2",len(ts_2))
# print("ts_3",len(ts_3))
# print("ts_4",len(ts_4))
# print("ts_5",len(ts_5))
# print("ts_6",len(ts_6))
# print("ts_7",len(ts_7))
# print("ts_8",len(ts_8))
# print("ts_9",len(ts_9))
#
# ts=np.array(ts)
# print("avg_time:",np.average(ts))
# print("max_time:", np.max(ts))
# print("min_time:", np.min(ts))
#



##用于可视化音频
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence

SR = 16000  # sample rate
path = r'./dataset/covid/cough_F7woupPn6n_1589583301325.wav'
soundwav = AudioSegment.from_file(path,format='wav')
# wav,s = librosa.load(path,sr= SR)
# 使用split_on_silence()分割音频，它将音频分割为多个部分，静音部分和非静音部分
chunks = split_on_silence(soundwav,
                              # must be silent for at least half a second,沉默半秒
                              min_silence_len=400,
                              # consider it silent if quieter than -16 dBFS
                              silence_thresh=-40,
                              keep_silence=100
                              )
print(len(chunks))
# 创建一个新的音频对象用于存储拼接后的结果
output = AudioSegment.silent(duration=0)
# 将非静音部分拼接在一起
for chunk in chunks:
    output += chunk
chunk_data = output.get_array_of_samples()
wav_data=soundwav.get_array_of_samples()   #这是array类型的数据
max_abs1 = np.max(np.abs(wav_data))
normalized_array1 = wav_data / max_abs1
max_abs2 = np.max(np.abs(chunk_data))
normalized_array2 = chunk_data / max_abs2
print(len(wav_data)/16000)
t1 = np.arange(0, len(normalized_array1)) * (1.0 / SR)
plt.subplot(211)
plt.plot(t1, normalized_array)
plt.title("covid cough without split")
plt.xlabel("Time(s)")
plt.ylabel("normalized amplitude")
#
# # 绘制每个chunk的波形图
plt.subplot(212)
# chunk_data=chunks[0].get_array_of_samples()
# print(len(chunk_data)/16000)
t2 = np.arange(0, len(normalized_array2)) * (1.0 / SR)
plt.plot(t2, normalized_array2)
plt.title("covid cough with split")
plt.xlabel("Time(s)")
plt.ylabel("normalized amplitude")
plt.show()