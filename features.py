import numpy as np
import csv
from scipy.fft import fftfreq, fft
import pywt
import matplotlib.pyplot as plt
import copy

def wavelet_denoising(signal):
    # 使用小波变化对数据进行去噪
    wavelet_basis = 'db8'  # 此处选择db8小波基
    w = pywt.Wavelet(wavelet_basis)
    maxlevel = pywt.dwt_max_level(len(signal), w.dec_len)  # 根据数据长度计算分解层数
    threshold = 0.1
    coeffs = pywt.wavedec(signal, wavelet_basis, level=maxlevel)  # 将信号进行小波分解
    # 使用阈值滤除噪声
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
    # 重建信号
    signalrec = pywt.waverec(coeffs, wavelet_basis)
    return signalrec

def median_filter(data, width=100):
    # 中值滤波算法
    new_data = np.zeros((len(data)))
    new_data[0: width] = data[0:width]
    for i in range(width, len(data)):
        tmp = copy.deepcopy(data[i - width:i])
        tmp.sort()
        if width % 2 == 0:
            med = (tmp[int(width / 2 - 1)] + tmp[int(width / 2)]) / 2
        else:
            med = tmp[int((width - 1) / 2)]
        new_data[i - width:i] = copy.deepcopy(data[i - width:i]) - med
    return new_data

# 读取原始信号数据
file = open('ECG.txt', encoding='utf-8')
data = file.readlines()
file.close()

# 对原始信号进行去噪和滤波
for i in range(len(data)):
    data[i] = eval(data[i])
data = np.array(data)
data = wavelet_denoising(data)
data = median_filter(data).reshape((1084, 3750))
feature_matrix = np.zeros((1084, 4))  # 记录特征值

for i in range(1084):
    feature_matrix[i][0] = data[i].mean()  # 计算时域均值

    data_fft = np.abs(fft(data[i])[: 1084 // 2])
    data_freq = fftfreq(1084, 1 / 125)[: 1084 // 2]
    data_power = data_fft ** 2 / 2

    index = np.argmax(data_power)
    feature_matrix[i][1] = data_freq[index]  # 计算峰值频率
    feature_matrix[i][2] = np.array([data_power[i] * data_freq[i]
                                    for i in range(len(data_power))]).sum() / data_power.sum()  # 计算重心频率
    feature_matrix[i][3] = np.sqrt(np.array([data_power[i] * (data_freq[i] - feature_matrix[i][2]) ** 2
                                          for i in range(len(data_power))]).sum() / data_power.sum())  # 频率标准差

# 将特征数据归一化
for j in range(4):
    std, mean = feature_matrix[:, j].std(), feature_matrix[:, j].mean()
    feature_matrix[:, j] = (feature_matrix[:, j] - mean) / std

# 将特征写入csv文件
with open('ECG_feature.csv', 'w', newline='', encoding='utf-8') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(feature_matrix)

