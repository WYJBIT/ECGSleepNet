import numpy as np
import pywt
import copy

def data_process(classes=5, seed=10):
    # 数据处理
    ECG_file = open("ECG.txt", encoding='utf-8')
    label_file = open("label_profusion.txt", encoding='utf-8')
    ECG_data = ECG_file.readlines()
    label_data = label_file.readlines()
    ECG_file.close()
    label_file.close()
    for i in range(len(ECG_data)):
        ECG_data[i] = eval(ECG_data[i])
    for i in range(len(label_data)):
        label_data[i] = eval(label_data[i])
    min_val, max_val = min(ECG_data), max(ECG_data)
    for i in range(len(ECG_data)):  # 数据标准化
        ECG_data[i] = (ECG_data[i] - min_val) / (max_val - min_val)
    ECG_data = np.array(ECG_data).reshape(1084, 3750).astype('float32')
    label_data = np.array(label_data).reshape(1084, 1).astype('float32')
    if classes == 3:
        for i in range(1084):
            if label_data[i] == 1 or label_data[i] == 2 or label_data[i] == 3:
                label_data[i] = 1
            elif label_data[i] == 4:
                label_data[i] = 2
    data = np.append(ECG_data, label_data, axis=-1)
    np.random.seed(seed)
    data = np.random.permutation(data)  # 采用相同的随机数种子将保证打乱顺序是一致的
    ECG_data = data[:, 0:3750]
    label_data = data[:, -1].reshape(1084, 1)

    return ECG_data, label_data

def data_augmentation(train_set, label, nums=4):
    # 基于滑动的数据增广方法
    new_train_set = copy.deepcopy(train_set)
    new_label = copy.deepcopy(label)
    data_num = len(train_set)
    for _ in range(nums):
        for i in range(1, data_num - 1):
            rand_num = int(3750 * np.random.randn() * 0.1)
            tmp_data = copy.deepcopy(train_set[i])
            tmp_data.reshape(3750)
            if rand_num < 0:
                # print(rand_num)
                tmp_data = tmp_data[0: 3750+rand_num]
                # print(tmp_data.shape)
                tmp_data = np.flip(tmp_data)
                tmp_data = np.append(tmp_data, train_set[i-1, rand_num:])
                # print(tmp_data.shape)
                tmp_data = np.flip(tmp_data)
            else:
                # print(rand_num)
                tmp_data = tmp_data[rand_num:]
                # print(tmp_data.shape)
                tmp_data = np.append(tmp_data, train_set[i+1, 0:rand_num])
                # print(tmp_data.shape)
            tmp_data = np.expand_dims(tmp_data, axis=0)
            tmp_label = copy.deepcopy(label[i])
            tmp_label = np.expand_dims(tmp_label, axis=0)
            new_train_set = np.append(new_train_set, tmp_data, axis=0)
            new_label = np.append(new_label, tmp_label, axis=0)
    return new_train_set, new_label


def wavelet_denoising(signal):
    # signal: list
    wavelet_basis = 'db8'  # 此处选择db8小波基
    w = pywt.Wavelet(wavelet_basis)
    maxlevel = pywt.dwt_max_level(len(signal), w.dec_len)  # 根据数据长度计算分解层数
    threshold = 0.1  # threshold for filtering
    coeffs = pywt.wavedec(signal, wavelet_basis, level=maxlevel)  # 将信号进行小波分解
    # 使用阈值滤除噪声
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
    # 重建信号
    signalrec = pywt.waverec(coeffs, wavelet_basis)
    return signalrec

def median_filter(data, width=100):
    new_data = np.zeros((len(data)))
    new_data[0: width] = data[0:width]
    for i in range(width, len(data)):
        tmp = copy.deepcopy(data[i - width:i])
        tmp.sort()
        if width % 2 == 0:
            med = (tmp[int(width / 2 - 1)] + tmp[int(width / 2)]) / 2
        else:
            med = tmp[int((width - 1) / 2)]
        new_data[i - width:i] = np.array(copy.deepcopy(data[i - width:i])) - med
    return new_data
