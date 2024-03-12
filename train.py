import torch
import numpy as np
import data_prepare
import model
import utils
import matplotlib.pyplot as plt

if __name__ == '__main__':
    classes = 3  # 选择五分类还是三分类
    augmentation_flag = True
    # 读入预处理好的数据并划分训练集和验证集
    # 此处train和test文件使用相同的seed=10，以保证打乱后的顺序是相同的
    ECG_data, labels = data_prepare.data_process(classes=classes, seed=10)
    ECG_data_train = ECG_data[0: int(len(ECG_data) * 0.5), :]
    ECG_data_valid = ECG_data[int(len(ECG_data) * 0.5): int(len(ECG_data) * 0.7), :]
    labels_train = labels[0: int(len(labels) * 0.5), :]
    labels_valid = labels[int(len(labels) * 0.5): int(len(labels) * 0.7), :]

    # 数据增广
    if augmentation_flag:
        ECG_data_train, labels_train = data_prepare.data_augmentation(ECG_data_train, labels_train)

    # 实例化网络
    net = model.ResnetBlock(block=model.BasicBlock, block_stride=[1, 2, 2, 2, 2, 2, 2, 2],
                            in_channel=1, out_channel=32, classes=classes)

    # 训练网络
    train_list, valid_list = utils.train(net, ECG_data_train, labels_train,
                                         ECG_data_valid, labels_valid, classes=classes)

    # 绘制损失函数图并保存
    plt.plot(train_list, color='blue', label='training_loss')
    plt.plot([i * 5 for i in range(len(valid_list))], valid_list, color='orange', label='validation_loss')
    plt.legend()
    if not augmentation_flag:
        if classes == 3:
            plt.savefig("./figs/training_loss_3.png", dpi=300)
        else:
            plt.savefig("./figs/training_loss_5.png", dpi=300)
    else:
        if classes == 3:
            plt.savefig("./figs/training_loss_3_aug.png", dpi=300)
        else:
            plt.savefig("./figs/training_loss_5_aug.png", dpi=300)
    plt.show()