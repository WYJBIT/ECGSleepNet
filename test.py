import torch
import numpy as np
import data_prepare
import model
import utils
import matplotlib.pyplot as plt

if __name__ == '__main__':
    classes = 3  # 选择五分类还是三分类

    # 读入预处理好的数据并划分测试集
    # 此处train和test文件使用相同的seed=10，以保证打乱后的顺序是相同的
    ECG_data, labels = data_prepare.data_process(classes=classes, seed=10)
    ECG_data_test = ECG_data[int(len(ECG_data) * 0.7):, :]
    labels_test = labels[int(len(labels) * 0.7):, :]

    # 实例化网络
    net = model.ResnetBlock(block=model.BasicBlock, block_stride=[1, 2, 2, 2, 2, 2, 2, 2],
                            in_channel=1, out_channel=32, classes=classes)

    # 加载已完成训练的模型参数
    if classes == 3:
        net.load_state_dict(torch.load("./trained_model_3/100.pt"))
    else:
        net.load_state_dict(torch.load("./trained_model_5/100.pt"))
    net = net.cuda()

    # 将测试集输入到神经网络
    with torch.no_grad():
        test_input = torch.tensor(ECG_data_test.reshape(len(ECG_data_test), 1, 3750),
                               dtype=torch.float32).cuda()
        test_labels = torch.tensor(labels_test.reshape(len(labels_test)),
                                dtype=torch.float32).cuda().to(dtype=torch.long)
        test_output = net(test_input)
    test_output = np.array(test_output.cpu())

    if classes == 3:
        confusion_matrix = np.zeros((3, 3))
    else:
        confusion_matrix = np.zeros((5, 5))

    # 计算网络输出的混淆矩阵
    for i in range(len(test_output)):
        confusion_matrix[int(labels_test[i])][int(np.argmax(test_output[i]))] += 1

    # 计算准确率、精确率、召回率和F1指标
    precision, recall = 0, 0
    for i in range(classes):
        if sum(confusion_matrix[:, i]) != 0:
            precision += confusion_matrix[i, i] / sum(confusion_matrix[:, i])
        if sum(confusion_matrix[i, :]) != 0:
            recall += confusion_matrix[i, i] / sum(confusion_matrix[i, :])
    precision /= classes
    recall /= classes
    F1 = 2 * precision * recall / (precision + recall)

    print("混淆矩阵为:\n",confusion_matrix)
    print("准确率为：", sum([confusion_matrix[i, i] for i in range(classes)]) / len(test_output))
    print("精确率为：", precision)
    print("召回率为：", recall)
    print("F1分数为：", F1)
