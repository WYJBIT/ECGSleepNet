import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(model, train_set, train_label, valid_set, valid_label, max_epochs=100, lr=1e-4,
          batch_size=32, l2_weight=1e-4, decay_step=10, classes=3):
    # 网络训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if classes == 3:
        class_weights = np.array([1, 1, 1.3]).reshape(3, 1)
    else:
        class_weights = np.array([1, 1.5, 1, 1, 1.2]).reshape(5, 1)
    class_weights = torch.from_numpy(class_weights).type(torch.FloatTensor)
    class_weights = class_weights.to('cuda')
    loss_fcn = nn.CrossEntropyLoss(weight=class_weights)

    model = model.cuda()
    training_loss_list = []
    validation_loss_list = []
    for epoch in tqdm(range(max_epochs)):
        training_loss = 0.0
        if ((epoch + 1) % decay_step) == 0:
            for param_group in optimizer.param_groups:
                ii = (epoch + 1) // decay_step
                param_group['lr'] = lr * 0.1 ** ii

        if ((epoch + 1) % 5) == 0:
            torch.save(model.state_dict(), "./trained_model_" + str(classes) + "./" + str(epoch+1) + ".pt")

        for i in range(len(train_set) // batch_size - 1):
            optimizer.zero_grad()
            input = torch.tensor(train_set[i * batch_size: (i + 1) * batch_size, :].reshape(batch_size, 1, 3750),
                                 dtype=torch.float32).cuda()
            labels = torch.tensor(train_label[i * batch_size: (i + 1) * batch_size, :].reshape(batch_size),
                                  dtype=torch.float32).cuda().to(dtype=torch.long)
            output = model(input)
            cnn_weights = [parm for name, parm in model.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = reg_loss * l2_weight

            loss = loss_fcn(output, labels)
            loss = loss + reg_loss  # 采用正则化
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            training_loss += loss.item()
        training_loss_list.append(training_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                # model.eval()
                valid_loss, sample_count = 0, 0
                for j in range(len(valid_set) // batch_size - 1):
                    v_input = torch.tensor(valid_set[j * batch_size: (j + 1) * batch_size, :].reshape(batch_size, 1, 3750),
                                         dtype=torch.float32).cuda()
                    v_labels = torch.tensor(valid_label[j * batch_size: (j + 1) * batch_size, :].reshape(batch_size),
                                          dtype=torch.float32).cuda().to(dtype=torch.long)
                    v_output = model(v_input)
                    v_loss = loss_fcn(v_output, v_labels)
                    cnn_weights = [parm for name, parm in model.named_parameters() if 'conv' in name]
                    v_reg_loss = 0
                    for p in cnn_weights:
                        v_reg_loss += torch.sum(p ** 2) / 2
                    v_reg_loss = v_reg_loss * l2_weight
                    v_loss = v_loss + v_reg_loss
                    valid_loss += v_loss.item()
                validation_loss_list.append(valid_loss)
    return training_loss_list, validation_loss_list

def plot_confusion_matrix():
    # 标签
    classes = ['0', '1', '2']
    classNamber =3 # 类别数量
    # 混淆矩阵
    confusion_matrix = np.array([
        (83, 18, 1),
        (5, 177, 14),
        (4, 8, 16)
    ], dtype=np.float64)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('confusion_matrix-3-unaug')  # 改图名
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)
    iters = np.reshape([[[i, j] for j in range(classNamber)] for i in range(classNamber)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]), va='center', ha='center')  # 显示对应的数字
    plt.ylabel('Ture')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('./figs/confusion_matrix-3-unaug.png',dpi=300)
    plt.show()

