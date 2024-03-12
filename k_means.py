import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


if __name__ == '__main__':
    # 读取csv中的特征
    ECG_features_og = np.genfromtxt("ECG_feature.csv", delimiter=' ', dtype=str)
    file = open("label_profusion.txt", encoding='utf-8')
    ECG_label = file.readlines()
    file.close()
    ECG_features = []
    for i in range(len(ECG_label)):
        tmp_list = ECG_features_og[i].split(',')
        new_list = [eval(item) for item in tmp_list]
        ECG_features.append(new_list)
        ECG_label[i] = eval(ECG_label[i])
    X = np.array(ECG_features)

    distortions = []
    sil_score = []
    result = None
    K = range(2, 15)  # 设置K值的范围
    X = np.append(X[:, 1].reshape(1084, 1), X[:, 2].reshape(1084, 1), axis=1)  # 选取两个特征进行聚类
    for k in K:
        # 分别构建各种K值下的聚类器
        model6 = KMeans(n_clusters=k).fit(X)
        y_pred = model6.labels_
        # 计算各个样本到其所在簇类中心欧式距离
        distortions.append(sum(np.min(cdist(X, model6.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        # 计算轮廓系数
        sil_score.append(silhouette_score(X, y_pred))
        if k == 3:
            result = y_pred

    color_list = ['red', 'blue', 'green', 'yellow', 'brown']
    # 绘制图形
    plt.subplot(221)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('optimal K')
    plt.ylabel('SSE')
    plt.subplot(222)
    plt.plot(K, sil_score, 'bx-', color='orange')
    plt.xlabel('optimal K')
    plt.ylabel('silhouette_score')
    plt.subplot(223)
    plt.scatter(X[:, 0], X[:, 1], marker='o', color='blue')
    plt.xlabel('Gravity frequency')
    plt.ylabel('Frequency standard deviation')
    plt.subplot(224)
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], marker='o', color=color_list[int(result[i])])
    plt.xlabel('Gravity frequency')
    plt.ylabel('Frequency standard deviation')
    plt.tight_layout()
    plt.savefig("./figs/k_means.png", dpi=500)
    plt.show()
