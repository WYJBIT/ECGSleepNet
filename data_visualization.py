import matplotlib.pyplot as plt

# 读取文件并转换数据格式
file1 = open('ECG.txt', encoding='utf-8')
data1 = file1.readlines()
file1.close()
for i in range(len(data1)):
    data1[i] = eval(data1[i])

# 数据可视化
plt.plot(data1, color='blue', label='ECG_data')
plt.legend()
plt.savefig("./figs/ECG_visualization.png", dpi=300)
plt.show()
