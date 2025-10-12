import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmod, softmax


# 读取数据
def get_data():
    # 从文件加载数据集
    data = pd.read_csv('../data/train.csv')
    # 划分数据集
    X = data.drop('label', axis=1)
    Y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 特征转换
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test


def init_network():
    network_s = joblib.load("../data/nn_sample")
    return network_s


def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 逐层进行计算传递
    a1 = np.dot(x, w1) + b1
    z1 = sigmod(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmod(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


#----- 主流程
# 1. 获取数据
x, y = get_data()
# 2. 创建模型（加载参数）
network = init_network()


batch_size = 100
accuracy_cnt = 0
n = x.shape[0]

# 循环迭代：分批次测试（前向传播），兵累积预测准确个数
for i in range(0, n, batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = forward(network, x_batch)
    y_pred = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(y_pred == y[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / n))
