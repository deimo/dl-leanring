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
# 获取数据
x, y = get_data()
# print(x.shape)
# print(y.shape)
# 创建模型（加载参数）
network = init_network()
# 前向传播
y_preba = forward(network, x)
# print(y_preba.shape)

# 将分类概率转换为分类标签
y_pre_label = np.argmax(y_preba, axis=1)
print(y_pre_label.shape)

# 计算分类准确率
accuracy = np.sum(y_pre_label == y) / len(y)
print('x.shape[0]: ', x.shape[0])
print('len(y): ', len(y))
print(accuracy)
