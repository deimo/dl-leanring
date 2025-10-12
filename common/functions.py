import numpy as np


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 损失函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 转换为顺序编码
    if t.size == y.size:
        t = t.argmax(axis=1)
    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n
