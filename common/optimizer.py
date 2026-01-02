# 随机梯度下降SGD
import numpy as np


class SGD:
    # 初始化
    def __init__(self, lr=0.01):
        self.lr = lr

    # 参数更新
    def update(self, params: dict, grads: dict):
        # 遍历传入的所有参数，按公式更新
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# 动量法
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: dict, grads: dict):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 按照公式，进行参数更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params: dict, grads: dict):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, decay=0.99):
        self.lr = lr
        self.decay = decay
        self.h = None

    def update(self, params: dict, grads: dict):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] *= self.decay
            self.h[key] += (1 - self.decay) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)


class Adam:
    def __init__(self, lr=0.001, alpha1=0.9, alpha2=0.999):
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.v = None
        self.h = None
        self.t = 0

    def update(self, params: dict, grads: dict):
        # 初始化v和h
        if self.v is None:
            self.v = {}
            self.h = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.t += 1
        # 按照当前的迭代次数修改学习率
        lr_t = self.lr * np.sqrt(1 - self.alpha2 ** self.t) / (1 - self.alpha1 ** self.t)
        for key in params.keys():
            # self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            # self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * grads[key] * grads[key]
            self.v[key] += (1 - self.alpha1) * (grads[key] - self.v[key])
            self.h[key] += (1 - self.alpha2) * (grads[key] ** 2 - self.h[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-7)