import numpy as np
from common.functions import sigmod, softmax, cross_entropy_error


class Relu:
    def __init__(self):
        # 记录<=0的x
        self.mask = None
        pass

    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Sigmoid:
    def __init__(self):
        # 定义内部属性，记录输出值y，用于反向传播时计算梯度
        self.y = None

    def forward(self, x):
        y = sigmod(x)
        self.y = y
        return y

    def backward(self, dy):
        dx = dy * self.y * (1 - self.y)

        return dx


class Affine:
    # 初始化
    def __init__(self, W, b):
        self.W = W
        self.b = b
        # 对输入数据X做一个保存，方便反向传播
        self.X = None
        self.original_X_shape = None
        # 将权重和偏置参数的梯度保存成属性，方便梯度下降法计算
        self.dW = None
        self.db = None

    def forward(self, X):
        self.original_X_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = np.dot(X, self.W) + self.b
        return y

    def backward(self, dy):
        dX = np.dot(dy, self.W.T)
        dX = dX.reshape(*self.original_X_shape)
        self.dW = np.dot(self.X.T, dy)
        self.db = np.sum(dy, axis=0)

        return dX


# 输出层
class SoftmaxWithLoss:

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dy=1):
        n = self.t.shape[0]
        # 如果是独热编码标签
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 顺序编码的case
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
        return dx / n
