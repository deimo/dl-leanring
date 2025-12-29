import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient
from common.functions import softmax, sigmod, cross_entropy_error


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weiht_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weiht_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weiht_init_std
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(X, W1) + b1
        z1 = sigmod(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, X, t):
        y = self.forward(X)

        return cross_entropy_error(y, t)

    def accuracy(self, X, t):
        y_proba = self.forward(X)
        y = np.argmax(y_proba, axis=1) if y_proba.ndim == 2 else int(np.argmax(y_proba))
        # Support both one-hot (N, C) and class indices (N,)
        if t.ndim == 2:
            t_idx = np.argmax(t, axis=1)
        else:
            t_idx = t.astype(np.int64)
        accuracy = np.mean(y == t_idx)
        return accuracy

    def numerical_gradient(self, X, t):
        # 定义目标函数
        loss_f = lambda _: self.loss(X, t)
        # 使用数值微分法进行梯度更新
        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = numerical_gradient(loss_f, self.params[key])

        return grads
