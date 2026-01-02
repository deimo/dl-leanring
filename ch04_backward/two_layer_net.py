import numpy as np

from common.gradient import numerical_gradient
from common.functions import softmax, sigmoid, cross_entropy
from collections import OrderedDict

from common.layers import Affine, Relu, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weiht_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weiht_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weiht_init_std
        self.params['b2'] = np.zeros(output_size)
        # 定义层结构
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def forward(self, X):
        # 对于网络中的每一层依次调用forward方法
        for layer in self.layers.values():
            X = layer.forward(X)

        return X

    def loss(self, x, t):
        y = self.forward(x)
        loss_value = self.lastLayer.forward(y, t)

        return loss_value

    def accuracy(self, X, t):
        y_pred = self.forward(X)
        y = np.argmax(y_pred, axis=1)
        accuracy = np.sum(y == t) / float(X.shape[0])

        return accuracy

    # 使用数值微分计算梯度
    def numerical_gradient(self, X, t):
        # 定义目标函数
        loss_f = lambda _: self.loss(X, t)
        # 使用数值微分法进行梯度更新
        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = numerical_gradient(loss_f, self.params[key])

        return grads

    # 使用反向传播计算梯度
    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t)
        # 反向传播
        dy = self.lastLayer.backward()
        for layer in reversed(self.layers.values()):
            dy = layer.backward(dy)
        # 提取各层的参数梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
