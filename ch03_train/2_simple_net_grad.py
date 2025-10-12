import numpy as np

from common.functions import softmax, cross_entropy_error

from common.gradient import numerical_gradient


# 定义一个简单的神经网络类
class SimpleNet:
    def __init__(self):
        # 基于正态分布生成随机参数
        self.W = np.random.randn(2, 3)

    def forward(self, X):
        a = X @ self.W
        y = softmax(a)

        return y

    # 计算损失值
    def loss(self, x, t):
        y_hat = self.forward(x)
        loss = cross_entropy_error(y_hat, t)

        return loss

if __name__ == "__main__":
    # 1. 定义数据
    x = np.array([[0.6, 0.9]])
    t = np.array([0, 0, 1])
    # 2. 定义神经网络模型
    net = SimpleNet()
    # 3. 计算梯度
    f = lambda _: net.loss(x, t)
    grad_w = numerical_gradient(f, net.W)
    print(grad_w)
