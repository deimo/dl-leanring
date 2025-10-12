import numpy as np
from common.functions import sigmod, identity


def init_network():
    network = {}
    # 第一层
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # 第二层
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # 第三层
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, X):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 逐层进行计算传递
    a1 = np.dot(X, w1) + b1
    z1 = sigmod(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmod(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity(a3)

    return y

if __name__ == '__main__':
    network = init_network()
    X = np.array(np.random.rand(2, ))
    print(X)
    print(X.shape)
    Y = forward(network, X)
    print(Y)
    print("--------------")
    X2 = np.array([1.0, 0.5])
    print(X2)
    print(X2.shape)
    Y2 = forward(network, X2)
    print(Y2)