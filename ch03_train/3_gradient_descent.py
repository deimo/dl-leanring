import numpy as np
import matplotlib.pyplot as plt

from common.gradient import numerical_gradient


# 定义梯度下降法的函数
def gradient_descent(f, init_x, lr=0.01, num_iters=100):
    x = init_x
    # 定义列表保存x的变化
    x_history = []
    # 循环迭代
    for i in range(num_iters):
        x_history.append(x.copy())
        # 计算梯度
        grad = numerical_gradient(f, x)
        # 更新参数
        x = x - lr * grad

    return x, np.array(x_history)

# 定义目标函数:
def f(x):
    return x[0] ** 2 + x[1] ** 2

if __name__ == "__main__":
    # 初始化
    init_x = np.array([-3.0, 4.0])
    # 超参数
    lr = 0.1
    num_iters = 50
    # 使用梯度下降法计算最小值点
    x, x_history = gradient_descent(f, init_x, lr, num_iters)
    print("x:", x)
    print("f(x):", f(x))

    plt.plot([-5, 5], [0, 0], "--b")
    plt.plot([0, 0], [-5, 5], "--b")
    plt.scatter(x_history[:, 0], x_history[:, 1], s=10)
    plt.xlim([-3.5, 3.5])
    plt.ylim([-4.5, 4.5])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

