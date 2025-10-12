import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff


def f(x):
    return 0.01 * x ** 2 + 0.1 * x


def tangent_line(f, x):
    y = f(x)
    # 计算x处切线的斜率
    a = numerical_diff(f, x)
    print("切线斜率：", a)
    b = y - a * x
    return lambda t: a * t + b

x = np.arange(0, 20.0, 0.1)
y = f(x)

f_line = tangent_line(f, x=5)
y_line = f_line(x)

# 原函数曲线
plt.plot(x, y)
# 切线
plt.plot(x, y_line)
plt.show()