import numpy as np
import matplotlib.pyplot as plt

from two_layer_net import  TwoLayerNet
from common.load_data import get_data

# 加载数据
x_train, x_test, t_train, t_test = get_data()
# x_train = x_train[0: 500]
# x_test = x_test[-100:]
# t_train = t_train[0: 500]
# t_test = t_test[-100:]

# 创建模型
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置超参数
lr = 0.5
batch_size = 100
num_epochs = 10

# 计算迭代次数
train_size = x_train.shape[0]
iter_per_epoch = np.ceil(train_size / batch_size)
iters_num = int(num_epochs * iter_per_epoch)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 开始训练：梯度下降，循环迭代
for i in range(iters_num):
    # 随机选取批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    print("grad: ========= ", i)
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= lr * grad[key]
    # 计算并保存当前的训练误差
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 每完成一个epoch，就计算并保存训练和准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('Epoch: {}, Loss: {}, Train Acc: {}, Test Acc: {}'.format(i, loss, train_acc, test_acc))

# 绘图
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='best')
plt.show()
