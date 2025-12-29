import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



# 读取数据
def get_data():
    # 从文件加载数据集
    data = pd.read_csv('../data/train.csv')
    # 划分数据集
    X = data.drop('label', axis=1)
    print("X.shape: ", X.shape)
    Y = data['label']
    print("Y.shape: ", Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # 特征转换
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = y_train.values
    y_test = y_test.values

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x, y = get_data()
    # print(x.shape)
    # print(y.shape)