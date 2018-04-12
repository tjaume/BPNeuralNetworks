# encoding:utf-8

'''
获取数据集，包括训练集以及测试集
'''

import numpy as np
import os

data_file = 'IrisDataSet.csv'
def load_data():
    if not os.path.exists(data_file):
        raise IOError(u'文件不存在')
    data_set = np.loadtxt(data_file, dtype=float, delimiter=',')
    np.random.seed(1)
    train_data = data_set[:-5]
    test_data = data_set[-5:]
    train_data_r = get_train_data(train_data)
    test_data_r = get_test_data(test_data)
    return (train_data_r, test_data_r)

def get_train_data(train_data):
    train_x , train_y = train_data[:, :-1], train_data[:, -1]
    train_x_r = [x.reshape(len(x), 1) for x in train_x]
    train_y_r = [np.array([[1 if y==0 else 0],
                           [1 if y==1 else 0],
                           [1 if y==2 else 0]])
                           for y in train_y]
    train_data_r = [[x, y] for x, y in zip(train_x_r, train_y_r)]
    return train_data_r

def get_test_data(test_data):
    test_x , test_y = test_data[:, :-1], test_data[:, -1]
    test_x_r = [x.reshape(len(x), 1) for x in test_x]
    test_data_r = [[x, y] for x, y in zip(test_x_r, test_y)]
    return test_data_r
