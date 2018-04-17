# encoding:utf-8
'''
BP神经网络回归测试
'''

import BPNN
import numpy as np
import DoubanRateLoader

#一元二次多项式
def oneQua(x):
    return  2 * x ** 2 + 3 * x + 5

def get_train_data():
    x = np.linspace(-3.5, 3.5, 50)
    y = oneQua(x)
    train_data = [[sx.reshape(1,1), sy.reshape(1,1)] for sx, sy in zip(x, y)]
    return train_data

def get_test_data():
    x = np.array([-3.7, 2.6, 1.3, 0.3, -2.5])
    y = oneQua(x)
    test_data = [np.reshape(sx, (1,1)) for sx in x]
    print test_data
    return test_data, y
    
def test1():
    '''
    多项式预测
    '''
    bp1 = BPNN.BPNNRegression([1, 20, 1])

    train_data = get_train_data()
    test_data, y = get_test_data()

    bp1.MSGD(train_data, 10000, 8, 0.2)
    print y
    print bp1.predict(test_data)

def test2():
    train_data = DoubanRateLoader.load_data_for_regresstion()
    bp = BPNN.BPNNRegression([5, 7, 1])
    err_arr = []
    for rate in [0.1, 0.2, 0.4, 0.5, 0.6, 0.9]:
        err_arr.append(bp.MSGD(train_data, 10000, len(train_data), 0.3))
        break
    print err_arr

test2()

