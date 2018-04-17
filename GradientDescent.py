# encoding:utf-8

'''
梯度下降算法
'''
import numpy as np;

test_result = [[np.array([[0.5]]), np.array([[0.3]])],
     [np.array([[0.5]]), np.array([[0.3]])],
     [np.array([[0.5]]), np.array([[0.3]])],
]
t = [0.5 * (x - y) ** 2 for (x, y) in test_result]
print np.sum(t)
a = np.sum(0.5 * (x - y) ** 2 for (x, y) in test_result)

print a
if a > np.array([[0.04]]):
    print('ok')

print np.sum([
    np.array([[0.04]]),
    np.array([[0.04]])
])