# encoding:utf-8

'''
BP神经网络Python实现
'''
import numpy as np;


def sigmoid(x):
    '''
    激活函数
    '''
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class BPNeuralNetwork:
    def __init__(self, sizes):

        # 神经网络结构
        self.num_layers = len(sizes)
        self.sizes = sizes

        # 初始化偏差，除输入层外， 其它每层每个节点都生成一个 biase 值（0-1）
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        # 随机生成每条神经元连接的 weight 值（0-1）
        self.weights = [np.random.randn(r, c)
                        for c, r in zip(sizes[:-1], sizes[1:])]
        
    def feed_forward(self, a):
        '''
        前向传输计算输出神经元的值
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def MSGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        小批量随机梯度下降法
        '''
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            # 随机打乱训练集顺序
            np.random.shuffle(training_data)
            # 根据小样本大小划分子训练集集合
            mini_batchs = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            # 利用每一个小样本训练集更新 w 和 b
            for mini_batch in mini_batchs:
                self.updata_WB_by_mini_batch(mini_batch, eta)
            
            #迭代一次后结果
            if test_data:
                print("Epoch {0}: {1} / {2}", j, self.predict(test_data), n)
    
    def updata_WB_by_mini_batch(self, mini_batch, eta):
        '''
        利用小样本训练集更新 w 和 b
        mini_batch: 小样本训练集
        eta: 学习率
        '''
        # 创建存储迭代小样本得到的 b 和 w 偏导数空矩阵，大小与 biases 和 weights 一致，初始值为 0   
        batch_par_b = [np.zeros(b.shape) for b in self.biases]
        batch_par_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 根据小样本中每个样本的输入 x, 输出 y, 计算 w 和 b 的偏导
            delta_b, delta_w = self.back_propagation(x, y)
            # 累加偏导 delta_b, delta_w 
            batch_par_b = [bb + dbb for bb, dbb in zip(batch_par_b, delta_b)]
            batch_par_w = [bw + dbw for bw, dbw in zip(batch_par_w, delta_w)]
        # 根据累加的偏导值 delta_b, delta_w 更新 b, w
        # 由于用了小样本，因此 eta 需除以小样本长度
        self.weights = [w - (eta / len(mini_batch)) * dw
                        for w, dw in zip(self.weights, batch_par_w)]
        self.biases = [b - (eta / len(mini_batch)) * db
                        for b, db in zip(self.biases, batch_par_b)]

    def back_propagation(self, x, y):
        '''
        利用误差后向传播算法对每个样本求解其 w 和 b 的更新量
        x: 输入神经元，行向量
        y: 输出神经元，行向量
        '''
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播，求得输出神经元的值
        a = x # 神经元输出值
        # 存储每个神经元输出
        activations = [x] 
        # 存储经过 sigmoid 函数计算的神经元的输入值，输入神经元除外
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z) # 输出神经元
            activations.append(a)
        
        # 求解输出层δ
        delta = self.cost_function(activations[-1], y) * sigmoid_prime(zs[-1])
        delta_b[-1] = [delta]
        delta_w[-1] = np.dot(delta, activations[-2].T)
        for lev in xrange(2, self.num_layers):
            # 从倒数第1层开始更新，因此需要采用-lev
            # 利用 lev + 1 层的 δ 计算 l 层的 δ 
            z = zs[-lev]
            zp = sigmoid_prime(z)
            delta = np.dot(self.weights[-lev+1].T, delta) * zp
            delta_b[-lev] = delta
            delta_w[-lev] = np.dot(delta, activations[-lev-1].T)
        return (delta_b, delta_w)
    
    def predict(self, test_data):
        test_result = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def cost_function(self, output_a, y):
        '''
        损失函数
        '''
        return (output_a - y)
