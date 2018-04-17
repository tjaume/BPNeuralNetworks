#encoding:utf-8

from network import Network
from BPNN import BPNNClassification
import IrisLoader
import DoubanRateLoader

def test1():
    train_data, test_data = IrisLoader.load_data()
    # bp = Network([4, 15, 3])
    # bp.SGD(train_data, 10000, 10, 0.5, test_data=test_data)

    bp1 = BPNNClassification([4, 15, 3])
    bp1.MSGD(train_data, 10000, 10, 0.5, test_data=test_data)

def test2():
    train_data, test_data = DoubanRateLoader.load_data()
    # bp = Network([4, 15, 3])
    # bp.SGD(train_data, 10000, 10, 0.5, test_data=test_data)

    bp1 = BPNNClassification([5, 15, 4])
    bp1.MSGD(train_data, 200, 30, 0.3, test_data=test_data)

test2()