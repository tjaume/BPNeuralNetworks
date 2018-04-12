#encoding:utf-8

from network import Network
import IrisLoader

train_data, test_data = IrisLoader.load_data()
bp = Network([4, 10, 3])