import cPickle as pickle
import numpy as np
import random

from collections import namedtuple

DataSet = namedtuple('DataSet', ['X', 'y'])

class DataContainer:
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def split(self, train=0.7, test=0.2, validation=0.1):
        count = len(self.X)
        shuffled = np.random.permutation(count)
        
        ind = map(round, [train*count, test*count, validation*count])
        ind[1] += ind[0] ; ind[2] += ind[1]
        
        self.train, self.test, self.validation = np.split(shuffled, ind)

def load_container(filename):
    return pickle.load(open(filename, 'rb'))
