import cPickle as pickle
import numpy as np
import random

from collections import namedtuple

DataSet = namedtuple('DataSet', ['X', 'y'])

class DataContainer:
    def __init__(self, data=[]):
        self.data = data
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def split(self, train=0.7, test=0.2, validation=0.1):
        # shuffle the data
        random.shuffle(self.data)
        # get the index information we'll need
        count = len(self.data)
        fst, snd = int(count*train), int(count*test)
        # split into the train, test, and validation sets
        self.train = self.data[:fst]
        self.test = self.data[fst:fst+snd]
        self.val = self.data[fst+snd:]

    def get_max_length(self):
        return max(map(lambda x: x[0].shape[0], self.data))

    def pad(self):
        m = self.get_max_length()
        self.X = np.zeros((len(self.data), m, 2206))
        self.y = np.zeros((len(self.data), m, 88))
        
        for i, datum in enumerate(self.data):
            self.X[i, m-len(datum[0]):, :] = datum[0]
            self.y[i, m-len(datum[1]):, :] = datum[1]
            

def load_container(filename):
    return pickle.load(open(filename, 'rb'))
