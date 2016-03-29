import cPickle as pickle
import numpy as np
import random
import h5py

from collections import namedtuple

DataSet = namedtuple('DataSet', ['X', 'y'])

class DataContainer:
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]

    def save(self, filename):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('X', data=self.X)
            hf.create_dataset('y', data=self.y)
            
    def split(self, train=0.7, test=0.2, validation=0.1):
        count = len(self.X)
        shuffled = np.random.permutation(count)
        
        ind = map(int, map(round, [train*count, test*count, validation*count]))
        ind[1] += ind[0] ; ind[2] = count+1
        
        self.train, self.test, self.validation, nix = np.split(shuffled, ind)
