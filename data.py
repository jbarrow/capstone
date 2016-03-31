import numpy as np
import random
import h5py
import math

from collections import namedtuple

DataSet = namedtuple('DataSet', ['X', 'y'])

class DataContainer:
    def __init__(self, filename):
        self.f = h5py.File(filename)
        self.count = self.f['X'].shape[0]
        self.split()
            
    def split(self, train=1.0, test=0.0, validation=0.0):
        shuffled = np.random.permutation(self.count)
        
        ind = map(int, map(round, [train*self.count, test*self.count, validation*self.count]))
        ind[1] += ind[0] ; ind[2] = self.count+1
        
        self.train, self.test, self.validation, nix = np.split(shuffled, ind)

    def minibatches(self, batch_size):
        its = int(math.ceil(1.0*len(self.train)/batch_size))
        for i in range(its):
            batch = np.sort(self.train[i*batch_size:(i+1)*batch_size])
            X = self.f['X'][batch, :, :]
            y = self.f['y'][batch, :, :]
            yield (X, y)

    def close(self):
        self.f.close()
