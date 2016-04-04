import numpy as np
import random
import h5py
import math

from collections import namedtuple

DataSet = namedtuple('DataSet', ['X', 'y'])

class DataContainer:
    def __init__(self, filename, in_memory=False):
        self.in_memory = in_memory
        self.f = h5py.File(filename)
        self.count = self.f['X'].shape[0]
        self.split()

        if self.in_memory:
            self.X_train = self.f['X'][self.train, :, :]
            self.y_train = self.f['y'][self.train, :, :]
            self.X_test = self.f['X'][self.test, :, :]
            self.y_test = self.f['y'][self.test, :, :]
            
    def split(self, train=0.9, test=0.1, validation=0.0):
        shuffled = np.random.permutation(self.count)
        
        ind = map(int, map(round, [train*self.count, test*self.count, validation*self.count]))
        ind[1] += ind[0] ; ind[2] = self.count+1
        
        self.train, self.test, self.validation, nix = np.split(shuffled, ind)
        self.train = np.sort(self.train)
        self.test = np.sort(self.test)
        self.validation = np.sort(self.validation)
        
    def minibatches(self, batch_size):
        its = int(math.ceil(1.0*len(self.train)/batch_size))
        for i in range(its):
            batch = np.sort(self.train[i*batch_size:(i+1)*batch_size])
            X = self.f['X'][batch, :, :]
            y = self.f['y'][batch, :, :]
            yield (X, y)

    def close(self):
        self.f.close()
