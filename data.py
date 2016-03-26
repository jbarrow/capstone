import cPickle as pickle
import random

from collections import namedtuple

DataSet = namedtuple('DataSet', ['X', 'y'])

class DataContainer:
    def __init__(self, data=[]):
        self.data = data

    def load(self, filename):
        self = pickle.load(open(filename, 'rb'))
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def split(self, train=0.7, test=0.2, validation=0.1):
        # ensure we've been given a valid split
        assert train + test + validation == 1.0
        # shuffle the data
        random.shuffle(self.data)
        # get the index information we'll need
        count = len(self.data)
        fst, snd = int(count*train), int(count*test)
        # split into the train, test, and validation sets
        self.train = self.data[:fst]
        self.test = self.data[fst:fst+snd]
        self.val = self.data[fst+snd:]

