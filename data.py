
"""
download_maps(dir, all=False)  # ftp
download_timit(dir, all=False) # ftp
download_gmuas(dir) # dropbox

dataset = load_maps(specify subset of maps, split)
dataset = load_timit(specify subset of timit, split)
dataset = load_gmuas(specify subset of gmuas, split)

dataset.save(filename)
dataset = load_dataset(filename)

dataset.train.X, dataset.train.y
dataset.test.X, dataset.test.y
dataset.validation.X, dataset.validation.y


"""

import scipy.io.wavfile
import pandas as pd
import numpy as np
import math
import os

import cPickle as pickle
import random

from collections import namedtuple

from stairway import Stairway, Escalator
from stairway.steps import stft, r_load_pairs, print_data

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

