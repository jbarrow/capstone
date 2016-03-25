
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

#import cPickle as pickle
import scipy.io.wavfile
import pandas as pd
import numpy as np
import math
#import operator
#import random

#from collections import namedtuple

from stairway import Stairway
from stairway.steps import stft, get_output, print_data

def load_pandas(filename, sep):
    data = pd.read_csv(filename, sep=sep)
    data.columns = ['onset_time', 'offset_time', 'midi_pitch']
    return data

def label(labels, data, hop_size):
    y = np.zeros((np.shape(data)[0], 88))
    for i, row in labels.iterrows():
        onset_index = int(math.floor(row.onset_time / hop_size + hop_size))
        offset_index = int(round(row.offset_time / hop_size + hop_size))
        y[onset_index:offset_index, int(row.midi_pitch-21)] = 1.0
    return y
    
hop_size = 0.025
    
s = Stairway()\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('load_label', ['label_file'], load_pandas, '\t')\
    .step('stft', ['load_audio'], stft, 0.1, hop_size)\
    .step('label', ['stft', 'load_label'], label, hop_size)\
    .step('output', ['stft', 'label'], get_output)

base = './data/ISOL/RE/MAPS_ISOL_RE_F_S0_M37_ENSTDkCl'

print s.process(audio_file=base+'.wav', label_file=base+'.txt')


#DataSet = namedtuple('DataSet', ['X', 'y'])

#class DataContainer:
#    def __init__():
#        pass
#
#    def save(self, filename):
#        pickle.dump(self, open(filename, 'wb'))
#
#    def split(self, train=0.7, test=0.2, validation=0.1):
#        # ensure we've been given a valid split
#        assert train + test + validation == 1.0
#        # shuffle the data
#        random.shuffle(self.data)
#        # get the index information we'll need
#        count = len(self.data)
#        fst, snd = int(count*train), int(count*test)
#        # split into the train, test, and validation sets
#        self.train = self.data[:fst]
#        self.test = self.data[fst:fst+snd]
#        self.val = self.data[fst+snd:]
#
# FilePath -> DataContainer
#def load_data(filename):
#    return pickle.load(open(filename, 'rb'))
