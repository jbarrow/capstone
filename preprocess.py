import scipy.io.wavfile
import pandas as pd
import numpy as np
import argparse
import math
import os
import cPickle as pickle

from data import *
from stairway import Stairway, Escalator
from stairway.steps import stft, r_load_pairs, print_data

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

def pad_sequences(max_length, data=[]):
    with open(data, 'rb') as pf:
        datum = pickle.load(pf)
        X = np.zeros((1, max_length, datum[0].shape[1]))
        y = np.zeros((1, max_length, datum[1].shape[1]))

        X[0, max_length-len(datum[0]):, :] = datum[0]
        y[0, max_length-len(datum[1]):, :] = datum[1]

    return X, y

def rmax(x, data):
    if data[0].shape[0] > x: return data[0].shape[0]
    return x

def incremental_save(count_and_filename, data):
    cnt, filename = count_and_filename
    if cnt == 0:
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('X', data=data[0], maxshape=(None,)+data[0].shape[1:])
            hf.create_dataset('y', data=data[1], maxshape=(None,)+data[1].shape[1:])
    else:
        with h5py.File(filename) as hf:
            X, y = hf['X'], hf['y']
            X.resize((X.shape[0]+1,)+X.shape[1:])
            y.resize((y.shape[0]+1,)+y.shape[1:])
            X[cnt, :, :] = data[0]
            y[cnt, :, :] = data[1]
            
    return (cnt+1, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='recursive_top_dir')

    args = parser.parse_args()
    rdir = args.recursive_top_dir
    
    frame_size = 0.1
    hop_size = 0.025

    s = Stairway(False)\
        .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
        .step('load_label', ['label_file'], load_pandas, '\t')\
        .step('stft', ['load_audio'], stft, frame_size, hop_size)\
        .step('label', ['stft', 'load_label'], label, hop_size)\
        .step('output', ['stft', 'label'], DataSet)

    e = Escalator(r_load_pairs, ['directory', 'master', 'exts'])\
        .mapper(s.process, ['audio_file', 'label_file'], name='map_process')\
        .reducer(rmax, 0, [], name='reduce_max', deps=['map_process'])\
        .mapper(pad_sequences, ['data'], 'map_padding', ['map_process', 'reduce_max'])\
        .reducer(incremental_save, (0, 'data.h5'), [], name='save', deps=['map_padding'])
    
    data = e.start(directory=rdir, master='.txt', exts=['.wav', '.txt'])
