import scipy.io.wavfile
import pandas as pd
import numpy as np
import math
import os

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

frame_size = 0.1
hop_size = 0.025

s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('load_label', ['label_file'], load_pandas, '\t')\
    .step('stft', ['load_audio'], stft, frame_size, hop_size)\
    .step('label', ['stft', 'load_label'], label, hop_size)\
    .step('output', ['stft', 'label'], DataSet)

e = Escalator(r_load_pairs, ['directory', 'master', 'exts'])
e.mapper(s, ['audio_file', 'label_file']).graph\
    .step('get_container', ['map'], DataContainer)\
    .step('save', ['get_container'], lambda x: x.save('re.pkl'))

data = e.start(directory='./data/ISOL/RE', master='.txt', exts=['.wav', '.txt'])
