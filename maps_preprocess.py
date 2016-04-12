import scipy.io.wavfile
import pandas as pd
import numpy as np
import argparse
import h5py
import math

from stairway import Stairway
from stairway.steps import stft, r_load_pairs, cqt

frame_size = 0.1
hop_size = 0.025
fs = 400
#bins=64

def label(labels, data, hop_size):
    y = np.zeros((np.shape(data)[0], 89))
    for i, row in labels.iterrows():
        onset_index = int(math.floor(row.OnsetTime / hop_size))
        offset_index = int(round(row.OffsetTime / hop_size))
        y[onset_index:offset_index, int(row.MidiPitch-21)] = 1.0
    for i, r in enumerate(y):
        if np.sum(r) == 0.0:
            y[i, 88] = 1.0
    return y

def split(audio, label, fs=400):
    lens = [fs*(i+1) for i in range(audio.shape[0]/fs)]
    return np.split(audio, lens), np.split(label, lens)

def pad(data, fs):
    cnt = len(data[0][-1])
    if cnt < fs:
        data[0][-1] = np.lib.pad(data[0][-1], ((fs-cnt, 0),(0, 0)), 'constant')
        data[1][-1] = np.lib.pad(data[1][-1], ((fs-cnt, 0),(0, 0)), 'constant')
        data[1][-1][:cnt, 88] = 1.0
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='recursive_top_dir')

    args = parser.parse_args()
    rdir = args.recursive_top_dir
    
    s = Stairway(False)\
        .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
        .step('load_label', ['label_file'], pd.read_csv, sep='\t')\
        .step('transform', ['load_audio'], stft, frame_size, hop_size)\
        .step('label', ['transform', 'load_label'], label, hop_size)\
        .step('split', ['transform', 'label'], split, fs=fs)\
        .step('pad', ['split'], pad, fs=fs)

    files = r_load_pairs(rdir, exts=['.wav', '.txt'])
    
    with h5py.File('data_test.h5', 'w') as hf:
        X = hf.create_dataset('X', (0, fs, 518), maxshape=(None, fs, 5), dtype='float32')
        y = hf.create_dataset('y', (0, fs, 89), maxshape=(None, fs, 89), dtype='float32')

        cnt = 0
        for f in files:
            print 'Preprocessing:', f[0]
            data = s.process(audio_file=f[0], label_file=f[1])
            X.resize((X.shape[0]+len(data[0]),)+X.shape[1:])
            y.resize((y.shape[0]+len(data[1]),)+y.shape[1:])
            
            for i in range(len(data[0])):
                X[cnt, :, :] = data[0][i]
                y[cnt, :, :] = data[1][i]
                cnt += 1
