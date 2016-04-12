import pandas as pd
import numpy as np
import subprocess
import os.path
import glob
import h5py
import math

from scipy.io import wavfile
from stairway import Stairway
from stairway.steps import stft, r_load_pairs, split, pad

frame_size=0.1
hop_size=0.025
fs = 400

def fix_wavs():
    for d in ['TRAIN', 'TEST']:
        timit_files = glob.glob('./data/TIMIT/{0}/*/*/*.WAV'.format(d))
        wav_files = []

        for f in timit_files:
            name, ext = os.path.split(f)
            name += '_tmp'
            wav_files.append(name+ext)

        sx_cmd = 'sox {0} -t wav {1}'
        mv_cmd = 'mv {0} {1}'
        
        for i, f in enumerate(timit_files):
            subprocess.call(sx_cmd.format(f, wav_files[i]), shell=True)
            os.remove(f)
            subprocess.call(mv_cmd.format(wav_files[i], f), shell=True)

    with open('./data/TIMIT/corrected.txt', 'w+') as f:
        f.write('True')

def load_phonemes(filename):
    with open(filename, 'r') as f:
        phonemes = filter(None, [p.strip() for p in f])
    return phonemes

def label(labels, data, hop_size, sample_rate, phone_to_index):
    y = np.zeros((data.shape[0], len(phone_to_index)))
    for i, row in labels.iterrows():
        onset_index = int(math.floor((1.0 * row[0] / sample_rate / hop_size)))
        offset_index = int(math.floor((1.0 * row[1] / sample_rate / hop_size)))
        y[onset_index:offset_index, phone_to_index[row[2]]] = 1.0
    return y

if __name__ == '__main__':
    if not os.path.isfile('./data/TIMIT/corrected.txt'):
        print "Fixing wav files..."
        fix_wavs()
    else: print "Wav files already corrected..."

    index_to_phone = load_phonemes('./data/TIMIT/phonemes.txt')
    phone_to_index = dict([(w, i) for i, w in enumerate(index_to_phone)])
    phone_cnt = len(index_to_phone)
    
    s = Stairway(False)\
        .step('load_audio', ['audio_file'], wavfile.read)\
        .step('load_label', ['label_file'], pd.read_csv, sep=' ', header=None)\
        .step('transform', ['load_audio'], stft, frame_size, hop_size)\
        .step('label', ['transform', 'load_label'], label, hop_size, 16000, phone_to_index)\
        .step('split', ['transform', 'label'], split, fs=fs)\
        .step('pad', ['split'], pad, fs=fs, silence=True, silence_index=phone_to_index['h#'])

    files = r_load_pairs('./data/TIMIT/TRAIN', exts=['.WAV', '.PHN'], master='.PHN')

    with h5py.File('timit.h5', 'w') as hf:
        X = hf.create_dataset('X', (0, fs, 801), maxshape=(None, fs, 801), dtype='float32')
        y = hf.create_dataset('y', (0, fs, phone_cnt), maxshape=(None, fs, phone_cnt), dtype='float32')

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
