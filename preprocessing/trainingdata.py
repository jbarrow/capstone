import pandas as pd
import numpy as np
import scipy
import scipy.io.wavfile
import math

def stft(x, fs, framesz, hop, pad=0):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = np.abs(scipy.array([np.fft.rfft(np.lib.pad(w*x[i:i+framesamp], (framesamp*pad,), 'constant', constant_values=(0,)))
                     for i in range(0, len(x)-framesamp, hopsamp)]))/framesamp*2
    return X

class TrainingData:
    def __init__(self, file_name, filterbank, window_size=0.05, hop_size=0.025, pad=5):
        # read in the audio of the file
        self.rate, self.x = scipy.io.wavfile.read(file_name+'.wav')
        self.duration = len(self.x) / self.rate
        # read in the training data
        self.notes = pd.read_csv(file_name+'.txt', sep='\t')
        self.notes.columns = ['onset_time', 'offset_time', 'midi_pitch']
        # some key constants
        self.window_size = window_size
        self.hop_size = hop_size
        self.pad = pad
        # take the audio transformation
        self.transform()
        self.preprocess(filterbank)
        self.label()
        self.clean()
        
    def transform(self):
        self.data = stft(self.x[:, 0], self.rate, self.window_size, self.hop_size, self.pad)
        #self.F = np.fft.rfftfreq(int(self.rate*self.window_size*(2*self.pad+1)), 1.0/self.rate)

    def preprocess(self, filterbank):
        count = np.shape(self.data)[0]
        self.X = np.zeros((count, filterbank.width))
        for i in range(count):
            self.X[i, :] = filterbank.apply_filterbank(self.data[i, :])

    def label(self):
        count = np.shape(self.X)[0]
        self.Y = np.zeros((count, 88))
        factor = (self.window_size - self.hop_size)
        for i, row in self.notes.iterrows():
            onset_index = int(math.floor(row.onset_time / factor + self.hop_size))
            offset_index = int(round(row.offset_time / factor + self.hop_size))
            self.Y[onset_index:offset_index, int(row.midi_pitch-21)] = 1.0

    def clean(self):
        del self.x
        del self.data
