import pandas as pd
import numpy as np
import scipy
import scipy.io.wavfile

def stft(x, fs, framesz, hop, pad=0):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = np.abs(scipy.array([np.fft.rfft(np.lib.pad(w*x[i:i+framesamp], (framesamp*pad,), 'constant', constant_values=(0,)))
                     for i in range(0, len(x)-framesamp, hopsamp)]))/framesamp*2
    return X

class TrainingData:
    def __init__(self, file_name):
        # read in the audio of the file
        self.rate, self.x = scipy.io.wavfile.read(file_name+'.wav')
        self.duration = len(self.x) / self.rate
        # read in the training data
        self.notes = pd.read_csv(file_name+'.txt', sep='\t')
        self.notes.columns = ['onset_time', 'offset_time', 'midi_pitch']
        
    def transform(self):
        framesz = 0.050
        pad = 5
        self.X = stft(self.x[:, 0], self.rate, framesz, hop=0.025, pad=pad)
        self.F = np.fft.rfftfreq(int(self.rate*framesz*(2*pad+1)), 1.0/self.rate)
        # clean up the variables we will no longer be using
        del self.x

    def preprocess(self, filterbank):
        count = np.shape(self.X)[0]
        self.data = np.zeros((count, filterbank.width))
        for i in range(count):
            self.data[i, :] = filterbank.apply_filterbank(self.X[i, :])
