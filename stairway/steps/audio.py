import scipy
import numpy as np

def stft(loaded_data, framesz, hop, pad=0):
    # unpack the data loaded from scipy.io.wavfile.read
    fs, x = loaded_data
    x = x[:, 0]
    # compute our constants
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    # construct our hanning window
    w = scipy.hanning(framesamp)
    # compute the stft -- really really really need to break this up!
    X = np.abs(
        scipy.array(
            [np.fft.rfft(
                np.lib.pad(
                    w*x[i:i+framesamp], (framesamp*pad,), 'constant', constant_values=(0,)))
             for i in range(0, len(x)-framesamp, hopsamp)]))/framesamp*2
    return X
    
