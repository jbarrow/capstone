import scipy
import numpy as np

from itertools import imap
from nsgt import NSGT, OctScale
from scipy.interpolate import interp1d

# Taken from the NSGT library examples
class Interpolate:
    def __init__(self, cqt, Ls):
        self.intp = [interp1d(np.linspace(0, Ls, len(r)), r) for r in cqt]

    def __call__(self, x):
        return np.array([i(x) for i in self.intp])


def stft(loaded_data, framesz, hop, pad=0):
    # unpack the data loaded from scipy.io.wavfile.read
    fs, x = loaded_data
    if len(x.shape) > 1: x = np.mean(x, axis=1)
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
    
def cqt(loaded_data, framesz, bins):
    # unpack the data (and convert to stereo if necessary)
    fs, x = loaded_data
    if len(x.shape) > 1: x = np.mean(x, axis=1)
    Lx = len(x)
    # create the NSGT instance
    nsgt = NSGT(OctScale(80, 22050, bins), fs, Lx, real=True)
    # compute the constant q transfrom
    frames = nsgt.forward(x)
    # get the data into a grid
    g = np.linspace(0, Lx, Lx//(framesz*fs))
    X = Interpolate(imap(np.abs, frames[2:-1]), Lx)(g)
    return X.T
