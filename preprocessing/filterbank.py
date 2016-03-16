import numpy as np
import math

from band import Band

def compute_freq(f0, count):
    return f0 * math.pow(math.pow(2, 1.0/12.0), count)

class FilterBank:
    def __init__(self, sample_rate, window_size, padding):
        bins = (sample_rate * window_size * (2 * padding + 1)) / 2
        res = 0.5 * sample_rate / bins
        self.bands = []
        self.freqs = np.array([res*i for i in range(int(round(bins)))])
        self.diff = self.freqs[1]-self.freqs[0]
        self.offset = self.freqs[0]/self.diff
    
    def discretize(self, value):
        return int(round(value/self.diff)-self.offset)

    def band(self, low, center, high):
        l, c, h = self.discretize(low), self.discretize(center), self.discretize(high)
        self.bands.append(Band(l, c, h))
    
    def construct_bands(self, f0, n_below, n_above):
        for i in range(1, n_below+1)[::-1]:
            self.band(compute_freq(f0, -i-1), compute_freq(f0, -i), compute_freq(f0, -i+1))
        for i in range(n_above+1):
            self.band(compute_freq(f0, i-1), compute_freq(f0, i), compute_freq(f0, i+1))
        self.width = n_below + n_above + 1
    
    def apply_filterbank(self, freqs):
        values = np.zeros(len(self.bands))
        for i, band in enumerate(self.bands):
            values[i-11] = band.values.dot(freqs[band.low:band.high+1]) / band.total
        return values
