import numpy as np

class Band:
    def __init__(self, low, center, high):
        self.low, self.center, self.high = low, center, high
        self.construct()
        
    def construct(self):
        f, s = self.center-self.low, self.high-self.center
        self.values = np.zeros(self.high-self.low+1)
        for i in range(f):
            self.values[i] = float(i)/f
        self.values[f] = 1.0
        for i in range(1, s):
            self.values[f+i] = float(s-i)/s
        self.total = np.sum(self.values)
