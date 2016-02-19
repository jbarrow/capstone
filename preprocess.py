from preprocessing.trainingdata import TrainingData
from preprocessing.filterbank import FilterBank
from preprocessing.band import Band

import numpy as np

print "Constructing filterbank."
f = FilterBank(44100, 0.05, 5)
f.construct_bands(440.0, 20, 20)

print "Loading and preprocessing training data, MAPS_MUS-alb_se7_AkPnBsdf."
audio = TrainingData('data/MUS/MAPS_MUS-alb_se7_AkPnBsdf', f)
