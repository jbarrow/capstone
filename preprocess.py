from preprocessing.trainingdata import TrainingData
from preprocessing.filterbank import FilterBank
from preprocessing.band import Band

import numpy as np

print "Loading training data, MAPS_MUS-alb_se7_AkPnBsdf."
audio = TrainingData('data/MUS/MAPS_MUS-alb_se7_AkPnBsdf')
print "Windowing and running STFT."
audio.transform()
print "Constructing filterbank."
f = FilterBank(audio.F)
f.construct_bands(440.0, 20, 20)
print "Preprocessing with constructed filterbank."
audio.preprocess(f)
print "Resulting shape:", np.shape(audio.data)
