import scipy.io.wavfile
import numpy as np
import h5py

import matplotlib.pyplot as plt

from preprocessing import FilterBank
from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *

def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')
    return model

f = FilterBank(44100, 0.1, 0)
f.construct_bands(440.0, 48, 39)
    
s = Stairway(False)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)\
    .step('filterbank', ['stft'], f.apply_filterbank)

file_name = 'data/basic.wav'
d = s.process(audio_file=file_name)
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

print "Loading model..."
model = load_model('models/maps_cqt_centered')
#m1 = load_model('models/maps_preprocess')

p0 = model.predict([d_train, d_train], batch_size=1)
#p1 = m1.predict([d_train, d_train], batch_size=1)
y = np.zeros((89, len(p0[0])))

for i in range(len(p0[0])):
 #   if np.max(p0[0][i]) < np.max(p1[0][i]):
  #      y[np.argmax(p1[0][i]), i] = 1.0
  #  else:
    y[np.argmax(p0[0][i]), i] = 1.0

plt.imshow(np.flipud(y[:-1, :]), cmap=plt.cm.binary)
plt.show()
