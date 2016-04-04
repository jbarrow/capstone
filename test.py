import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json

model = model_from_json(open('maps_lstm_simple.json').read())
model.load_weights('maps_lstm_simple.h5')

s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)

d = s.process(audio_file='data/test/GScaleSlow.wav')
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

p = model.predict_classes(d_train, batch_size=1)
y = np.zeros((89, len(p[0])))
for i, e in enumerate(p[0]):
    y[e-1, i] = 1.0 

plt.imshow(y)
plt.show()
