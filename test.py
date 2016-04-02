import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json

model = model_from_json(open('maps_lstm.json').read())
model.load_weights('maps_lstm.h5')

s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)

d = s.process(audio_file='scale.wav')

plt.imshow(p[0])
plt.show()
