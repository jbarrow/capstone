import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.cm as cm

from stairway import Stairway
from stairway.steps import stft, cqt
from keras.models import model_from_json


def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')

    return model
    
s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], cqt, 0.05)

d = s.process(audio_file='scale.wav')
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

m0 = load_model('models/model_cqt')


p0 = m0.predict(d_train, batch_size=1)

#p = model.predict(d_train, batch_size=1)

p = p0

print p[0]

y = np.zeros((89, len(p[0])))
for i, r in enumerate(p[0]):
    c = np.argmax(r)
    y[c, i] = 1.0
    
plt.imshow(y[:-1,:], cmap=cm.binary)
plt.show()
