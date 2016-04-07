import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.cm as cm

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json


def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')

    return model
    
s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)

d = s.process(audio_file='minuet.wav')
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

m0 = load_model('models/model_0')
m1 = load_model('models/model_1')
m2 = load_model('models/model_2')
m3 = load_model('models/model_3')
m4 = load_model('models/model_4')
m5 = load_model('models/model_5')
m6 = load_model('models/bdlstm')


p0 = m0.predict(d_train, batch_size=1)
p1 = m1.predict(d_train, batch_size=1)
p2 = m2.predict(d_train, batch_size=1)
p3 = m2.predict(d_train, batch_size=1)
p4 = m2.predict(d_train, batch_size=1)
p5 = m2.predict(d_train, batch_size=1)
p6 = m6.predict([d_train, d_train], batch_size=1)

#p = model.predict(d_train, batch_size=1)

p = [p0[0] + p1[0] + p2[0] + p3[0] + p4[0] + p5[0] + p6[0]]

print p[0]

y = np.zeros((89, len(p[0])))
for i, r in enumerate(p[0]):
    c = np.argmax(r)
    y[c, i] = 1.0
    
plt.imshow(y[:-1,:], cmap=cm.binary)
plt.show()
