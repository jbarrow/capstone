import scipy.io.wavfile
import numpy as np
import h5py
import sklearn

import matplotlib.pyplot as plt
import operator

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *
from collections import defaultdict

def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')
    return model

def predict(models, training):
    predictions = []
    for m in models:
        predictions.append(m.predict(training, batch_size=1)[0])
    
    y = np.zeros((89, len(predictions[0])))

    for i in range(len(predictions[0])):
        final = defaultdict(float)
        for p in predictions:
            final[np.argmax(p[i])] += p[i, np.argmax(p[i])]
        pred = max(final.iteritems(), key=operator.itemgetter(1))[0]
        y[pred, i] = 1.0
    return y
    
s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.0125)

models = []
for i in range(6):
    print "Loading model {0}".format(i+1)
    models.append(load_model('models/ensemble/uni_nm_s{0}_e20'.format(pow(17, i+1))))

while(True):
    file_name = raw_input("Enter a file name to test: ")

    d = s.process(audio_file=file_name)
    d_train = np.zeros((1,)+d.shape)
    d_train[0] = d

    y = predict(models, d_train)

    plt.imshow(np.flipud(y[:-1, :]), cmap=plt.cm.binary, interpolation='nearest')
    plt.show()

    save = raw_input("Would you like to save the output (y/n): ")
    if save == 'y':
        fname = raw_input("Enter a filename: ")
        np.save(fname, y)
