import cPickle as pickle
import numpy as np

from data import DataContainer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

nb_epoch=50
batch_size=200

print "Loading data..."
data = DataContainer('data.h5')

print "Assembling model..."
model = Sequential()
model.add(LSTM(input_shape=(263,2206), output_dim=256, return_sequences=True))
model.add(LSTM(output_dim=256, return_sequences=True))
model.add(TimeDistributedDense(input_dim=256, output_dim=88))
model.add(Activation('softmax'))

print "Compiling model..."
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print "Fitting model..."
for e in range(nb_epoch):
    print("Epoch: {0}/{1}".format(e, nb_epoch))
    for X_train, Y_train in data.minibatches(batch_size):
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1)

print "Saving fitted model..."
json_string = model.to_json()
open('maps_lstm.json', 'w').write(json_string)
model.save_weights('maps_lstm.h5')

data.close()
