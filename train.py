import cPickle as pickle
import numpy as np

from data import DataContainer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

nb_epoch=30
batch_size=100

print "Loading data..."
data = DataContainer('data.h5', in_memory=True)

print "Assembling model..."
model = Sequential()
model.add(LSTM(input_shape=(100,2206), output_dim=128, return_sequences=True))
model.add(Dropout(0.25))
model.add(TimeDistributedDense(input_dim=128, output_dim=89))
model.add(Activation('softmax'))

print "Compiling model..."
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

print "Fitting model..."
model.fit(data.X_train, data.y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(data.X_test, data.y_test))

print "Saving fitted model..."
json_string = model.to_json()
open('maps_lstm_all_dropout.json', 'w').write(json_string)
model.save_weights('maps_lstm_all_dropout.h5', overwrite=True)

data.close()
