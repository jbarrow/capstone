import cPickle as pickle
import numpy as np

from data import DataContainer, load_container
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

print "Loading data..."
data = load_container('./re.pkl')

print "Splitting data..."
data.split()

print "Assembling model..."
model = Sequential()
model.add(LSTM(input_shape=(263,2206), output_dim=256, return_sequences=True))
model.add(LSTM(output_dim=256, return_sequences=True))
model.add(TimeDistributedDense(input_dim=256, output_dim=88))
model.add(Activation('softmax'))

print "Compiling model..."
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print "Fitting model..."
model.fit(data.X[data.train], data.y[data.train], batch_size=22, nb_epoch=50)

print "Saving fitted model..."
json_string = model.to_json()
open('maps_lstm.json', 'w').write(json_string)
model.save_weights('maps_lstm.h5')
