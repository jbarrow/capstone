import cPickle as pickle
import numpy as np

from data import DataContainer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential

nb_epoch=15
batch_size=100

np.random.seed(1994)

print "Loading data..."
data = DataContainer('data.h5', in_memory=True)

print "Assembling model..."
hidden_units=256

left = Sequential()
left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_dim=2206, dropout_W=0.25,
               dropout_U=0.25))
right = Sequential()
right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_dim=2206, go_backwards=True,
               dropout_W=0.25, dropout_U=0.25))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

model.add(TimeDistributedDense(input_dim=256, output_dim=89))
model.add(Activation('softmax'))

print "Compiling model..."
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

print "Fitting model..."
model.fit([data.X_train, data.X_train], data.y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.1)

print "Saving fitted model..."
json_string = model.to_json()
open('models/bdlstm.json', 'w').write(json_string)
model.save_weights('models/bdlstm.h5', overwrite=True)
