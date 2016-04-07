import cPickle as pickle
import numpy as np

from data import DataContainer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

nb_epoch=30
batch_size=100
nb_models=10

np.random.seed(1994)

print "Loading data..."
data = DataContainer('data.h5', in_memory=True)

def model_factory():
    print "Assembling model..."
    model = Sequential()
    #model.add(TimeDistributedDense(input_shape=(100, 2206), output_dim=256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(LSTM(input_dim=2206, output_dim=256, return_sequences=True, activation='tanh', dropout_U=0.25, dropout_W=0.25, W_regularizer='l2'))
    model.add(TimeDistributedDense(input_dim=256, output_dim=89))
    model.add(Activation('softmax'))

    print "Compiling model..."
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


models = [None,]*nb_models
for i in range(nb_models):
    models[i] = model_factory()
    print "Fitting model {0}...".format(i+3)
    models[i].fit(data.X_train, data.y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_split=0.1)

    print "Saving fitted model..."
    json_string = models[i].to_json()
    open('models/model_{0}.json'.format(i+3), 'w').write(json_string)
    models[i].save_weights('models/model_{0}.h5'.format(i+3), overwrite=True)

data.close()
