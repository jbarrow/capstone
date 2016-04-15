import cPickle as pickle
import numpy as np
import h5py

from data import DataContainer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

nb_epoch=30
batch_size=100

np.random.seed(1994)

print "Loading data..."
data = DataContainer('data.h5', in_memory=True)

def model_factory():
    print "Assembling model..."
    model = Sequential()
    model.add(
        LSTM(
            input_dim=2206, output_dim=256,
            return_sequences=True, activation='tanh',
            dropout_U=0.25, dropout_W=0.25, W_regularizer='l2'
        )
    )
    
    model.add(TimeDistributedDense(input_dim=256, output_dim=89))
    model.add(Activation('softmax'))

    print "Compiling model..."
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model

checkpoint = ModelCheckpoint('models/weights.{epoch:02d}-{val_loss:.2f}.h5')
model = model_factory()
print "Fitting model..."

model.fit(
    data.X_train, data.y_train,
    batch_size=batch_size, nb_epoch=nb_epoch,
    show_accuracy=True, verbose=1,
    validation_split=0.4, callbacks=[checkpoint]
)

print "Saving fitted model..."
json_string = model.to_json()
open('models/model_cqt.json', 'w').write(json_string)
model.save_weights('models/model_cqt.h5', overwrite=True)

data.close()
