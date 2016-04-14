import numpy as np
import argparse
import h5py

from data import DataContainer
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

nb_epoch=30
batch_size=100

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', dest='model_name', default='maps_lstm')
    parser.add_argument('--seed', dest='seed', type=int, default=1994)
    args = parser.parse_args()
    
    # seed the random number generator
    np.random.seed(args.seed)

    checkpoint = ModelCheckpoint('models/{0}.h5'.format(args.model_name), save_best_only=True)
    model = model_factory()
    print "Fitting model..."
    
    model.fit(
        data.X_train, data.y_train,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1,
        validation_split=0.5, callbacks=[checkpoint]
    )

    print "Saving fitted model..."
    json_string = model.to_json()
    open('models/{0}.json'.format(args.model_name), 'w').write(json_string)
    model.save_weights('models/{0}_final.h5'.format(args.model_name), overwrite=True)
    
    data.close()
