import numpy as np
import argparse
import h5py

from data import DataContainer
from keras.layers.core import TimeDistributedDense, Dropout, Activation, Merge, Masking
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

nb_epoch=20
batch_size=100

print "Loading data..."
data = DataContainer('maps_full_stft.h5', in_memory=True)

def model_factory():
    print "Assembling model..."
    #left = Sequential()
    #left.add(Masking(mask_value=0., input_shape=(400, 2206)))
    #left.add(
    #    LSTM(
    #        input_shape=(400, 2206), output_dim=128,
    #        return_sequences=True, activation='tanh',
    #        dropout_U=0.2, dropout_W=0.2, W_regularizer='l2'
    #    )
    #)

    #right = Sequential()
    #right.add(Masking(mask_value=0., input_shape=(400, 2206)))
    #right.add(
    #    LSTM(
    #        input_shape=(400, 2206), output_dim=128,
    #        return_sequences=True, activation='tanh',
    #        dropout_U=0.2, dropout_W=0.2, W_regularizer='l2', go_backwards=True
    #    )
    #)

    model = Sequential()
    #model.add(Masking(mask_value=0., input_shape=(400, 2206)))
    model.add(
        LSTM(
            input_dim=2206, output_dim=128,
            return_sequences=True, activation='tanh',
            dropout_U=0.2, dropout_W=0.2, W_regularizer='l2',
            go_backwards=True
        )
    )

    #model.add(Merge([left, right], mode='sum'))
    
    model.add(TimeDistributedDense(89))
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

    json_string = model.to_json()
    open('models/{0}.json'.format(args.model_name), 'w').write(json_string)
    
    print "Fitting model..."
    
    model.fit(
        data.X_train, data.y_train,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1, show_accuracy=True,
        validation_split=0.1, callbacks=[checkpoint]
    )

    print "Saving fitted model..."
    model.save_weights('models/{0}_final.h5'.format(args.model_name), overwrite=True)
    
    data.close()
 
