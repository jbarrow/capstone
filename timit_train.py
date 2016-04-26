import numpy as np

from data import DataContainer
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

nb_epoch=30
batch_size=100

np.random.seed(1994)

print "Loading data..."
data = DataContainer('timit.h5', in_memory=True)

def model_factory():
    model = Sequential()
    
    model.add(TimeDistributedDense(input_dim=40, output_dim=93))
    model.add(Activation('sigmoid'))
    
    model.add(
        LSTM(
            input_dim=93, output_dim=93,
            return_sequences=True
        )
    )

    model.add(TimeDistributedDense(input_dim=93, output_dim=63))
    model.add(Activation('softmax'))

    print "Compiling model..."
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model

checkpoint = ModelCheckpoint('models/timit_weieghts.h5')
print "Assembling model..."
model = model_factory()

model.fit(
    data.X_train, data.y_train,
    batch_size=batch_size, nb_epoch=nb_epoch,
    show_accuracy=True, verbose=1,
    validation_split=.1, callbacks=[checkpoint]
)

print "Saving fitted model..."
json_string = model.to_json()
open('models/timit.json', 'w').write(json_string)
model.save_weights('models/timit.h5', overwrite=True)

data.close()
