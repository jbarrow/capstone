import cPickle as pickle
import numpy as np

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
    validation_split=0.1, callbacks=[checkpoint]
)

print "Saving fitted model..."
json_string = model.to_json()
open('models/model_cqt.json', 'w').write(json_string)
model.save_weights('models/model_cqt.h5', overwrite=True)

print "Running model..."
pred = model.predict(data.X_train, batch_size=1)

print "Compute note probabilities..."
(runs, steps, notes) = pred.shape
pred_unrolled = np.reshape(pred, (runs*steps, notes))
y_train_unrolled = np.reshape(data.y_train, (runs*steps, notes))
notes_played = np.argmax(y_train_unrolled, axis=1)
note_prob = np.zeros((notes, notes))
for note in range(notes):
    where_notes_played = np.where(notes_played == note)
    note_is_played = where_notes_played[0].size > 0
    if note_is_played:
        note_prob[note] = np.mean(pred_unrolled[where_notes_played][:], axis=0)

print "Saving note probabilities..."


data.close()
