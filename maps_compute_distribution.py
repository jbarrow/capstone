import numpy as np
import h5py

from data import DataContainer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')
    return model

print "Loading model..."
model = load_model('models/model_cqt')

print "Loading data..."
data = DataContainer('data.h5', in_memory=True)

print "Running model..."
pred = model.predict(data.X_train, batch_size=1)

print "Computing note distribution..."
(runs, steps, notes) = pred.shape
pred_unrolled = np.reshape(pred, (runs*steps, notes))
y_train_unrolled = np.reshape(data.y_train, (runs*steps, notes))
notes_played = np.argmax(y_train_unrolled, axis=1)
note_prob = np.zeros((notes, notes)) # note x note probability
note_cov = np.zeros((notes, notes, notes)) # note x (cov mat of note)
for note in range(notes):
    where_notes_played = np.where(notes_played == note)
    note_is_played = where_notes_played[0].size > 0
    if where_notes_played[0].size < 89:
    	print "not enough examples for note " + str(note)
    if note_is_played:
        note_prob[note] = np.mean(pred_unrolled[where_notes_played][:], axis=0)
        note_cov[note] = np.cov(pred_unrolled[where_notes_played][:].T)

print "Saving note distribution..."
with h5py.File('note_distribution.h5', 'w') as hf:
    hf.create_dataset('note_prob', data=note_prob)
    hf.create_dataset('note_cov', data=note_cov)
