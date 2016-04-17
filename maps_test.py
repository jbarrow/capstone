import scipy.io.wavfile
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *

def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')
    return model
    
s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)

file_name = 'basic.wav'
d = s.process(audio_file=file_name)
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

print "Loading model..."
model = load_model('models/model_1')

print "Predicting with model..."
pred = model.predict(d_train, batch_size=1)
pred = np.squeeze(pred)

print "Loading note distribution..."
hf = h5py.File('note_distribution.h5', 'r')
note_prob = hf.get('note_prob')
note_cov = hf.get('note_cov')

print "Creating HMM..."
hmm = HiddenMarkovModel()
# simple 5-note sequence
notes = [39, 41, 43, 44, 46, 88] # C D E F G Silence
# durations for the file 'c d e f g.wav'
durations = [10., 13., 12., 12., 10., 15.]
self_trans_prob = np.ones(len(notes)) - np.ones(len(notes)) / durations
trans_prob = np.ones(len(notes)) - self_trans_prob
# add states
states = []
for i in range(len(notes)):
	note = notes[i]
	distr = DiscreteDistribution(dict(enumerate(note_prob[note])))
	state = State(distr, name=str(note))
	hmm.add_states(state)
	states.append(state)
# add transition probabilities
hmm.add_transition(hmm.start, states[5], 1.)
# silence
hmm.add_transition(states[5], states[5], self_trans_prob[5])
hmm.add_transition(states[5], states[0], trans_prob[5])
# note 0
hmm.add_transition(states[0], states[0], self_trans_prob[0])
hmm.add_transition(states[0], states[1], trans_prob[0])
# note 1
hmm.add_transition(states[1], states[1], self_trans_prob[1])
hmm.add_transition(states[1], states[2], trans_prob[1])
# note 2
hmm.add_transition(states[2], states[2], self_trans_prob[2])
hmm.add_transition(states[2], states[3], trans_prob[2])
# note 3
hmm.add_transition(states[3], states[3], self_trans_prob[3])
hmm.add_transition(states[3], states[4], trans_prob[4])
# note 4
hmm.add_transition(states[4], states[4], self_trans_prob[4])
hmm.add_transition(states[4], states[5], trans_prob[4])
# silence
hmm.add_transition(states[5], states[5], self_trans_prob[5])
hmm.add_transition(states[5], hmm.end, trans_prob[5])
# bake
hmm.bake()

print "Testing with HMM..."
pred_indices = np.argmax(pred, axis=1)
prob_path = hmm.predict_proba(pred_indices)
path = np.argmax(prob_path, axis=1)
print "Path of states:"
print str(path)
print "For notes:"
print [state.name for state in hmm.states]

hf.close()
