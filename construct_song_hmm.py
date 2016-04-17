import scipy.io.wavfile
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *

class Song:
    note_distribution_file = 'note_distribution.h5'
    
    def __init__(self, notes, durations):
        # get the note distributions
        hf = h5py.File(self.note_distribution_file, 'r')
        self.note_prob = hf.get('note_prob')
        self.states = []
        
        # initialize the hidden markov model for the score
        self.hmm = HiddenMarkovModel()
        
        for i, note in enumerate(notes):
            self.add_note(note, i)
        for i, state in enumerate(self.states[:-1]):
            self.add_transition(self.states[i], self.states[i+1], durations[i])

        self.hmm.add_transition(self.hmm.start, self.states[0], 1.)
        self.hmm.add_transition(self.states[-1], self.hmm.end, 1. / durations[-1])
        self.hmm.bake()
        hf.close()
            
    def add_note(self, note, i):
        dist = DiscreteDistribution(dict(enumerate(self.note_prob[note])))
        state = State(dist, name=str(note)+str(i))
        # add the state to our model
        self.hmm.add_states(state)
        self.states.append(state)

    def add_transition(self, state, next_state, duration):
        # compute our transition probabilities
        stp = 1. - 1. / duration
        otp = 1. - stp
        # add all the transitions
        self.hmm.add_transition(state, state, stp)
        self.hmm.add_transition(state, next_state, otp)

    def play(self, predictions):
        pred_indices = np.argmax(predictions, axis=1)
        prob_path = self.hmm.predict_proba(pred_indices)
        path = np.argmax(prob_path, axis=1)
        print str(path)
        print [state.name for state in self.hmm.states]
        
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

s = Song([88, 39, 41, 43, 44, 46, 88], [15., 10., 10., 10., 10., 10., 15.])
s.play(pred)
