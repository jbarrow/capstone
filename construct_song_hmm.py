import scipy.io.wavfile
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *

class Song:
    
    # Constants
    p_correct = 0.9
    
    def __init__(self, notes, durations, note_distribution_file):
        self.note_states = []
        self.notes = notes
        self.durations = durations
        hf = h5py.File(note_distribution_file, 'r')
        self.note_prob = hf.get('note_prob')[:][:].copy()
        hf.close()
        # initialize the hidden markov model for the score
        self.hmm = HiddenMarkovModel()
        self.add_note_states()
        self.add_mistake_state()
        self.add_mistake_transitions()
        self.add_note_transitions()
        self.hmm.bake()

    def add_note_states(self):
        for i, note in enumerate(notes):
            self.add_note(note, i)

    def add_note(self, note, i):
        dist = DiscreteDistribution(dict(enumerate(self.note_prob[note])))
        state = State(dist, name='{}: {}'.format(i, note))
        # add the state to our model
        self.hmm.add_state(state)
        self.note_states.append(state)

    def add_mistake_state(self):
        mean_prob = np.mean(self.note_prob, axis=0)
        distr = DiscreteDistribution(dict(enumerate(mean_prob)))
        self.mistake_state = State(distr, name='mistake')
        self.hmm.add_state(self.mistake_state)

    def add_mistake_transitions(self):
        p_mistake_to_note = 1. / (len(self.note_states) + 1)
        p_note_to_mistake = 1. - self.p_correct
        for note_state in self.note_states:
            self.hmm.add_transition(note_state, self.mistake_state, p_note_to_mistake)
            self.hmm.add_transition(self.mistake_state, note_state, p_mistake_to_note)
        # start & end states
        self.hmm.add_transition(self.hmm.start, self.mistake_state, p_note_to_mistake)
        self.hmm.add_transition(self.mistake_state, self.hmm.end, p_mistake_to_note)

    def add_note_transitions(self):
        for i in range(len(self.note_states)-1):
            duration = self.durations[i]
            state = self.note_states[i]
            next_state = self.note_states[i+1]
            self.add_note_note_transition(state, next_state, duration)
        # start & end note states
        self.hmm.add_transition(self.hmm.start, self.note_states[0], self.p_correct)
        self.hmm.add_transition(self.note_states[-1], self.hmm.end, 1. / durations[-1])

    def add_note_note_transition(self, state, next_state, duration):
        # compute our transition probabilities
        otp = 1. / duration
        stp = self.p_correct - otp
        # add all the transitions
        self.hmm.add_transition(state, state, stp)
        self.hmm.add_transition(state, next_state, otp)

    def play(self, predictions):
        pred_indices = np.argmax(predictions, axis=1)
        prob_path = self.hmm.predict_proba(pred_indices)
        self.state_index_path = np.argmax(prob_path, axis=1)
        self.state_path = [self.hmm.states[i] for i in self.state_index_path]
        print "HMM prediction (state index: note index):"
        print [state.name for state in self.state_path]
        print "for RNN prediction:"
        print pred_indices
        
def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')
    return model
    
s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)

print "Loading audio file..."
file_name = 'data/basic.wav'
d = s.process(audio_file=file_name)
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

print "Loading model..."
model = load_model('models/model_1')

print "Predicting with model..."
pred = model.predict(d_train, batch_size=1)
pred = np.squeeze(pred)

print "Predicting with HMM..."
note_distribution_file = 'note_distribution.h5'
notes = [88, 39, 41, 43, 44, 46, 88] # DOES THE AUDIO INCLUDE SILENCES
bad_notes = [88, 39, 70, 43, 44, 46, 88]
durations = [15., 10., 10., 10., 10., 10., 15.] 
s = Song(bad_notes, durations, note_distribution_file)
s.play(pred)
