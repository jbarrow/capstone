import scipy.io.wavfile
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *
from itertools import groupby
from operator import itemgetter

class Song:
    
    # Constants
    p_correct = 0.9
    
    def __init__(self, notes, durations, note_distribution_file):
        self.note_states = []
        self.mistake_states = []
        self.notes = notes
        self.durations = durations
        hf = h5py.File(note_distribution_file, 'r')
        self.note_prob = hf.get('note_prob')[:][:].copy()
        hf.close()
        # initialize the hidden markov model for the score
        self.hmm = HiddenMarkovModel()
        self.add_note_states()
        self.add_mistake_states()
        self.add_mistake_transitions()
        self.add_note_transitions()
        self.hmm.bake()

    def add_note_states(self):
        for i, note in enumerate(self.notes):
            self.add_note(note, i)

    def add_note(self, note, i):
        dist = DiscreteDistribution(dict(enumerate(self.note_prob[note])))
        state = State(dist, name='{}: {}'.format(i, note))
        # add the state to our model
        self.hmm.add_state(state)
        self.note_states.append(state)

    def add_mistake_states(self):
        for n in range(len(self.notes)):
            del_notes = [self.notes[n]]
            if n < len(self.notes)-1:
                del_notes.append(self.notes[n+1])
            notes = np.delete(range(self.note_prob.shape[0]), del_notes)
            prob = np.delete(self.note_prob, del_notes, axis=0)
            num_notes = prob.shape[0]
            distr = [DiscreteDistribution(dict(enumerate(prob[i]))) for i in range(num_notes)]
            states = [State(distr[i], name='m_{}_{}'.format(n, notes[i])) for i in range(num_notes)]
            self.mistake_states.append(states)
            self.hmm.add_states(states)

    def states_without_notes(self, notes_to_remove):
        notes = np.delete(range(self.note_prob.shape[0]), del_notes)
        prob = np.delete(self.note_prob, del_notes, axis=0)
        num_notes = prob.shape[0]
        distr = [DiscreteDistribution(dict(enumerate(prob[i]))) for i in range(num_notes)]
        states = [State(distr[i], name='m_{}_{}'.format(n, notes[i])) for i in range(num_notes)]
        return states

    def add_mistake_transitions(self):
        p_mistake_to_mistake = 1. - 1. / np.mean(self.durations)
        p_mistake_to_note = (1. - p_mistake_to_mistake) / (len(self.note_states) + 1)
        p_note_to_mistake = (1. - self.p_correct) / len(self.mistake_states[0])
        for i in range(len(self.notes)):
            states = self.mistake_states[i]
            note_state = self.note_states[i]
            for state in states:
                self.hmm.add_transition(note_state, state, p_note_to_mistake)
                self.hmm.add_transition(state, note_state, p_mistake_to_note)
                self.hmm.add_transition(state, state, p_mistake_to_mistake)

    def add_note_transitions(self):
        for i in range(len(self.note_states)-1):
            duration = self.durations[i]
            state = self.note_states[i]
            next_state = self.note_states[i+1]
            self.add_note_note_transition(state, next_state, duration)
        # start & end note states
        self.hmm.add_transition(self.hmm.start, self.note_states[0], self.p_correct)
        self.hmm.add_transition(self.note_states[-1], self.hmm.end, 1. / self.durations[-1])

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
        print "HMM prediction ('state index: note index', count):"
        state_list = [state.name for state in self.state_path]
        state_sum = [(key, len(list(group))) for key, group in groupby(state_list)]
        print "\t{}".format(state_sum)
        print "for RNN prediction (note, count):"
        pred_sum = [(key, len(list(group))) for key, group in groupby(pred_indices)]
        print "\t{}".format(pred_sum)

if __name__ == '__main__':

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
    notes =     [39, 41, 43, 44, 46]
    bad_notes = [39, 20, 45, 44, 46]
    durations = [10., 10., 10., 10., 10.]
    song = Song(bad_notes, durations, note_distribution_file)
    song.play(pred)
