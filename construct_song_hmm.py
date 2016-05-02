import scipy.io.wavfile
import numpy as np
import h5py

from stairway import Stairway
from stairway.steps import stft
from keras.models import model_from_json
from pomegranate import *
from itertools import groupby
from operator import itemgetter

class NoteType:
    NONE = 0
    CORRECT = 1
    MISTAKE = 2

class Note:

    def __init__(self, order, index, note_type, duration=0):
        self.order = order
        self.index = index
        self.type = note_type
        self.duration = duration

    def __eq__(self, other):
        return (self.order == other.order) and \
               (self.index == other.index)

    # (order:index, duration)
    def __repr__(self):
        if self.type is NoteType.CORRECT:
            return "({}:{}, {})".format(self.order, self.index, self.duration)
        else:
            return "M({}:{}, {})".format(self.order, self.index, self.duration)

class MistakeType:
    NONE = 0
    DELETION = 1
    INSERTION = 2
    SUBSTITUTION = 3

class Mistake:

    def __init__(self, note, mistake_type):
        self.note = note
        self.type = mistake_type

    def __repr__(self):
        if self.type is MistakeType.DELETION:
            return "DELETION: {}".format(self.note)
        elif self.type is MistakeType.INSERTION:
            return "INSERTION: {}".format(self.note)
        elif self.type is MistakeType.SUBSTITUTION:
            return "SUBSTITUTION: {}".format(self.note)
        return "None mistake"

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
        # initialize the hidden Markov model for the score
        self.hmm = HiddenMarkovModel()
        self.add_note_states()
        self.add_mistake_states()
        self.add_mistake_transitions()
        self.add_note_transitions()
        self.hmm.bake()

    def add_note_states(self):
        for i, note in enumerate(self.notes):
            dist = DiscreteDistribution(dict(enumerate(self.note_prob[note])))
            state = State(dist, name='{}:{}'.format(i, note))
            self.hmm.add_state(state)
            self.note_states.append(state)

    def add_mistake_states(self):
        for n in range(len(self.notes)):
            notes_to_remove = [self.notes[n]]
            if n < len(self.notes)-1:
                notes_to_remove.append(self.notes[n+1])
            states = self.mistake_states_without_notes(notes_to_remove, n)
            self.mistake_states.append(states)
            self.hmm.add_states(states)

    def mistake_states_without_notes(self, notes_to_remove, note):
        notes = np.delete(range(self.note_prob.shape[0]), notes_to_remove)
        prob = np.delete(self.note_prob, notes_to_remove, axis=0)
        num_notes = prob.shape[0]
        distr = [DiscreteDistribution(dict(enumerate(prob[i]))) for i in range(num_notes)]
        states = [State(distr[i], name='m:{}:{}'.format(note, notes[i])) for i in range(num_notes)]
        return states

    def add_mistake_transitions(self):
        # transition probabilities
        p_mistake_to_mistake = 1. - 1. / np.mean(self.durations)
        p_mistake_to_note = (1. - p_mistake_to_mistake) / 2.
        p_note_to_mistake = (1. - self.p_correct) / len(self.mistake_states[0])
        for i in range(len(self.notes)):
            mistakes = self.mistake_states[i]
            # add mistake transitions
            note = self.note_states[i]
            for mistake in mistakes:
                self.hmm.add_transition(note, mistake, p_note_to_mistake)
                self.hmm.add_transition(mistake, note, p_mistake_to_note)
                self.hmm.add_transition(mistake, mistake, p_mistake_to_mistake)
            # transition from ith mistake states to (i+1)th note state
            if i < len(self.notes)-1:
                next_note = self.note_states[i+1]
                for mistake in mistakes:
                    self.hmm.add_transition(mistake, next_note, p_mistake_to_note)

    def add_note_transitions(self):
        for i in range(len(self.note_states)):
            duration = self.durations[i]
            # compute transition probabilities
            otp = 1. / duration # out transition probability
            stp = self.p_correct - otp # self transition probability
            state = self.note_states[i]
            # transition: self transition
            self.hmm.add_transition(state, state, stp)
            # transition: transition to next note
            if i < len(self.note_states)-1:
                next_state = self.note_states[i+1]
                self.hmm.add_transition(state, next_state, otp)
        # start & end note transitions
        self.hmm.add_transition(self.hmm.start, self.note_states[0], self.p_correct)
        self.hmm.add_transition(self.note_states[-1], self.hmm.end, 1. / self.durations[-1])

    def play(self, predictions):
        self.pred_indices = np.argmax(predictions, axis=1)
        prob_path = self.hmm.predict_proba(self.pred_indices)
        self.state_index_path = np.argmax(prob_path, axis=1)
        # self.state_path = [self.hmm.states[i] \
        #     for i in self.state_index_path]
        # self.performance = [state.name.split(':') \
        #     for state in self.state_path]
        self.state_path = [self.hmm.states[i] for i in self.state_index_path]
        string_path = [state.name.split(':') for state in self.state_path]
        grouped_string_path = [(i, len(list(g))) for i, g in groupby(string_path)]
        self.performance = [Note(int(i[0][1]), int(i[0][2]), NoteType.MISTAKE, int(i[1])) \
            if len(i[0]) > 2 else Note(int(i[0][0]), int(i[0][1]), NoteType.CORRECT, int(i[1])) \
            for i in grouped_string_path]
        self.print_performance()

    def print_performance(self):
        print "\tHMM prediction ('state index: note index', count):"
        # state_sum = ['{}:{}'.format(key, len(list(group))) \
        #     for key, group in groupby(self.performance)]
        # print "\t{}".format(state_sum)
        print "\t{}".format(self.performance)
        print "\tfor RNN prediction (note, count):"
        pred_sum = [(key, len(list(group))) \
            for key, group in groupby(self.pred_indices)]
        print "\t{}".format(pred_sum)

    def detect_mistakes(self):
        self.mistakes = []
        last_index = 0
        for i in range(len(self.performance)):
            note = self.performance[i]
            if note.duration is 1:
                # substitution
                if note.index is last_index:
                    if i < len(self.performance)-1:
                        next_note = self.performance[i+1]
                        self.mistakes.append(Mistake(next_note, MistakeType.SUBSTITUTION))
                else: # deletion
                    self.mistakes.append(Mistake(note, MistakeType.DELETION))
            # insertion
            if note.type is NoteType.MISTAKE and i > 0:
                prev_note = self.performance[i-1]
                if prev_note.duration > 1:
                    self.mistakes.append(Mistake(note, MistakeType.INSERTION))
            last_index = note.index
        self.print_mistakes()

    def print_mistakes(self):
        print "Mistakes:"
        print "\t{}".format(self.mistakes)

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

    print "Creating HMM..."
    note_distribution_file = 'note_distribution.h5'
    # correct performance
    correct_notes = [39, 41, 43, 44, 46]
    durations =     [10., 10., 10., 10., 10.]
    song = Song(correct_notes, durations, note_distribution_file)
    # insertion
    label1 =        [39, 43, 44, 46]
    durations1 =    [10., 10., 10., 10.]
    song1 = Song(label1, durations1, note_distribution_file)
    # double insertion
    label2 =        [39, 44, 46]
    durations2 =    [10., 10., 10.]
    song2 = Song(label2, durations2, note_distribution_file)
    # deletion
    label3 =        [39, 41, 35, 43, 44, 46]
    durations3 =    [10., 10., 10., 10., 10., 10.]
    song3 = Song(label3, durations3, note_distribution_file)
    # double deletion
    label4 =        [39, 41, 35, 32, 43, 44, 46]
    durations4 =    [10., 10., 10., 10., 10., 10., 10.]
    song4 = Song(label4, durations4, note_distribution_file)
    # substitution
    label5 =        [39, 30, 43, 44, 46]
    durations5 =    [10., 10., 10., 10., 10.]
    song5 = Song(label5, durations5, note_distribution_file)
    # double substitution
    label6 =        [39, 30, 35, 44, 46]
    durations6 =    [10., 10., 10., 10., 10.]
    song6 = Song(label6, durations6, note_distribution_file)

    print "Predicting with HMM..."
    print "\nSong: {}".format(correct_notes)
    print "Correct performance"
    song.play(pred)
    song.detect_mistakes()
    print "\nSong: {}".format(label1)
    print "Insertion of note 41"
    song1.play(pred)
    song1.detect_mistakes()
    print "\nSong: {}".format(label2)
    print "Insertion of note 41 and 43"
    song2.play(pred)
    song2.detect_mistakes()
    print "\nSong: {}".format(label3)
    print "Deletion of note 35"
    song3.play(pred)
    song3.detect_mistakes()
    print "\nSong: {}".format(label4)
    print "Deletion of note 35 and 32"
    song4.play(pred)
    song4.detect_mistakes()
    print "\nSong: {}".format(label5)
    print "Substitution of note 41 by 30"
    song5.play(pred)
    song5.detect_mistakes()
    print "\nSong: {}".format(label6)
    print "Substitution of note 41 by 30 and 43 by 35"
    song6.play(pred)
    song6.detect_mistakes()

