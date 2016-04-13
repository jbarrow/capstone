import scipy.io.wavfile
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.cm as cm

from stairway import Stairway
from stairway.steps import stft, cqt
from keras.models import model_from_json
from pomegranate import *

def load_model(f_base):
    model = model_from_json(open(f_base + '.json').read())
    model.load_weights(f_base + '.h5')
    return model
    
s = Stairway(False)\
    .step('load_audio', ['audio_file'], scipy.io.wavfile.read)\
    .step('stft', ['load_audio'], stft, 0.1, 0.025)

file_name = 'c scale one octave Classic Electric Piano.wav'
d = s.process(audio_file=file_name)
d_train = np.zeros((1,)+d.shape)
d_train[0] = d

print "Loading model..."
model = load_model('models/model_0')

print "Predicting with model..."
p0 = model.predict(d_train, batch_size=1)

prediction = p0

# Get probability of notes
y = np.zeros((89, len(prediction[0])))
for i, r in enumerate(prediction[0]):
    c = np.argmax(r)
    y[c, i] = 1.0
probNotes = np.mean(y, axis=1)

duration = 12.0
selfTransProb = 1. - (1./duration)
transProb = 1. - selfTransProb

# Distribution
probForNotes = {i:probNotes[i] for i in range(len(probNotes))}
distr = DiscreteDistribution(probForNotes)
# Create HMM
hmm = HiddenMarkovModel()
# Add States
states = [] # hmm.states returning None?
for i in range(len(probForNotes)):
	s = State(distr, name=str(i+1))
	hmm.add_state(s)
	states.append(s)
# Add Transition Probabilities
for s1 in states:
	hmm.add_transition(hmm.start, s1, transProb)
	for s2 in states:
		if s1 == s2: # self transition prob
			hmm.add_transition(s1, s2, selfTransProb)
		else: # transition prob
			hmm.add_transition(s1, s2, transProb)
hmm.bake()

# Find path via HMM
prediction = np.squeeze(prediction)
predictionIndices = np.argmax(prediction, axis=1)
logProb = hmm.log_probability(predictionIndices)
probPath = hmm.predict_proba(predictionIndices)
path = np.argmax(probPath, axis=1)
print "Path of states:"
print str(path)

# plt.imshow(y[:-1,:], cmap=cm.binary)
# plt.show()
