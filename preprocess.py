from preprocessing.trainingdata import TrainingData
from preprocessing.filterbank import FilterBank

import cPickle as pickle
import numpy as np
import argparse
import os

def load_directory(directory, f):
    for item in os.listdir(directory):
        if item[0] != '.':
            current = directory + '/' + item
            if os.path.isdir(current):
                load_directory(current, f)
            if current[-4:] == '.wav':
                current = current[:-4]
                print "Loading and preprocessing training data,", current
                audio = TrainingData(current, f)
                pickle.dump(audio, open(current+'.pkl', 'wb'))
                del audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='recursive_top_dir')

    args = parser.parse_args()
    rdir = args.recursive_top_dir

    print "Constructing filterbank."
    f = FilterBank(44100, 0.05, 5)
    f.construct_bands(440.0, 20, 20)

    load_directory(rdir, f)
