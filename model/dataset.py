import cPickle as pickle
import numpy as np
import theano
import os, sys

sys.path.append('..')
import preprocessing.trainingdata

# TODO: Split train set into batches
# TODO: Remove the filterbank

class DataSet:
    def __init__(self, dir):
        self.input_size = 41
        self.output_size = 88
        self.X, self.y = self.load_files(dir)
        self.X, self.y = np.array(self.X), np.array(self.y)
        
        self.train, self.validation, self.test = self.split()
        #self.rectify_data()

        self.batch_size = len(self.train)
        self.max_seq = self.max_length(np.arange(len(self.X)))

    def load_files(self, dir):
        """
        load_files: Function to recursively load all the training data.

        DataSet -> FilePath -> Tuple ([Data], [Label])
        """
        X, y = [], []
        for item in os.listdir(dir):
            # don't want to dig into any system directories
            if item[0] is '.':
                continue
            current = dir + '/' + item
            # we know we have to make a recursive jump
            if os.path.isdir(current):
                Xs, ys = load_files(current)
                # add the results to our current X and y's
                X.extend(Xs) ; y.extend(ys)
            elif current[-4:] == '.pkl':
                # we've hit a node
                audio = pickle.load(open(current, 'rb'))
                X.append(np.asarray(audio.X, dtype=theano.config.floatX))
                y.append(np.asarray(audio.Y, dtype=theano.config.floatX))
        return X, y

    def split(self):
        """
        split: Return a list of indices representing the training (60%),
          validation (20%), and testing (20%) sets.
        
        DataSet -> Tuple ([Int], [Int], [Int])
        """
        # create a list of indices of each data element
        ind = np.arange(len(self.X))
        # shuffle the list
        np.random.shuffle(ind)
        # get ranges for each set
        tr, te = round(len(ind) * 0.6), round(len(ind) * 0.2)
        # return (train, validation, test)
        return (ind[0:tr], ind[tr:tr+te], ind[tr+te:-1])

    def max_length(self, indices):
        """
        max_length: Return the length of the longest sequence in
          the specified dataset.

        DataSet -> [Int] -> Int
        """
        return max([d.shape[0] for d in self.X[indices]])

    def min_length(self, indices):
        """
        min_length: Return the length of the shortest sequence in
          the specified dataset.

        DataSet -> [Int] -> Int
        """
        return min([d.shape[0] for d in self.X[indices]])
        
    def rectify_data(self):
        """
        rectify_data: Create a mask and rectify the data into one big
          numpy array.

        DataSet -> None
        """
        # get the max length of a sequence
        m = self.max_length(np.arange(len(self.X)))
        # create a new numpy array to hold all that data
        X = np.zeros((len(self.X), m, self.input_size))
        y = np.zeros((len(self.y), m, self.output_size))
        mask = np.zeros((len(self.X), m))
        # populate our X, y, and mask arrays
        for i in range(len(self.X)):
            X[i, :len(self.X[i]), :] = self.X[i]
            y[i, :len(self.y[i]), :] = self.y[i]
            mask[i, :len(self.X[i])] = 1
        # update the fields to represent the new array
        self.X = X.astype(theano.config.floatX)
        self.y = y.astype(theano.config.floatX)
        self.mask = mask.astype(theano.config.floatX)
