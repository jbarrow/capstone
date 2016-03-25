import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

from dataset import DataSet

sys.path.append('..')

import preprocessing.trainingdata

def main():
    # load our test, train, and validation data
    d = DataSet('../data/ISOl/RE')

    # build our network model
    print "Building Network"
    l_in = lasagne.layers.InputLayer(shape=(d.batch_size, d.max_seq, d.input_size))
    l_mask = lasagne.layers.InputLayer(shape=(d.batch_size, d.max_seq))
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, 100, mask_input=l_mask, grad_clipping=100,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True
    )
    #l_backward = lasagne.layers.RecurrentLayer(
    #    l_in, 100, mask_input=l_mask, grad_clipping=100,
    #    W_in_to_hid=lasagne.init.HeUniform(),
    #    W_hid_to_hid=lasagne.init.HeUniform(),
    #    nonlinearity=lasagne.nonlinearities.tanh,
    #    only_return_final=True, backwards=True
    #)
    #l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
    l_out = lasagne.layers.DenseLayer(
        l_forward, num_units=88, nonlinearity=lasagne.nonlinearities.softmax
    )

    target_values = T.matrix('target_output')
    
    network_output = lasagne.layers.get_output(l_out)
    predicted_values = network_output.flatten()
    cost = T.mean((predicted_values - target_values)**2)
    all_params = lasagne.layers.get_all_params(l_out)

    print "Computing updates ..."
    updates = lasagne.updates.adagrad(cost, all_params, 0.01)

    print "Compiling functions ..."
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)

    print "Training ..."
    try:
        for epoch in range(100):
            for _ in range(5):
                train(d.X, d.y, d.mask)
            cost_val = compute_cost(d.X, d.y, d.mask)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass
    

#if __name__ == '__main__':
main()
