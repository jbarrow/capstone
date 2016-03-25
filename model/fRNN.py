import numpy as np
import theano as theano
import theano.tensor as T
import operator
import sys
from datetime import datetime

sys.path.append('..')
import preprocessing.trainingdata

from dataset import DataSet

new_X = [
    np.array([[1], [0], [1], [1]], dtype=theano.config.floatX),
    np.array([[0], [0], [1]], dtype=theano.config.floatX)
]

new_y = [
    np.array([[0], [1], [1], [0]], dtype=theano.config.floatX),
    np.array([[0], [0], [1]], dtype=theano.config.floatX)
]

val_X = [
    np.array([[0], [1], [0], [1], [0]], dtype=theano.config.floatX),
    np.array([[1], [1], [0], [0]], dtype=theano.config.floatX)
]

val_y = [
    np.array([[0], [1], [1], [1], [1]], dtype=theano.config.floatX),
    np.array([[0], [0], [1], [0]], dtype=theano.config.floatX)
]

class RNN:
    def __init__(self, input_dim, output_dim, hidden_dim=100, bptt_truncate=4):
        # assign instance variables
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # theano: create shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        # we store the theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.matrix('x')
        y = T.matrix('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U.dot(x_t) + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True
        )

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # gradients
        dU, dV, dW = T.grad(o_error, [U, V, W])

        print "Compiling functions..."
        # assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x, y, learning_rate], [],
                    updates=[(self.U, self.U - learning_rate * dU),
                             (self.V, self.V - learning_rate * dV),
                             (self.W, self.W - learning_rate * dW)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        num_samples = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_samples)


def train_with_sgd(model, X_train, y_train, X_val, y_val, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_val, y_val)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        for i in range(len(y_train)):
            #print model.bptt(X_train[i], y_train[i])
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    return losses

np.random.seed(10)
d = DataSet('../data/ISOL/NO')
model = RNN(6616, 88, 100, bptt_truncate=1000)

losses = train_with_sgd(model, d.X[d.train], d.y[d.train], d.X[d.validation], d.y[d.validation], nepoch=100, evaluate_loss_after=1)

def gradient_check_theano(model, x, y, h=0.000001, error_threshold=0.05):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    print model.U.get_value(), model.V.get_value(), model.W.get_value()
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    print bptt_gradients
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)


# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
#np.random.seed(10)
#model = RNN(2, 2, 4, bptt_truncate=1000)
#gradient_check_theano(model, d.X[0], d.y[0])
