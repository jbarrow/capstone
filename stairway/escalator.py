from stairway import Stairway

import cPickle as pickle
import functools
import glob
import os

@profile
def fmap(*data, **kwargs):
    """
    Function to map over the data. Because datasets can be astronomically large,
    we clear the data out of memory as soon as we use it, and save it in a tmp
    directory.
    """
    f, args, name = kwargs['f'], kwargs['args'], kwargs['name']
  
    return_data = []
    for i, d in enumerate(data[0]):
        res = None
        if type(d) == list or type(d) == tuple:
            res = f(*data[1:], **dict(zip(args, d)))
        else:
            res = f(*data[1:], **dict(zip(args, [d])))
        # save the data in a temporary file
        fname = 'tmp/{0}_{1}.pkl'.format(name, i)
        return_data.append(fname)
        with open(fname, 'wb') as pf:
            pickle.dump(res, pf)
        del res
    return return_data

@profile
def freduce(data, f, initializer):
    """
    Function to reduce the mapped data.
    """
    for i, d in enumerate(data):
        with open(d, 'rb') as pf:
            data = pickle.load(pf)
            initializer = f(initializer, data)

    return initializer

class Escalator:
    """
    Soon to be MapReduce framework for Stairway -- returns a new computational graph
    that maps over an entire dataset.
    """

    def __init__(self, loader, inputs):
        self.graph = Stairway()
        self.graph.step('loader', inputs, loader)

    def mapper(self, f, args, name='map', deps=['loader']):
        fmap_prime = functools.partial(fmap, name=name, f=f, args=args)
        self.graph.step(name, deps, fmap_prime)
        return self

    def reducer(self, r, initializer, args, name='reduce', deps=['map']):
        self.graph.step(name, deps, freduce, r, initializer)
        return self

    def start(self, **kwargs):
        # run the graph
        retval = self.graph.process(**kwargs)
        # remove the temporary files
        tmp_files = glob.glob('tmp/*.pkl')
        for f in tmp_files:
            os.remove(f)
        # return the overall return values
        return retval
