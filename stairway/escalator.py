
from stairway import Stairway

def fmap(data, f, args):
    if len(args) > 1:
        return [f(**dict(zip(args, d))) for d in data]
    return [f(d) for d in data]

def freduce(data, f): return data

class Escalator:
    """
    Soon to be MapReduce framework for Stairway -- returns a new computational graph
    that maps over an entire dataset.
    """

    def __init__(self, loader, inputs):
        self.graph = Stairway()
        self.graph.step('loader', inputs, loader)

    def mapper(self, stairway, args):
        self.graph.step('map', ['loader'], fmap, stairway.process, args)
        return self

    def reducer(self, r):
        self.graph.step('reduce', ['map'], freduce, r)
        return self

    def start(self, **kwargs):
        self.graph.process(**kwargs)
    
