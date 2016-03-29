from toposort import toposort_flatten

# Extensions:
#  - automatically parallelize nodes on the same "level"
#  - display a computational graph -- d3? Matplotlib?

class Stairway:
    """
    Library for functional data processing/transformation. Basically,
    allows a user to provide "curried" functions in python that
    transforms the data in a very stream-type environment.
    """

    def __init__(self, verbose=True):
        """
        Create our Stairway instance. For now, we assume that the user
        wants to load the data from a file, so the type of loader 
        (after currying) is FilePath -> Data.
        """
        self.graph, self.stairs = {}, {}
        self.verbose = verbose

    def step(self, name, deps, f, *args, **kwargs):
        """
        Function to add a step to the computational graph.
          deps [String]: a list of nodes in the graph that the function depends on
          f [Function]: node to execute
          args [List]: list of arguments to be passed into the function
        """
        # create a new stair for this function
        stair = Stair(name, deps, f,  *args, **kwargs)
        # update our computational graph with this stair
        self.stairs[name] = stair
        # check that we have the dependencies in the, and assume any
        # that aren't are inputs
        for dep in deps:
            # first check that the variable isn't already in the graph
            if dep in self.stairs: continue
            # create an input stair and enter it into our graph
            input_stair = Stair(dep, [], lambda v: v)
            self.stairs[dep] = input_stair
            self.graph[input_stair] = set([])
        # add our dependencie to the graph
        self.graph[stair] = set([self.stairs[dep] for dep in deps])
        # ensure that our new graph compiles
        self.slist = toposort_flatten(self.graph)
        # return self for chaining
        return self
    
    def process(self, **kwargs):
        """
        Actually perform the processing. This single function can be used
        over all inputs.
        """
        for s in self.slist:
            if self.verbose: print "Running step:", s.name
            if s.name in kwargs:
                s.args = tuple([kwargs[s.name]])
            s.run(self.graph[s])

        return self.slist[-1].results
    
class Stair:
    def __init__(self, name, deps, f, *args, **kwargs):
        self.name = name
        self.f = f
        self.deps = deps
        self.args = args
        self.kwargs = kwargs

    def run(self, deps):
        """
        Code to actually execute the function at the nodes. Make sure that the
        arguments passed in to deps is in the correct order!
        """
        lst = []
        for d in deps:
            lst.append(d.results)
        dep = lst + list(self.args)
        self.results = self.f(*dep, **self.kwargs)
    
    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name
