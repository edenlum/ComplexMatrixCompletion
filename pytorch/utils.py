import itertools

def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = kwargs.keys()
    value_lists = kwargs.values()

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))