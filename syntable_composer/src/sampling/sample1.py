import numpy as np

from distributions import Distribution
from output import Logger


class Sampler:
    """ For managing parameter sampling. """

    # Static variable of parameter set
    params = None

    def __init__(self, group=None):
        """ Construct a Sampler. Potentially set an associated group. """

        self.group = group

    def evaluate(self, val):
        """ Evaluate a parameter into a primitive. """

        if isinstance(val, Distribution):
            val = val.sample()
        elif isinstance(val, (list, tuple)):
            elems = val
            val = [self.evaluate(sub_elem) for sub_elem in elems]
            is_numeric = all([type(elem) == int or type(elem) == float for elem in val])
            if is_numeric:
                val = np.array(val, dtype=np.float32)

        return val

    def sample(self, key, group=None,tableBounds=None):
        """ Sample a parameter. """

        if group is None:
            group = self.group

        if key.startswith("obj") or key.startswith("light") and group:
            param_set = Sampler.params["groups"][group]
        else:
            param_set = Sampler.params
        
        if key in param_set:
            val = param_set[key]
        else:
            print('Warning key "{}" in group "{}" not found in parameter set.'.format(key, group))
            return None
        if key == "obj_coord" and group != "table" and tableBounds:
            min_val = tableBounds[0]
            max_val = tableBounds[1]
            val.min_val = min_val
            val.max_val = max_val
        val = self.evaluate(val)

        Logger.write_parameter(key, val, group=group)

        return val
