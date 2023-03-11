
import numpy as np

from distributions import Distribution


class Uniform(Distribution):
    """ For sampling uniformly from a continuous range. """

    def __init__(self, min_val, max_val):
        """ Construct Uniform distribution."""

        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return "Uniform(name={}, min={}, max={})".format(self.name, self.min_val, self.max_val)

    def setup(self, name):
        """ Parse input arguments. """

        self.name = name

        self.verify_args()

    def verify_args(self):
        """ Verify input arguments. """

        def verify_args_i(min_val, max_val):
            """ Verify number values. """

            valid = False
            if type(min_val) in (int, float) and type(max_val) in (int, float):
                valid = min_val <= max_val
            return valid

        valid = False
        if type(self.min_val) in (tuple, list) and type(self.max_val) in (tuple, list):
            if len(self.min_val) != len(self.max_val):
                raise ValueError(repr(self) + " must have min and max with same length.")
            valid = all([verify_args_i(self.min_val[i], self.max_val[i]) for i in range(len(self.min_val))])
        else:
            valid = verify_args_i(self.min_val, self.max_val)

        if not valid:
            raise ValueError(repr(self) + " is invalid.")

    def sample(self):
        """ Sample from continuous range. """

        return np.random.uniform(self.min_val, self.max_val)

    def get_type(self):
        """ Get value type. """

        if type(self.min_val) in (tuple, list):
            return tuple
        else:
            return float
