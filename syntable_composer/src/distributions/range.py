# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

from distributions import Distribution


class Range(Distribution):
    """ For sampling from a range of integers. """

    def __init__(self, min_val, max_val):
        """ Construct Range distribution. """

        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return "Range(name={}, min={}, max={})".format(self.name, self.min_val, self.max_val)

    def setup(self, name):
        """ Parse input arguments. """

        self.name = name

        self.range = range(self.min_val, self.max_val + 1)

        self.verify_args()

    def verify_args(self):
        """ Verify input arguments. """

        def verify_args_i(min_val, max_val):
            """ Verify number values. """

            valid = False
            if type(min_val) is int and type(max_val) is int:
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
        """ Sample from discrete range. """

        return np.random.choice(self.range)

    def get_type(self):
        """ Get value type. """

        if type(self.min_val) in (tuple, list):
            return tuple
        else:
            return int
