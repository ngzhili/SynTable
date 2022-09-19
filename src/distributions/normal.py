# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

from distributions import Distribution


class Normal(Distribution):
    """ For sampling a Gaussian. """

    def __init__(self, mean, var, min=None, max=None):
        """ Construct Normal distribution. """

        self.mean = mean
        self.var = var
        self.min_val = min
        self.max_val = max

    def __repr__(self):
        return "Normal(name={}, mean={}, var={}, min_bound={}, max_bound={})".format(
            self.name, self.mean, self.var, self.min_val, self.max_val
        )

    def setup(self, name):
        """ Parse input arguments. """

        self.name = name

        self.std_dev = np.sqrt(self.var)

        self.verify_args()

    def verify_args(self):
        """ Verify input arguments. """

        def verify_arg_i(mean, var, min_val, max_val):
            """ Verify number values. """

            if type(mean) not in (int, float):
                raise ValueError(repr(self) + " has incorrect mean type.")
            if type(var) not in (int, float):
                raise ValueError(repr(self) + " has incorrect variance type.")
            if var < 0:
                raise ValueError(repr(self) + " must have non-negative variance.")
            if min_val != None and type(min_val) not in (int, float):
                raise ValueError(repr(self) + " has incorrect min type.")
            if max_val != None and type(max_val) not in (int, float):
                raise ValueError(repr(self) + " has incorrect max type.")

            return True

        valid = False
        if type(self.mean) in (tuple, list) and type(self.var) in (tuple, list):
            if len(self.mean) != len(self.var):
                raise ValueError(repr(self) + " must have mean and variance with same length.")
            if self.min_val and len(self.min_val) != len(self.mean):
                raise ValueError(repr(self) + " must have mean and min bound with same length.")
            if self.max_val and len(self.max_val) != len(self.mean):
                raise ValueError(repr(self) + " must have mean and max bound with same length.")

            valid = all(
                [
                    verify_arg_i(
                        self.mean[i],
                        self.var[i],
                        self.min_val[i] if self.min_val else None,
                        self.max_val[i] if self.max_val else None,
                    )
                    for i in range(len(self.mean))
                ]
            )
        else:
            valid = verify_arg_i(self.mean, self.var, self.min_val, self.max_val)

        if not valid:
            raise ValueError(repr(self) + " is invalid.")

    def sample(self):
        """ Sample from Gaussian. """

        sample = np.random.normal(self.mean, self.std_dev)
        if self.min_val is not None or self.max_val is not None:
            sample = np.clip(sample, a_min=self.min_val, a_max=self.max_val)
        return sample

    def get_type(self):

        if type(self.mean) in (tuple, list):
            return tuple
        else:
            return float
