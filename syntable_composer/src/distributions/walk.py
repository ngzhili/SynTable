
import numpy as np

from distributions import Choice


class Walk(Choice):
    """ For sampling from a list of elems without replacement. """

    def __init__(self, input, filter_list=None, ordered=True):
        """ Constructs a Walk distribution. """

        super().__init__(input, filter_list=filter_list)

        self.ordered = ordered
        self.completed = False
        self.index = 0

    def __repr__(self):
        return "Walk(name={}, input={}, filter_list={}, ordered={})".format(
            self.name, self.input, self.filter_list, self.ordered
        )

    def setup(self, name):
        """ Parse input arguments. """

        self.name = name

        if not self.ordered:
            self.sampled_indices = list(range(len(self.elems)))

        super().setup(name)

    def sample(self):
        """ Samples from list of elems and updates the index tracker. """

        if self.ordered:
            self.index %= len(self.elems)
            sample = self.elems[self.index]
            self.index += 1
        else:
            if len(self.sampled_indices) == 0:
                self.sampled_indices = list(range(len(self.elems)))
            self.index = np.choice(self.sampled_indices)
            self.sampled_indices.remove(self.index)
            sample = self.elems[self.index]

        if type(sample) in (tuple, list):
            sample = np.array(sample)

        return sample
