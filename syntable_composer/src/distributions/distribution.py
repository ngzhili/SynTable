
from abc import ABC, abstractmethod


class Distribution:

    # Static variables
    mount = None
    nucleus_server = None
    param_suffix_to_file_type = None

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def verify_args(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_type(self):
        pass
