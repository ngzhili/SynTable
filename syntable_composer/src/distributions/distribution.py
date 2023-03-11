# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
