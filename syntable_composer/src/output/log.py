# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import datetime
import os
import time
import yaml


class Logger:
    """ For logging parameter samples and dataset generation metadata. """

    # Static variables set outside class
    verbose = None
    content_log_path = None

    def start_log_entry(index):
        """ Initialize a sample's log message. """

        Logger.start_time = time.time()
        Logger.log_entry = [{}]
        Logger.log_entry[0]["index"] = index
        Logger.log_entry[0]["metadata"] = {"params": [], "lines": []}
        Logger.log_entry[0]["metadata"]["timestamp"] = str(datetime.datetime.now())

        if Logger.verbose:
            print()

    def finish_log_entry():
        """ Output a sample's log message to the end of the content log. """

        duration = time.time() - Logger.start_time
        Logger.log_entry[0]["time_elapsed"] = duration

        if Logger.content_log_path:
            with open(Logger.content_log_path, "a") as f:
                yaml.safe_dump(Logger.log_entry, f)

    def write_parameter(key, val, group=None):
        """ Record a sample parameter value. """

        if key == "groups":
            return

        param_dict = {}
        param_dict["parameter"] = key
        param_dict["val"] = str(val)
        param_dict["group"] = group

        Logger.log_entry[0]["metadata"]["params"].append(param_dict)

    def print(line, force_print=False):
        """ Record a string and potentially output it to console. """

        Logger.log_entry[0]["metadata"]["lines"].append(line)

        if Logger.verbose or force_print:
            line = str(line)
            print(line)
