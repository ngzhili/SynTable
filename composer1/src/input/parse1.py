# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import numpy as np
import os
import yaml

from distributions import Distribution, Choice, Normal, Range, Uniform, Walk


class Parser:
    """ For parsing the input parameterization to Composer. """

    def __init__(self, args):
        """ Construct Parser. Parse input file. """

        self.args = args
        self.global_group = "[[global]]"
        self.param_suffix_to_file_type = {
            "model": [".usd", ".usdz", ".usda", ".usdc"],
            "texture": [".png", ".jpg", ".jpeg", ".hdr", ".exr"],
            "material": [".mdl"],
        }
        self.no_eval_check_params = {"output_dir", "nucleus_server", "inherit", "profiles"}
        Distribution.mount = args.mount
        Distribution.param_suffix_to_file_type = self.param_suffix_to_file_type

        self.default_params = self.parse_param_set("parameters/profiles/default1.yaml", default=True)
        additional_params_to_default_set = {"inherit": "", "profiles": [], "file_path": "", "profile_files": []}
        self.default_params = {**additional_params_to_default_set, **self.default_params}
        self.initialize_params(self.default_params)

        self.params = self.parse_input(self.args.input)

    def evaluate_param(self, key, val):
        """ Evaluate a parameter value in Python """

        # Skip evaluation on certain parameter with string values
        if not self.param_is_evaluated(key, val):
            return val

        if type(val) is str and len(val) > 0:
            val = eval(val)
            if type(val) in (tuple, list):
                try:
                    val = np.array(val, dtype=np.float32)
                except:
                    pass

        if isinstance(val, Distribution):
            val.setup(key)

        if type(val) in (tuple, list):
            elems = val
            val = [self.evaluate_param(key, sub_elem) for sub_elem in elems]

        return val

    def param_is_evaluated(self, key, val):
        if type(val) is np.ndarray:
            return True

        return not (key in self.no_eval_check_params or not val or (type(val) is str and val.startswith("/")))

    def initialize_params(self, params, default=False):
        """ Evaluate parameter values in Python. Verify parameter name and value type. """

        for key, val in params.items():
            if type(val) is dict:
                self.initialize_params(val)
            else:
                # Evaluate parameter
                try:
                    val = self.evaluate_param(key, val)
                    params[key] = val
                except Exception:
                    raise ValueError("Unable to evaluate parameter '{}' with value '{}'".format(key, val))

                # Verify parameter
                if not default:
                    if key.startswith("obj") or key.startswith("light"):
                        default_param_set = self.default_params["groups"][self.global_group]
                    else:
                        default_param_set = self.default_params

                    # Verify parameter name
                    if key not in default_param_set and key:
                        raise ValueError("Parameter '{}' is not a parameter.".format(key))

                    # Verify parameter value type
                    default_val = default_param_set[key]
                    if isinstance(val, Distribution):
                        val_type = val.get_type()
                    else:
                        val_type = type(val)

                    if isinstance(default_val, Distribution):
                        default_val_type = default_val.get_type()
                    else:
                        default_val_type = type(default_val)

                    if default_val_type in (int, float):
                        # Integer and Float equivalence
                        default_val_type = [int, float]
                    elif default_val_type in (tuple, list, np.ndarray):
                        # Tuple, List, and Array equivalence
                        default_val_type = [tuple, list, np.ndarray]
                    else:
                        default_val_type = [default_val_type]

                    if val_type not in default_val_type:
                        raise ValueError(
                            "Parameter '{}' has incorrect value type {}. Value type must be in {}.".format(
                                key, val_type, default_val_type
                            )
                        )

    def verify_nucleus_paths(self, params):
        """ Verify parameter values that point to Nucleus server file paths. """

        import omni.client

        for key, val in params.items():
            if type(val) is dict:
                self.verify_nucleus_paths(val)

            # Check Nucleus server file path of certain parameters
            elif key.endswith(("model", "texture", "material")) and not isinstance(val, Distribution) and val:
                # Check path starts with "/"
                if not val.startswith("/"):
                    raise ValueError(
                        "Parameter '{}' has path '{}' which must start with a forward slash.".format(key, val)
                    )

                # Check file type
                param_file_type = val[val.rfind(".") :].lower()
                correct_file_types = self.param_suffix_to_file_type.get(key[key.rfind("_") + 1 :], [])
                if param_file_type not in correct_file_types:
                    raise ValueError(
                        "Parameter '{}' has path '{}' with incorrect file type. File type must be one of {}.".format(
                            key, val, correct_file_types
                        )
                    )

                # Check file can be found
                file_path = self.nucleus_server + val
                (exists_result, _, _) = omni.client.read_file(file_path)
                is_file = exists_result.name.startswith("OK")

                if not is_file:
                    raise ValueError(
                        "Parameter '{}' has path '{}' not found on '{}'.".format(key, val, self.nucleus_server)
                    )

    def override_params(self, params):
        """ Override params with CLI args. """

        if self.args.output:
            params["output_dir"] = self.args.output
        if self.args.num_scenes is not None:
            params["num_scenes"] = self.args.num_scenes
        if self.args.num_views is not None: # added
            params["num_views"] = self.args.num_views
        if self.args.save_segmentation_data is not None: # added
            params["save_segmentation_data"] = self.args.save_segmentation_data
        if self.args.mount:
            params["mount"] = self.args.mount

        params["overwrite"] = self.args.overwrite
        params["headless"] = self.args.headless
        params["nap"] = self.args.nap
        params["visualize_models"] = self.args.visualize_models

    def parse_param_set(self, input, parse_from_file=True, default=False):
        """ Parse input parameter file. """

        if parse_from_file:
            # Determine parameter file path
            if input.startswith("/"):
                input_file = input
            elif input.startswith("*"):
                input_file = os.path.join(Distribution.mount, input[2:])
            else:
                input_file = os.path.join(os.path.dirname(__file__), "../../", input)

            # Read parameter file
            with open(input_file, "r") as f:
                params = yaml.safe_load(f)

            # Add a parameter for the input file path
            params["file_path"] = input_file
        else:
            params = input

        # Process parameter groups
        groups = {}
        groups[self.global_group] = {}
        for key, val in list(params.items()):
            # Add group
            if type(val) is dict:
                if key in groups:
                    raise ValueError("Parameter group name is not unique: {}".format(key))
                groups[key] = val
                params.pop(key)

            # Add param to global group
            if key.startswith("obj_") or key.startswith("light_"):
                groups[self.global_group][key] = val
                params.pop(key)

        params["groups"] = groups

        return params

    def parse_params(self, params):
        """ Parse params into a final parameter set. """

        import omni.client

        # Add a global group, if needed
        if self.global_group not in params["groups"]:
            params["groups"][self.global_group] = {}

        # Parse all profile parameter sets
        profile_param_sets = [self.parse_param_set(profile) for profile in params.get("profiles", [])[::-1]]

        # Set default as lowest param set and input file param set as highest
        param_sets = [copy.deepcopy(self.default_params)] + profile_param_sets + [params]

        # Union parameters sets
        final_params = param_sets[0]
        for params in param_sets[1:]:
            global_group_params = params["groups"][self.global_group]
            sub_global_group_params = final_params["groups"][self.global_group]

            for group in params["groups"]:
                if group == self.global_group:
                    continue
                group_params = params["groups"][group]
                if "inherit" in group_params:
                    inherited_group = group_params["inherit"]
                    if inherited_group not in final_params["groups"]:
                        raise ValueError(
                            "In group '{}' cannot find the inherited group '{}'".format(group, inherited_group)
                        )
                    inherited_params = final_params["groups"][inherited_group]
                else:
                    inherited_params = {}
                final_params["groups"][group] = {
                    **sub_global_group_params,
                    **inherited_params,
                    **global_group_params,
                    **group_params,
                }

            final_params["groups"][self.global_group] = {
                **final_params["groups"][self.global_group],
                **params["groups"][self.global_group],
            }

            final_groups = final_params["groups"].copy()
            final_params = {**final_params, **params}
            final_params["groups"] = final_groups

        # Remove non-final groups
        for group in list(final_params["groups"].keys()):
            if group not in param_sets[-1]["groups"]:
                final_params["groups"].pop(group)
        final_params["groups"].pop(self.global_group)

        params = final_params

        # Set profile file paths
        params["profile_files"] = [profile_params["file_path"] for profile_params in profile_param_sets]

        # Set Nucleus server and check connection
        if self.args.nucleus_server:
            params["nucleus_server"] = self.args.nucleus_server

        if "://" not in params["nucleus_server"]:
            params["nucleus_server"] = "omniverse://" + params["nucleus_server"]

        self.nucleus_server = params["nucleus_server"]
        (result, _) = omni.client.stat(self.nucleus_server)
        if not result.name.startswith("OK"):
            raise ConnectionError("Could not connect to the Nucleus server: {}".format(self.nucleus_server))

        Distribution.nucleus_server = params["nucleus_server"]

        # Initialize params
        self.initialize_params(params)

        # Verify Nucleus server paths
        self.verify_nucleus_paths(params)

        return params

    def parse_input(self, input, parse_from_file=True):
        """ Parse all input parameter files. """

        if parse_from_file:
            print("Parsing and checking input parameterization.")

        # Parse input parameter file
        params = self.parse_param_set(input, parse_from_file=parse_from_file)

        # Process params
        params = self.parse_params(params)

        # Override parameters with CLI args
        self.override_params(params)

        return params
