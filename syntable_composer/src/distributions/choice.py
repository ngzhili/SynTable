
import numpy as np
import os

from distributions import Distribution


class Choice(Distribution):
    """ For sampling from a list of elems. """

    def __init__(self, input, p=None, filter_list=None):
        """ Construct Choice distribution. """

        self.input = input
        self.p = p
        self.filter_list = filter_list
        if self.p:
            self.p = np.array(self.p)
            self.p = self.p / np.sum(self.p)

    def __repr__(self):
        return "Choice(name={}, input={}, p={}, filter_list={})".format(self.name, self.input, self.p, self.filter_list)

    def setup(self, name):
        """ Process input into a list of elems, with filter_list elems removed. """

        self.name = name
        self.valid_file_types = Distribution.param_suffix_to_file_type.get(self.name[self.name.rfind("_") + 1 :], [])

        self.elems = self.get_elem_list(self.input)
        if self.filter_list:
            filter_listed_elems = self.get_elem_list(self.filter_list)

            elem_set = set(self.elems)
            for elem in filter_listed_elems:
                if elem in elem_set:
                    self.elems.remove(self.elems)

        self.elems = self.unpack_elem_list(self.elems)

        self.verify_args()

    def verify_args(self):
        """ Verify elem list derived from input args. """

        if len(self.elems) == 0:
            raise ValueError(repr(self) + " has no elems.")

        if self.p != None:
            if len(self.elems) != len(self.p):
                raise ValueError(
                    repr(self)
                    + " must have equal num p weights '{}' and num elems '{}'".format(len(self.elems), len(self.p))
                )

        if len(self.elems) > 1:
            type_checks = []
            for elem in self.elems:
                if type(elem) in (int, float):
                    # Integer and Float equivalence
                    elem_types = [int, float]
                elif type(elem) in (tuple, list, np.ndarray):
                    # Tuple and List equivalence
                    elem_types = [tuple, list, np.ndarray]
                else:
                    elem_types = [type(elem)]

                type_check = type(self.elems[0]) in elem_types
                type_checks.append(type_check)

            all_elems_same_val_type = all(type_checks)

            if not all_elems_same_val_type:
                raise ValueError(repr(self) + " must have elems that are all the same value type.")

    def sample(self):
        """ Samples from the list of elems. """
        # print(self.__repr__())
        # print('len(self.elems):',len(self.elems))
        # print("self.elems:",self.elems)
        
        if self.elems:
            index = np.random.choice(len(self.elems), p=self.p)
            sample = self.elems[index]

            if type(sample) in (tuple, list):
                sample = np.array(sample)

            return sample
        else:
            return None

    def get_type(self):
        """ Get value type of elem list, which are all the same. """

        return type(self.elems[0])

    def get_elem_list(self, input):
        """ Process input into a list of elems. """

        elems = []
        if type(input) is str and input[-4:] == ".txt":
            input_file = input
            file_elems = self.parse_input_file(input_file)
            elems.extend(file_elems)
        elif type(input) is list:
            for elem in input:
                list_elems = self.get_elem_list(elem)
                elems.extend(list_elems)
        else:
            elem = input
            if type(elem) in (tuple, list):
                elem = np.array(elem)
            elems.append(input)

        return elems

    def parse_input_file(self, input_file):
        """ Parse an input file into a list of elems. """

        if input_file.startswith("/"):
            input_file = input_file
        elif input_file.startswith("*"):
            input_file = os.path.join(Distribution.mount, input_file[2:])
        else:
            input_file = os.path.join(os.path.dirname(__file__), "../../", input_file)

        if not os.path.exists(input_file):
            raise ValueError(repr(self) + " is unable to find file '{}'".format(input_file))

        with open(input_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            file_elems = []
            for elem in lines:
                if elem and not elem.startswith("#"):
                    try:
                        elem = eval(elem)
                        if type(elem) in (tuple, list):
                            try:
                                elem = np.array(elem, dtype=np.float32)
                            except:
                                pass
                    except Exception as e:
                        pass
                    file_elems.append(elem)
            return file_elems

    def unpack_elem_list(self, elems):
        """ Unpack all potential Nucleus server directories referenced in the parameter values. """

        all_unpacked_elems = []
        for elem in elems:
            unpacked_elems = [elem]
            if type(elem) is str:
                if not elem.startswith("/"):
                    raise ValueError(repr(self) + " with path elem '{}' must start with a forward slash.".format(elem))
                directory_elems = self.get_directory_elems(elem)
                if directory_elems:
                    directory = elem
                    unpacked_elems = self.unpack_directory(directory_elems, directory)

                # if "." in elem:
                #     file_type = elem[elem.rfind(".") :].lower()
                #     if file_type not in self.valid_file_types:
                #         raise ValueError(
                #             repr(self)
                #             + " has elem '{}' with incorrect file type. File type must be in '{}'.".format(
                #                 elem, self.valid_file_types
                #             )
                #         )

            all_unpacked_elems.extend(unpacked_elems)

        elems = all_unpacked_elems

        return elems

    def unpack_directory(self, directory_elems, directory):
        """ Unpack a directory on Nucleus into a list of file paths. """

        unpacked_elems = []
        for directory_elem in directory_elems:
            directory_elem = os.path.join(directory, directory_elem)
            file_type = directory_elem[directory_elem.rfind(".") :].lower()
            if file_type in self.valid_file_types:
                elem = os.path.join(directory, directory_elem)
                unpacked_elems.append(elem)
            else:
                sub_directory_elems = self.get_directory_elems(directory_elem)
                if sub_directory_elems:
                    # Recurse on subdirectories
                    unpacked_elems.extend(self.unpack_directory(sub_directory_elems, directory_elem))

        return unpacked_elems

    def get_directory_elems(self, elem):
        """ Grab files in a potential Nucleus server directory. """

        import omni.client

        elem_can_be_nucleus_dir = "." not in os.path.basename(elem)
        if elem_can_be_nucleus_dir:
            (_, directory_elems) = omni.client.list(self.nucleus_server + elem)
            directory_elems = [str(elem.relative_path) for elem in directory_elems]
            return directory_elems
        else:
            return ()
