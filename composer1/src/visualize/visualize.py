# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont

from distributions import Choice, Walk
from main import Composer
from sampling import Sampler


class Visualizer:
    """ For generating visuals of each input object model in the input parameterization. """

    def __init__(self, parser, input_params, output_dir):
        """ Construct Visualizer. Parameterize Composer to generate the data needed to post-process into model visuals. """

        self.parser = parser
        self.input_params = input_params
        self.output_dir = os.path.join(output_dir, "visuals")
        os.makedirs(self.output_dir, exist_ok=True)

        # Get all object models from input parameter file
        self.obj_models = self.get_all_obj_models()
        self.nucleus_server = self.input_params["nucleus_server"]

        # Copy model list to output file
        model_list = os.path.join(self.output_dir, "models.txt")
        with open(model_list, "w") as f:
            for obj_model in self.obj_models:
                f.write(obj_model)
                f.write("\n")

        # Filter obj models
        if not self.input_params["overwrite"]:
            self.filter_obj_models(self.obj_models)
            if not self.obj_models:
                print("All object model visuals are already created.")
                sys.exit()

        self.tile_width = 500
        self.tile_height = 500
        self.obj_size = 1
        self.room_size = 10 * self.obj_size
        self.cam_distance = 4 * self.obj_size
        self.camera_coord = np.array((-self.cam_distance, 0, self.room_size / 2))
        self.background_color = (160, 185, 190)
        self.group_name = "photoshoot"

        # Set hard-coded parameters
        self.params = {self.group_name: {}}
        self.set_obj_params()
        self.set_light_params()
        self.set_room_params()
        self.set_cam_params()
        self.set_other_params()

        # Parse parameters
        self.params = parser.parse_input(self.params, parse_from_file=False)

        # Set parameters
        Sampler.params = self.params

        # Initiate Composer
        self.composer = Composer(self.params, 0, self.output_dir)

    def visualize_models(self):
        """ Generate samples and post-process captured data into visuals. """

        num_models = len(self.obj_models)
        for i, obj_model in enumerate(self.obj_models):
            print("Model {}/{} - {}".format(i, num_models, obj_model))

            self.set_obj_model(obj_model)

            # Capture 4 angles per model
            outputs = [self.composer.generate_scene() for j in range(4)]
            image_matrix = self.process_outputs(outputs)
            self.save_visual(obj_model, image_matrix)

    def get_all_obj_models(self):
        """ Get all object models from input parameterization. """

        obj_models = []
        groups = self.input_params["groups"]
        for group_name, group in groups.items():
            obj_count = group["obj_count"]
            group_models = group["obj_model"]
            if group_models and obj_count:
                if type(group_models) is Choice or type(group_models) is Walk:
                    group_models = group_models.elems
                else:
                    group_models = [group_models]

                obj_models.extend(group_models)

        # Remove repeats
        obj_models = list(set(obj_models))

        return obj_models

    def filter_obj_models(self, obj_models):
        """ Filter out obj models that have already been visualized. """

        existing_filenames = set([f for f in os.listdir(self.output_dir)])

        for obj_model in obj_models:
            filename = self.model_to_filename(obj_model)
            if filename in existing_filenames:
                obj_models.remove(obj_model)

    def model_to_filename(self, obj_model):
        """ Map object model's Nucleus path to a filename. """

        filename = obj_model.replace("/", "__")
        r_index = filename.rfind(".")
        filename = filename[:r_index]
        filename += ".jpg"

        return filename

    def process_outputs(self, outputs):
        """ Tile output data from scene into one image matrix. """

        rgbs = [groundtruth["DATA"]["RGB"] for groundtruth in outputs]
        wireframes = [groundtruth["DATA"]["WIREFRAME"] for groundtruth in outputs]

        rgbs = [rgb[:, :, :3] for rgb in rgbs]
        top_row_matrix = np.concatenate(rgbs, axis=1)

        wireframes = [wireframe[:, :, :3] for wireframe in wireframes]
        bottom_row_matrix = np.concatenate(wireframes, axis=1)

        image_matrix = np.concatenate([top_row_matrix, bottom_row_matrix], axis=0)
        image_matrix = np.array(image_matrix, dtype=np.uint8)

        return image_matrix

    def save_visual(self, obj_model, image_matrix):
        """ Save image matrix as image. """

        image = Image.fromarray(image_matrix, "RGB")

        font_path = os.path.join(os.path.dirname(__file__), "RobotoMono-Regular.ttf")
        font = ImageFont.truetype(font_path, 24)

        draw = ImageDraw.Draw(image)
        width, height = image.size
        draw.text((10, 10), obj_model, font=font)

        model_name = self.model_to_filename(obj_model)
        filename = os.path.join(self.output_dir, model_name)
        image.save(filename, "JPEG", quality=90)

    def set_cam_params(self):
        """ Set camera parameters. """

        self.params["camera_coord"] = str(self.camera_coord.tolist())
        self.params["camera_rot"] = str((0, 0, 0))
        self.params["focal_length"] = 50

    def set_room_params(self):
        """ Set room parameters. """

        self.params["scenario_room_enabled"] = str(True)

        self.params["floor_size"] = str(self.room_size)
        self.params["wall_height"] = str(self.room_size)

        self.params["floor_color"] = str(self.background_color)
        self.params["wall_color"] = str(self.background_color)
        self.params["ceiling_color"] = str(self.background_color)
        self.params["floor_reflectance"] = str(0)
        self.params["wall_reflectance"] = str(0)
        self.params["ceiling_reflectance"] = str(0)

    def set_obj_params(self):
        """ Set object parameters. """

        group = self.params[self.group_name]
        group["obj_coord_camera_relative"] = str(False)
        group["obj_rot_camera_relative"] = str(False)
        group["obj_coord"] = str((0, 0, self.room_size / 2))
        group["obj_rot"] = "Walk([(25, -25, -45), (-25, 25, -225), (-25, 25, -45), (25, -25, -225)])"
        group["obj_size"] = str(self.obj_size)
        group["obj_count"] = str(1)

    def set_light_params(self):
        """ Set light parameters. """

        group = self.params[self.group_name]
        group["light_count"] = str(4)
        group["light_coord_camera_relative"] = str(False)
        light_offset = 2 * self.obj_size
        light_coords = [
            self.camera_coord + (0, -light_offset, 0),
            self.camera_coord + (0, 0, light_offset),
            self.camera_coord + (0, light_offset, 0),
            self.camera_coord + (0, 0, -light_offset),
        ]
        light_coords = str([tuple(coord.tolist()) for coord in light_coords])
        group["light_coord"] = "Walk(" + light_coords + ")"
        group["light_intensity"] = str(40000)
        group["light_radius"] = str(0.50)
        group["light_color"] = str([200, 200, 200])

    def set_other_params(self):
        """ Set other parameters. """

        self.params["img_width"] = str(self.tile_width)
        self.params["img_height"] = str(self.tile_height)
        self.params["write_data"] = str(False)
        self.params["verbose"] = str(False)
        self.params["rgb"] = str(True)
        self.params["wireframe"] = str(True)

        self.params["nucleus_server"] = str(self.nucleus_server)

        self.params["pause"] = str(0.5)
        self.params["path_tracing"] = True

    def set_obj_model(self, obj_model):
        """ Set obj_model parameter. """

        group = self.params["groups"][self.group_name]
        group["obj_model"] = str(obj_model)
