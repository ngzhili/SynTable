# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and itslicensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import argparse
from ntpath import join
import os
import shutil
import signal
import sys

from omni.isaac.kit import SimulationApp

config1 = {"headless": False} # zhili added: to display gui when generating scenes and dataset
kit = SimulationApp(config1) # minghao added

from distributions import Distribution
from input import Parser
from output import Metrics, Logger
from output.output1 import OutputManager
from sampling import Sampler
from scene.scene1 import SceneManager


class Composer:
    def __init__(self, params, index, output_dir):
        """ Construct Composer. Start simulator and prepare for generation. """

        self.params = params
        self.index = index
        self.output_dir = output_dir

        self.sample = Sampler().sample

        # Set-up output directories
        self.setup_data_output()

        # Start Simulator
        Logger.content_log_path = self.content_log_path
        Logger.start_log_entry("start-up")
        Logger.print("Isaac Sim starting up...")

        config = {"headless": self.sample("headless")}
        if self.sample("path_tracing"):
            config["renderer"] = "PathTracing"
            config["samples_per_pixel_per_frame"] = self.sample("samples_per_pixel_per_frame")
        else:
            config["renderer"] = "RayTracedLighting"

        #self.sim_app = SimulationApp(config) # minghao commented
        self.sim_app = kit # zhili added

        from omni.isaac.core import SimulationContext

        self.scene_units_in_meters = self.sample("scene_units_in_meters")
        self.sim_context = SimulationContext(physics_dt=0.5, #1.0 / 60.0, 
                                            stage_units_in_meters=self.scene_units_in_meters)
        # need to initialize physics getting any articulation..etc
        self.sim_context.initialize_physics()
        self.sim_context.play()

        self.num_scenes = self.sample("num_scenes")
        self.sequential = self.sample("sequential")

        self.scene_manager = SceneManager(self.sim_app, self.sim_context)
        self.output_manager = OutputManager(
            self.sim_app, self.sim_context, self.scene_manager, self.output_data_dir, self.scene_units_in_meters
        )

        # Set-up exit message
        signal.signal(signal.SIGINT, self.handle_exit)

        Logger.finish_log_entry()

    def handle_exit(self, *args, **kwargs):
        print("exiting dataset generation...")
        self.sim_context.clear_instance()
        self.sim_app.close()
        sys.exit()

    def generate_scene(self):
        """ Generate 1 dataset scene. Returns captured groundtruth data. """
        
        self.scene_manager.prepare_scene(self.index)

        self.scene_manager.populate_scene()
        
        # zhili added
        amodal1 = True
        amodal = False
        #self.scene_manager.print_instance_attributes()

        if self.sequential:
            sequence_length = self.sample("sequence_step_count")
            step_time = self.sample("sequence_step_time")
            for step in range(sequence_length):
                self.scene_manager.update_scene(step_time=step_time, step_index=step)
                groundtruth = self.output_manager.capture_groundtruth(
                    self.index, step_index=step, sequence_length=sequence_length
                )
                if step == 0:
                    Logger.print("stepping through scene...")
        
        # zhili added, iteratively hide objects
        elif amodal1:
            self.scene_manager.update_scene()
            
            # get entire scene
            groundtruth = \
                self.output_manager.capture_amodal_groundtruth(self.index, 
                                                               self.scene_manager) 




        elif amodal:
            self.scene_manager.update_scene()

            # print(len(self.scene_manager.objs))
            num_objects = len(self.scene_manager.objs)
            objects = self.scene_manager.objs
            groundtruths = []
            
            # get entire scene
            groundtruth = self.output_manager.capture_groundtruth(self.index) 

             # turn off visibility of all objects
            for i in range(num_objects):
                obj = objects[i]
                obj.off_prim()
                #print(obj.print_instance_attributes())

            # loop through objects and capture mask of each object
            for i in range(num_objects):
                obj = objects[i]
                obj.on_prim() # turn on object
                # capture instance segmentation of object j
                groundtruth = self.output_manager.capture_amodal_groundtruth(self.index, obj_index=i)    
                obj.off_prim() # turn off object

                #img_index = groundtruth["METADATA"]["image_id"]
                #groundtruth["METADATA"]["image_id"] = f"{img_index}_{i}"
                groundtruths.append(groundtruth)

            #print("\n\n==== PRINTING GROUNDTRUTHS ====\n\n")
            #print(groundtruths)
            # return
                    

        else:
            self.scene_manager.update_scene()
            groundtruth = self.output_manager.capture_groundtruth(self.index)

        self.scene_manager.finish_scene()

        return groundtruth

    def setup_data_output(self):
        """ Create output directories and copy input files to output. """

        # Overwrite output directory, if needed
        if self.params["overwrite"]:
            shutil.rmtree(self.output_dir, ignore_errors=True)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Create output directories, as needed
        self.output_data_dir = os.path.join(self.output_dir, "data")
        self.parameter_dir = os.path.join(self.output_dir, "parameters")
        self.parameter_profiles_dir = os.path.join(self.parameter_dir, "profiles")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.content_log_path = os.path.join(self.log_dir, "sampling_log.yaml")

        os.makedirs(self.output_data_dir, exist_ok=True)
        os.makedirs(self.parameter_profiles_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Copy input parameters file to output
        input_file_name = os.path.basename(self.params["file_path"])
        input_file_copy = os.path.join(self.parameter_dir, input_file_name)
        shutil.copy(self.params["file_path"], input_file_copy)

        # Copy profile parameters file(s) to output
        if self.params["profile_files"]:
            for profile_file in self.params["profile_files"]:
                profile_file_name = os.path.basename(profile_file)
                profile_file_copy = os.path.join(self.parameter_profiles_dir, profile_file_name)
                shutil.copy(profile_file, profile_file_copy)


def get_output_dir(params):
    """ Determine output directory. """

    if params["output_dir"].startswith("/"):
        output_dir = params["output_dir"]
    elif params["output_dir"].startswith("*"):
        output_dir = os.path.join(Distribution.mount, params["output_dir"][2:])
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", params["output_dir"])
    return output_dir


def get_starting_index(params, output_dir):
    """ Determine starting index of dataset. """

    if params["overwrite"]:
        return 0

    output_data_dir = os.path.join(output_dir, "data")
    if not os.path.exists(output_data_dir):
        return 0

    def find_min_missing(indices):
        if indices:
            indices.sort()
            for i in range(indices[-1]):
                if i not in indices:
                    return i
            return indices[-1]
        else:
            return -1

    camera_dirs = [os.path.join(output_data_dir, sub_dir) for sub_dir in os.listdir(output_data_dir)]

    min_indices = []
    for camera_dir in camera_dirs:
        data_dirs = [os.path.join(camera_dir, sub_dir) for sub_dir in os.listdir(camera_dir)]
        for data_dir in data_dirs:
            indices = []
            for filename in os.listdir(data_dir):
                try:
                    if "_" in filename:
                        index = int(filename[: filename.rfind("_")])
                    else:
                        index = int(filename[: filename.rfind(".")])
                    indices.append(index)
                except:
                    pass
            min_index = find_min_missing(indices)
            min_indices.append(min_index)

    if min_indices:
        minest_index = min(min_indices)
        return minest_index + 1
    else:
        return 0


def assert_dataset_complete(params, index):
    """ Check if dataset is already complete. """

    num_scenes = params["num_scenes"]
    if index >= num_scenes:
        print(
            'Dataset is completed. Number of generated samples {} satifies "num_scenes" {}.'.format(index, num_scenes)
        )
        sys.exit()
    else:
        print("Starting at index ", index)


def define_arguments():
    """ Define command line arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="parameters/warehouse.yaml", help="Path to input parameter file")
    parser.add_argument(
        "--visualize-models",
        "--visualize_models",
        action="store_true",
        help="Output visuals of all object models defined in input parameter file, instead of outputting a dataset.",
    )
    parser.add_argument("--mount", default="/tmp/composer", help="Path to mount symbolized in parameter files via '*'.")
    parser.add_argument("--headless", action="store_true", help="Will not launch Isaac SIM window.")
    parser.add_argument("--nap", action="store_true", help="Will nap Isaac SIM after the first scene is generated.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrites dataset in output directory.")
    parser.add_argument("--output", type=str, help="Output directory. Overrides 'output_dir' param.")
    parser.add_argument(
        "--num-scenes", "--num_scenes", type=int, help="Num scenes in dataset. Overrides 'num_scenes' param."
    )
    parser.add_argument(
        "--nucleus-server", "--nucleus_server", type=str, help="Nucleus Server URL. Overrides 'nucleus_server' param."
    )

    return parser


if __name__ == "__main__":
    # Create argument parser
    parser = define_arguments()
    args, _ = parser.parse_known_args()

    # Parse input parameter file
    parser = Parser(args)
    params = parser.params
    Sampler.params = params

    # Determine output directory
    output_dir = get_output_dir(params)

    # Run Composer in Visualize mode
    if args.visualize_models:
        from visualize import Visualizer

        visuals = Visualizer(parser, params, output_dir)
        visuals.visualize_models()

        # Handle shutdown
        visuals.composer.sim_context.clear_instance()
        visuals.composer.sim_app.close()
        sys.exit()

    # Set verbose mode
    Logger.verbose = params["verbose"]

    # Get starting index of dataset
    index = get_starting_index(params, output_dir)

    # Check if dataset is already complete
    assert_dataset_complete(params, index)

    # Initialize composer
    composer = Composer(params, index, output_dir)
    metrics = Metrics(composer.log_dir, composer.content_log_path)

    # Generate dataset
    while composer.index < params["num_scenes"]:
        composer.generate_scene()
        composer.index += 1

    # Handle shutdown
    composer.output_manager.data_writer.stop_threads()
    composer.sim_context.clear_instance()
    composer.sim_app.close()

    # Output performance metrics
    metrics.output_performance_metrics()
