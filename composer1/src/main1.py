# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and itslicensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Zhili's Replicator Composer Main
"""

import argparse
from ntpath import join
import os
import shutil
import signal
import sys
import numpy as np
import random
import math
import gc
import json
import datetime
import time
import glob
import cv2

from omni.isaac.kit import SimulationApp
config1 = {"headless": False} # to display gui when generating scenes and dataset
# config1["renderer"] = "PathTracing" # enable path tracing
# config1["samples_per_pixel_per_frame"] = 32
kit = SimulationApp(config1)

from distributions import Distribution
from input.parse1 import Parser
from output import Metrics, Logger
from output.output1 import OutputManager
# from sampling import Sampler
from sampling.sample1 import Sampler
from scene.scene1 import SceneManager
from helper_functions import compute_occluded_masks
from omni.isaac.kit.utils import set_carb_setting
from scene.light1 import Light

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

        # self.sim_app = SimulationApp(config)
        self.sim_app = kit # zhili added

        from omni.isaac.core import SimulationContext

        self.scene_units_in_meters = self.sample("scene_units_in_meters")
        self.sim_context = SimulationContext(physics_dt=1.0/60, #1.0 / 60.0, 
                                            rendering_dt =1.0/60, #1.0 / 60.0, 
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

    def generate_scene(self, img_index, ann_index, img_list,ann_list):
        """ Generate 1 dataset scene. Returns captured groundtruth data. """
        amodal = True
        self.scene_manager.prepare_scene(self.index)

        if amodal:
            roomTableSize = self.scene_manager.roomTableSize
            roomTableHeight = roomTableSize[-1]
            spawnLowerBoundOffset = 0.2
            spawnUpperBoundOffset = 1

            # calculate tableBounds to constraint objects' spawn locations
            x_width = roomTableSize[0] /2 
            y_length = roomTableSize[1] /2
            # print(roomTableSize[0],roomTableSize[1],roomTableSize[-1])
            min_val = (-x_width*0.6, -y_length*0.6, roomTableHeight+spawnLowerBoundOffset)
            max_val = (x_width*0.6, y_length*0.6, roomTableHeight+spawnUpperBoundOffset)
            tableBounds = [min_val,max_val]
            self.scene_manager.populate_scene(tableBounds=tableBounds) # populate the scene once 
        else:
            self.scene_manager.populate_scene()
        

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
        elif amodal:
            self.scene_manager.update_scene() # simulate physical dropping of objects
            self.sim_context.render()
            self.sim_context.pause()
            
            # stop all object motion and remove objects not on tabletop
            objects = self.scene_manager.objs.copy() 
            objects_filtered = []
            # print("\nList of objects:",[obj.name for obj in self.scene_manager.objs])
            # print("Number of Objects:", len(self.scene_manager.objs))
            for obj in objects:
                obj.coord, quaternion = obj.xform_prim.get_world_pose()
                obj.coord = np.array(obj.coord, dtype=np.float32)
                # print(f"{obj.name} coord:{obj.coord}")
                # objMinBounds, objMaxBounds  = obj.get_bounds()
                # objSize = objMaxBounds - objMinBounds
                # print(f"{obj.name} size:{objSize}")

                # if object is not on tabletop after simulation, remove object
                if (abs(obj.coord[0]) > (roomTableSize[0]/2)) \
                    or (abs(obj.coord[1]) > (roomTableSize[1]/2)) \
                    or (abs(obj.coord[2]) < roomTableSize[2]):
                    print(f"\nRemoving {obj.name} not on tabletop with coords: {obj.coord}")
                    obj.off_prim()
                    # from omni.isaac.core.utils import prims
                    # prims.delete_prim(obj.path) # delete prim from scene
                    # self.scene_manager.objs.remove(obj) # remove from objects
                else:
                    objects_filtered.append(obj)
            self.scene_manager.objs = objects_filtered        
            # print("\nList of objects:",[obj.name for obj in self.scene_manager.objs])
            print("\nNumber of Objects on tabletop:", len(self.scene_manager.objs))
            # print('\n\nScene Manager:\n')
            # print(self.scene_manager.print_instance_attributes())
            # print('\n\nCamera:\n')
            # print(self.scene_manager.camera.print_instance_attributes())
            
            # create camera orbit
            def camera_orbit_coord(r = 12, tableTopHeight=10):
                """
                constraints camera loc to a hemi-spherical orbit around tabletop origin
                origin z of hemisphere is offset by tabletopheight + 1m
                """
                u = random.uniform(0,1)
                v = random.uniform(0,1)
                phi = math.acos(1.0 - v) # phi: [0,0.5*pi]
                theta = 2.0 * math.pi * u # theta: [0,2*pi]
                x	=	r * math.cos(theta) * math.sin(phi)	
                y	=	r * math.sin(theta) * math.sin(phi)	
                z	=	r * math.cos(phi) + tableTopHeight # add table height offset
                # print("radius r:",r)
                # print("phi:",phi)
                # print("theta:",theta)
                return np.array([x,y,z])

            # Randomly move camera and light coordinates to be constrainted between 2 concentric hemispheres above tabletop
            numViews = self.params["num_views"]
            
            # set_carb_setting(kit._carb_settings, rtx_mode + "/sceneDb/ambientLightIntensity", 0)

            Logger.print(f"\n=== Capturing Groundtruth for each viewport in scene ===\n")
            # for view_id, cam_pose in enumerate(cam_pose_list):
            for view_id in range(numViews):
                Logger.print(f"\n==> Scene: {self.index}, View: {view_id} <==\n")
                # Resample camera coordinates and rotate to look at tabletop surface center
                cam_coord_w = camera_orbit_coord(r=random.uniform(0.7,1.4),tableTopHeight=roomTableHeight+0.2)
                self.scene_manager.camera.translate(cam_coord_w)
                self.scene_manager.camera.translate_rotate(target=(0,0,roomTableHeight)) #target coordinates
                print(f"Camera Coordinate:{cam_coord_w}")

                rtx_mode = "/rtx"
                ambient_light_intensity = 0 #random.uniform(0.2,3.5)
                set_carb_setting(kit._carb_settings, rtx_mode + "/sceneDb/ambientLightIntensity", ambient_light_intensity)
                print(f"Ambient Light Intensity:{ambient_light_intensity}")
                
                # Enable indirect diffuse GI
                set_carb_setting(kit._carb_settings, rtx_mode + "/indirectDiffuse/enabled", True)
                
                # Reset and delete all lights
                from omni.isaac.core.utils import prims
                # print("self.scene_manager.lights",self.scene_manager.lights)
                for light in self.scene_manager.lights:
                    prims.delete_prim(light.path)
                
                # print("self.scene_manager.ceilinglights",self.scene_manager.ceilinglights)
                # for light in self.scene_manager.ceilinglights:
                #     # print(light.path)
                #     prims.delete_prim(light.path)

                # Resample number of lights in viewport
                print(f"\nResampling Lights in Scene:\n")
                self.scene_manager.lights = []
                # self.scene_manager.ceilinglights = []
                for grp_index, group in enumerate(self.scene_manager.sample("groups")):
                    # print("group:",group)
                    if group == "ceilinglights":
                        # num_lights = self.scene_manager.sample("light_count", group=group)
                        # for i in range(num_lights):
                        for lightIndex, light in enumerate(self.scene_manager.ceilinglights):
                            # path = "{}/Ceilinglights/ceilinglights_{}".format(self.scene_manager.scene_path, len(self.scene_manager.ceilinglights))
                            # light = Light(self.scene_manager.sim_app, self.scene_manager.sim_context, path, self.scene_manager.camera, group)
                            if lightIndex == 0:
                                new_intensity = light.sample("light_intensity")
                                if light.sample("light_temp_enabled"):
                                    new_temp = light.sample("light_temp")
                            
                            # change light intensity
                            light.attributes["intensity"] = new_intensity
                            light.prim.GetAttribute("intensity").Set(light.attributes["intensity"])
                            print("Ceiling Light Intensity:",light.attributes["intensity"])
                            
                            # change light temperature
                            if light.sample("light_temp_enabled"):
                                light.attributes["colorTemperature"] = new_temp
                                light.prim.GetAttribute("colorTemperature").Set(light.attributes["colorTemperature"])
                                print("Ceiling Light Temperature (Kelvins):",light.attributes["colorTemperature"])
                            print("Ceiling Light Coords:",light.coord)
                            print("\n")

                            # self.scene_manager.ceilinglights.append(light)

                            # rtx_mode = "/rtx"
                            # ambient_light_intensity = 0 #random.uniform(0.2,3.5)
                            # set_carb_setting(kit._carb_settings, rtx_mode + "/sceneDb/ambientLightIntensity", ambient_light_intensity)
                            # print(f"Ambient Light Intensity:{ambient_light_intensity}")

                    if group == "lights":
                        num_lights = self.scene_manager.sample("light_count", group=group)
                        for i in range(num_lights):
                            path = "{}/Lights/lights_{}".format( self.scene_manager.scene_path, len(self.scene_manager.lights))
                            light = Light(self.scene_manager.sim_app, self.scene_manager.sim_context, path, self.scene_manager.camera, group)
                            
                            # change light intensity
                            light.attributes["intensity"] = light.sample("light_intensity")
                            light.prim.GetAttribute("intensity").Set(light.attributes["intensity"])
                            print("Light Intensity:",light.attributes["intensity"])
                            
                            # change light temperature
                            if light.sample("light_temp_enabled"):
                                light.attributes["colorTemperature"] =light.sample("light_temp")
                                light.prim.GetAttribute("colorTemperature").Set(light.attributes["colorTemperature"])
                                print("Light Temperature (Kelvins):",light.attributes["colorTemperature"])
                            
                            # change light coordinates
                            light_coord_w = camera_orbit_coord(r=random.uniform(1.5,2.5),tableTopHeight=roomTableHeight+0.2)
                            light.translate(light_coord_w)
                            light.coord, quaternion = light.xform_prim.get_world_pose()

                            # from omni.isaac.core.utils import rotations
                            # print(rotations.quat_to_euler_angles(quaternion, degrees = True))
                            # print(quaternion)

                            light.coord = np.array(light.coord, dtype=np.float32)
                            print("Light Coords:",light.coord)
                            print("\n")

                            self.scene_manager.lights.append(light)
                        print(f"Number of sphere light in scene: {len(self.scene_manager.lights)}")

                # capture groundtruth of entire scene
                groundtruth, img_index, ann_index, img_list, ann_list = \
                    self.output_manager.capture_amodal_groundtruth(self.index, 
                                                               self.scene_manager,
                                                               img_index, ann_index, view_id,
                                                               img_list, ann_list
                                                               )             

        else:
            self.scene_manager.update_scene()
            groundtruth = self.output_manager.capture_groundtruth(self.index)

        self.scene_manager.finish_scene()

        return groundtruth, img_index, ann_index, img_list, ann_list

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
    # def find_min_missing(indices):
    #     if indices:
    #         indices.sort()
    #         for i in range(1,indices[-1]+1):
    #             if i not in indices:
    #                 return i
    #         return indices[-1] + 1
    #     else:
    #         return -1

    camera_dirs = [os.path.join(output_data_dir, sub_dir) for sub_dir in os.listdir(output_data_dir)]

    min_indices = []
    for camera_dir in camera_dirs:
        data_dirs = [os.path.join(camera_dir, sub_dir) for sub_dir in os.listdir(camera_dir)]
        for data_dir in data_dirs:
            # print(data_dir)
            indices = []
            for filename in os.listdir(data_dir):
                # print(filename)
                try:
                    if "_" in filename:
                        index = int(filename[: filename.rfind("_")])
                        # print("_",index)
                    else:
                        index = int(filename[: filename.rfind(".")])
                        # print(".",index)
                    indices.append(index)
                except:
                    pass
            
            # print("indices",indices)
            min_index = find_min_missing(indices)
            # print("min_index",min_index)
            min_indices.append(min_index)
    # print("min_indices", min_indices)

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

import sys
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

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
        "--num-views", "--num_views", type=int, help="Num Views in scenes. Overrides 'num_views' param."
    )
    parser.add_argument(
        "--save-segmentation-data", "--save_segmentation_data", action="store_true", help="Save Segmentation data as PNG, Depth image as .pfm. Overrides 'save_segmentation_data' param."
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
    #print("params:",params)
    Sampler.params = params

    sample = Sampler().sample
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

    # if not overwrite
    if not params["overwrite"]:
        # Check if annotation.json is present, continue from last scene index
        json_files = [pos_json for pos_json in os.listdir(output_dir) if pos_json.endswith('.json')]
        if len(json_files)>0:
            last_scene_index = -1
            last_json_path = ""
            for i in json_files:
                if i != "uoais_train.json":
                    json_index = int(i.split('_')[-1].split('.')[0])
                    if json_index >= last_scene_index:
                        last_scene_index = json_index
                        last_json_path = os.path.join(output_dir,i)

            index = last_scene_index + 1 # get current index
            # read latest json file
            f = open(last_json_path)
            data = json.load(f)

            last_img_index = -1
            for i in data['images']:
                last_img_index = max(last_img_index, int(i['id']))
            last_ann_index = len(data['annotations'])
            f.close()
        
            # remove images more than last scene index, these images do not have annotations
            img_files = [img_path for img_path in os.listdir(output_dir) if img_path.endswith('.png')]
            for path, subdirs, files in os.walk(output_dir):
                for name in files:
                    if name.endswith('.png') or name.endswith('.pfm'):
                        img_scene = int(name.split("_")[0])
                        if img_scene > last_scene_index:
                            img_path = os.path.join(path, name)
                            os.remove(img_path)
            print(f"Removing Images from scene {index} onwards.")
            print(f"Continuing from scene {index}.")

    # Check if dataset is already complete
    assert_dataset_complete(params, index)
    
    # start simulation app
    # config1 = {"headless": False} # to display gui when generating scenes and dataset
    # if params["path_tracing"]:
    #     config1["renderer"] = "PathTracing"
    #     config1["samples_per_pixel_per_frame"] = sample("samples_per_pixel_per_frame")
    # else:
    #     config1["renderer"] = "RayTracedLighting"
    # kit = SimulationApp(config1)


    # Initialize composer
    composer = Composer(params, index, output_dir)
    metrics = Metrics(composer.log_dir, composer.content_log_path)
    if not params["overwrite"] and len(json_files) > 0:
        img_index, ann_index = last_img_index+1, last_ann_index+1
    else:
        img_index, ann_index = 1, 1
    img_list, ann_list = [],[]

    total_st = time.time()
    # Generate dataset
    while composer.index < params["num_scenes"]:
        # get the start time
        st = time.time()
        _, img_index, ann_index, img_list, ann_list = composer.generate_scene(img_index, ann_index,img_list,ann_list)
        # print("\nSize of Variables")
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
        #                  key= lambda x: -x[1])[:]:#10]:
        #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

        # remove all images not are not saved in json/csv

        scene_no = composer.index
        if (scene_no % params["checkpoint_interval"]) == 0 and (scene_no != 0): # save every 2 generated scenes
            gc.collect() # Force the garbage collector for releasing an unreferenced memory
            
            date_created = str(datetime.datetime.now())
            # create annotation file
            coco_json = {
            "info": {
                "description": "Unseen Object Amodal Instance Segmentation NVIDIA Synthetic Dataset ",
                "url": "nil",
                "version": "0.1.0",
                "year": 2022,
                "contributor": "Ng Zhili",
                "date_created": date_created
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "object",
                    "supercategory": "object"
                }
            ],
            "images":img_list,
            "annotations":ann_list}
            
            # if save background segmentation
            if params["save_background"]:
                coco_json["categories"].append({
                    "id": 0,
                    "name": "background",
                    "supercategory": "background"
                })
            
            # remove previous checkpoint json
            # files_in_directory = os.listdir(output_dir)
            # for file in files_in_directory:
            #     if file.endswith(".json"):
            #         path_to_file = os.path.join(output_dir, file)
            #         os.remove(path_to_file)

            # save annotation dict
            with open(f'{output_dir}/uoais_train_{scene_no}.json', 'w') as write_file:
                json.dump(coco_json, write_file, indent=4)
            print(f"\n[Checkpoint] Finished scene {scene_no}, saving annotations to {output_dir}/uoais_train_{scene_no}.json")

            if (scene_no + 1) != params["num_scenes"]:
                # reset lists to prevent memory error
                img_list, ann_list = [],[]
                coco_json = {}

        composer.index += 1
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = time.time() - st
        print(f'\nExecution time for scene {scene_no}:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    date_created = str(datetime.datetime.now())
    # create annotation file
    coco_json = {
    "info": {
        "description": "Unseen Object Amodal Instance Segmentation NVIDIA Synthetic Dataset ",
        "url": "nil",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "Ng Zhili",
        "date_created": date_created
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "object",
            "supercategory": "object"
        }
    ],
    "images":img_list,
    "annotations":ann_list}
    
    # if save background segmentation
    if params["save_background"]:
        coco_json["categories"].append({
            "id": 0,
            "name": "background",
            "supercategory": "background"
        })

    # remove previous checkpoint json
    # files_in_directory = os.listdir(output_dir)
    # for file in files_in_directory:
    #     if file.endswith(".json"):
    #         path_to_file = os.path.join(output_dir, file)
    #         os.remove(path_to_file)

    # save json
    with open(f'{output_dir}/uoais_train_{scene_no}.json', 'w') as write_file:
        json.dump(coco_json, write_file, indent=4)
    print(f"\n[End] Finished last scene {scene_no}, saving annotations to {output_dir}/uoais_train_{scene_no}.json")
    
    # reset lists to prevent memory error
    del img_list
    del ann_list
    del coco_json
    gc.collect()

    elapsed_time = time.time() - total_st
    print(f'\nExecution time for all scenes * {params["num_views"]} views:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # Handle shutdown
    composer.output_manager.data_writer.stop_threads()
    composer.sim_context.clear_instance()
    composer.sim_app.close()

    # Output performance metrics
    metrics.output_performance_metrics()

    # concatenate all coco.json checkpoint files to final coco.json
    final_json_path = f'{output_dir}/uoais_train.json'
    json_files = [os.path.join(output_dir,pos_json) for pos_json in os.listdir(output_dir) if pos_json.endswith('.json')]
    json_files = sorted(json_files)

    coco_json = {"info":{},"licenses":[],"categories":[],"images":[],"annotations":[]}
    for i, file in enumerate(json_files):
        if file != final_json_path:
            f = open(file)
            data = json.load(f)
            if i == 0:
                coco_json["info"] = data["info"]
                coco_json["licenses"] = data["licenses"]
                coco_json["categories"] = data["categories"]
            
            coco_json["images"].extend(data["images"])
            coco_json["annotations"].extend(data["annotations"])
            f.close()

    with open(final_json_path, 'w') as write_file:
        json.dump(coco_json, write_file, indent=4)

    # visualize annotations
    if params["save_segmentation_data"]:
        print("[INFO] Generating occlusion masks...")
        rgb_dir = f"{output_dir}/data/mono/rgb"
        occ_dir = f"{output_dir}/data/mono/occlusion"
        instance_dir = f"{output_dir}/data/mono/instance"
        vis_dir = f"{output_dir}/data/mono/visualize"
        vis_occ_dir = f"{vis_dir}/occlusion"
        vis_instance_dir = f"{vis_dir}/instance"

        # make visualisation output directory
        for dir in [vis_dir,vis_occ_dir, vis_instance_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # iterate through scenes
        rgb_paths = [pos_json for pos_json in os.listdir(rgb_dir) if pos_json.endswith('.png')]
        for scene_index in range(0,params["num_scenes"]):
            # scene_index = str(scene_index_raw) +"_"+str(view_id)
            for view_id in range(0,params["num_views"]):
                rgb_img_list = glob.glob(f"{rgb_dir}/{scene_index}_{view_id}.png")
                rgb_img = cv2.imread(rgb_img_list[0], cv2.IMREAD_UNCHANGED)
                
                occ_img_list = glob.glob(f"{occ_dir}/{scene_index}_{view_id}_*.png")
                #occ_mask_list = []
                if len(occ_img_list) > 0:
                    occ_img = rgb_img.copy()
                    overlay = rgb_img.copy()
                    combined_mask = np.zeros((occ_img.shape[0],occ_img.shape[1]))
                    background = f"{occ_dir}/{scene_index}_background.png"
                    # iterate through all occlusion masks
                    for i in range(len(occ_img_list)):
                        occ_mask_path = occ_img_list[i]
                        if occ_mask_path == background:
                            occ_img_back = rgb_img.copy()
                            overlay_back = rgb_img.copy()
                            occluded_mask = cv2.imread(occ_mask_path, cv2.IMREAD_UNCHANGED)
                            occluded_mask = occluded_mask.astype(bool) # boolean mask
                            overlay_back[occluded_mask] = [0, 0, 255]
                            
                            alpha =0.5                  
                            occ_img_back = cv2.addWeighted(overlay_back, alpha, occ_img_back, 1 - alpha, 0, occ_img_back)      

                            occ_save_path = f"{vis_occ_dir}/{scene_index}_{view_id}_background.png"
                            cv2.imwrite(occ_save_path, occ_img_back)
                        else:
                            occluded_mask = cv2.imread(occ_mask_path, cv2.IMREAD_UNCHANGED)
                            combined_mask += occluded_mask

                    combined_mask = combined_mask.astype(bool) # boolean mask
                    overlay[combined_mask] = [0, 0, 255]
                    alpha =0.5                  
                    occ_img = cv2.addWeighted(overlay, alpha, occ_img, 1 - alpha, 0, occ_img)      
                    occ_save_path = f"{vis_occ_dir}/{scene_index}_{view_id}.png"
                    cv2.imwrite(occ_save_path, occ_img)

                    combined_mask = combined_mask.astype('uint8')
                    occ_save_path = f"{vis_occ_dir}/{scene_index}_{view_id}_mask.png"
                    cv2.imwrite(occ_save_path, combined_mask*255)


                vis_img_list = glob.glob(f"{instance_dir}/{scene_index}_{view_id}_*.png")
                if len(vis_img_list) > 0:
                    vis_img = rgb_img.copy()
                    overlay = rgb_img.copy()
                    background = f"{instance_dir}/{scene_index}_{view_id}_background.png"
                    # iterate through all occlusion masks
                    for i in range(len(vis_img_list)):
                        vis_mask_path = vis_img_list[i]
                        if vis_mask_path == background:
                            vis_img_back = rgb_img.copy()
                            overlay_back = rgb_img.copy()
                            visible_mask = cv2.imread(vis_mask_path, cv2.IMREAD_UNCHANGED)
                            visible_mask = visible_mask.astype(bool) # boolean mask
                            overlay_back[visible_mask] = [0, 0, 255]
                            
                            alpha =0.5                  
                            vis_img_back = cv2.addWeighted(overlay_back, alpha, vis_img_back, 1 - alpha, 0, vis_img_back)      

                            vis_save_path = f"{vis_instance_dir}/{scene_index}_{view_id}_background.png"
                            cv2.imwrite(vis_save_path, vis_img_back)
                        else:
                            visible_mask = cv2.imread(vis_mask_path, cv2.IMREAD_UNCHANGED)
                            vis_combined_mask = visible_mask.astype(bool) # boolean mask                    
                            colour = list(np.random.choice(range(256), size=3))
                            overlay[vis_combined_mask] = colour

                    alpha =0.5   
                    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)        
                    vis_save_path = f"{vis_instance_dir}/{scene_index}_{view_id}.png"
                    cv2.imwrite(vis_save_path,vis_img)

    """
    # generate occlusion masks
    print("[INFO] Generating occlusion masks...")
    rgb_dir = f"{output_dir}/data/mono/rgb"
    rgb_occ_dir = f"{output_dir}/data/mono/rgb_occ_dir"
    semantic_dir = f"{output_dir}/data/mono/semantic"
    occlusion_dir = f"{output_dir}/data/mono/occlusion"
    occlusion_vis_dir = f"{occlusion_dir}/visualize"

    # make directory
    for dir in [rgb_occ_dir,occlusion_dir,occlusion_vis_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    import glob
    import cv2
    import numpy as np

    # iterate through scenes
    for scene_index in range(params["num_scenes"]):
        rgb_img = cv2.imread(f"{rgb_dir}/{scene_index}.png", cv2.IMREAD_UNCHANGED)
        occ_index = 1
        img_list = glob.glob(f"{semantic_dir}/{scene_index}*.png")
        
        # remove completely occluded objects by comparing amodal masks. 
        # if an amodal mask i is completely contained within another amodal mask j then, then the object i is considered completely occluded
        # remove compfinish_scene
        # remove duplicated occluded masks by storing mask index in a set or dictionary

        n = len(img_list) # number of objects

        if n > 0:
            for i in range(n):
                path1 = img_list[i]
                mask1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
                for j in range(i,n):
                    if i == j:
                        pass
                    else:
                        path2 = img_list[j]
                        mask2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
                        iou, intersection_mask = compute_occluded_masks(mask1, mask2)
                        # add occluded masks to image
                        if iou > 0: # occlusion detected
                            save_path = f"{occlusion_dir}/{scene_index}_{occ_index}.png"
                            cv2.imwrite(save_path,intersection_mask)
                            
                            save_path = f"{occlusion_vis_dir}/{scene_index}_{occ_index}.png"
                            cv2.imwrite(save_path,intersection_mask* 255)
                            
                            # visulize occlusion masks on rgb image
                            red = np.ones(intersection_mask.shape)
                            red = red*255
                            rgb_img[:,:,0][intersection_mask>0] = red[intersection_mask>0]
                            occ_index += 1
        
        save_path = f"{rgb_occ_dir}/{scene_index}.png"
        cv2.imwrite(save_path,rgb_img)

        # visulize occlusion masks on rgb image
            #red = np.ones(occ_mask.shape)
            #red = red*255
            #curr_rgb_img[:,:,0][occ_mask>0] = red[occ_mask>0]
            # occ_mask_3c = np.stack((occ_mask,)*3, -1)
            # colour = np.array(list(np.random.choice(range(256), size=3)))
            # rows, cols = np.where(occ_mask_3c[:,:,1]==1)
            # occ_mask_3c[rows,cols,:] = colour
            # occ_mask_list.append(occ_mask_3c)
            
            # combined_mask = np.zeros(occ_mask_3c.shape)
            # for i in occ_mask_list:
            #     combined_mask += i

            # combined_mask = combined_mask.astype('uint8')

            # alpha =0.5
            # vis_img = cv2.addWeighted(curr_rgb_img, alpha, combined_mask, 1 - alpha, 0)
            # occ_save_path = f"{vis_occ_dir}/{scene_index}.png"
            # cv2.imwrite(occ_save_path, vis_img)

                                    # visulize occlusion masks on rgb image
                #red = np.ones(vis_mask.shape)
                #red = red*255
                #curr_rgb_img[:,:,0][vis_mask>0] = red[vis_mask>0]


                # visualize occlusion masks on rgb
                # visible_mask_3c = np.stack((vis_mask,)*3, -1)
                # colour = np.array(list(np.random.choice(range(256), size=3)))
                # rows, cols = np.where(visible_mask_3c[:,:,1]==1)
                # visible_mask_3c[rows,cols,:] = colour
                # vis_mask_list.append(visible_mask_3c)
            # combined_mask = np.zeros(visible_mask_3c.shape)
            # for i in vis_mask_list:
            #     combined_mask += i
            # combined_mask = combined_mask.astype('uint8')
            # alpha =0.5
            # vis_img = cv2.addWeighted(curr_rgb_img, alpha, combined_mask, 1 - alpha, 0)
            # vis_save_path = f"{vis_instance_dir}/{scene_index}.png"
            # cv2.imwrite(vis_save_path,vis_img)

            
    """

    """
    ./python1.sh tools/composer/src/main1.py --input */parameters/uoais.yaml --output */dataset/scenario_room --mount /home/knowledge/zhili --num_scenes 1 --overwrite
    """