"""
SynTable Replicator Composer Main
"""

# import dependencies
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
from distributions import Distribution
from input.parse1 import Parser
from output import Metrics, Logger
from output.output1 import OutputManager
from sampling.sample1 import Sampler
from scene.scene1 import SceneManager
from helper_functions import compute_occluded_masks
from omni.isaac.kit.utils import set_carb_setting
from omni.isaac.core.utils import prims
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

        self.sim_app = SimulationApp(config)

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

    def generate_scene(self, img_index, ann_index, img_list,ann_list,regen_scene):
        """ Generate 1 dataset scene. Returns captured groundtruth data. """
        amodal = True
        self.scene_manager.prepare_scene(self.index)

        # reload table into scene
        self.scene_manager.reload_table()

        kit = self.sim_app
        # if generate amodal annotations
        if amodal:
            roomTableSize = self.scene_manager.roomTableSize
            roomTableHeight = roomTableSize[-1]
            spawnLowerBoundOffset = 0.2
            spawnUpperBoundOffset = 1

            # calculate tableBounds to constraint objects' spawn locations to be within tableBounds
            x_width = roomTableSize[0] /2 
            y_length = roomTableSize[1] /2
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
        
        # if generate amodal annotations
        elif amodal:
            # simulate physical dropping of objects
            self.scene_manager.update_scene() 
            # refresh UI rendering
            self.sim_context.render()
            # pause simulation
            self.sim_context.pause()
            
            # stop all object motion and remove objects not on tabletop
            objects = self.scene_manager.objs.copy() 
            objects_filtered = []

            # remove objects outside tabletop regions after simulation          
            for obj in objects:
                obj.coord, quaternion = obj.xform_prim.get_world_pose()
                obj.coord = np.array(obj.coord, dtype=np.float32)

                # if object is not on tabletop after simulation, remove object
                if (abs(obj.coord[0]) > (roomTableSize[0]/2)) \
                    or (abs(obj.coord[1]) > (roomTableSize[1]/2)) \
                    or (abs(obj.coord[2]) < roomTableSize[2]):
                    # remove object by turning off visibility of object
                    obj.off_prim()
                # else object on tabletop, add obj to filtered list
                else:
                    objects_filtered.append(obj)
            
            
            self.scene_manager.objs = objects_filtered    
            # if no objects left on tabletop, regenerate scene
            if len(self.scene_manager.objs) == 0:
                print("No objects found on tabletop, regenerating scene.")
                self.scene_manager.finish_scene()
                return None, img_index, ann_index, img_list, ann_list, regen_scene 
            else:
                regen_scene = False
            print("\nNumber of Objects on tabletop:", len(self.scene_manager.objs))
            
            # get camera coordinates based on hemisphere of radus r and tabletop height
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
                return np.array([x,y,z])

            # Randomly move camera and light coordinates to be constrainted between 2 concentric hemispheres above tabletop
            numViews = self.params["num_views"]

            # get hemisphere radius bounds
            autoHemisphereRadius = self.sample("auto_hemisphere_radius")
            if not autoHemisphereRadius:
                camHemisphereRadiusMin = self.sample("cam_hemisphere_radius_min")
                camHemisphereRadiusMax = self.sample("cam_hemisphere_radius_max")
                lightHemisphereRadiusMin = self.sample("spherelight_hemisphere_radius_min")
                lightHemisphereRadiusMax = self.sample("spherelight_hemisphere_radius_max")
            else:
                camHemisphereRadiusMin = max(x_width,y_length) * 0.8
                camHemisphereRadiusMax = camHemisphereRadiusMin + 0.7*camHemisphereRadiusMin
                lightHemisphereRadiusMin = camHemisphereRadiusMax + 0.1
                lightHemisphereRadiusMax = lightHemisphereRadiusMin + 1
            print(x_width,y_length)
            print("\n===Camera & Light Hemisphere Parameters===")
            print(f"autoHemisphereRadius:{autoHemisphereRadius}")
            print(f"camHemisphereRadiusMin = {camHemisphereRadiusMin}")
            print(f"camHemisphereRadiusMax = {camHemisphereRadiusMax}")
            print(f"lightHemisphereRadiusMin = {lightHemisphereRadiusMin}")
            print(f"lightHemisphereRadiusMax = {lightHemisphereRadiusMax}")

            Logger.print(f"\n=== Capturing Groundtruth for each viewport in scene ===\n")
            for view_id in range(numViews):
                random.seed(None)
                Logger.print(f"\n==> Scene: {self.index}, View: {view_id} <==\n")
                # resample radius of camera hemisphere between min and max radii bounds
                r = random.uniform(camHemisphereRadiusMin,camHemisphereRadiusMax)
                print('sampled radius r of camera hemisphere:',r)

                # resample camera coordinates and rotate camera to look at tabletop surface center
                cam_coord_w = camera_orbit_coord(r=r,tableTopHeight=roomTableHeight+0.2)
                print("sampled camera coordinate:",cam_coord_w)
                self.scene_manager.camera.translate(cam_coord_w)
                self.scene_manager.camera.translate_rotate(target=(0,0,roomTableHeight)) #target coordinates

                # initialise ambient lighting as 0 (for ray tracing), path tracing not affected
                rtx_mode = "/rtx"
                ambient_light_intensity = 0 #random.uniform(0.2,3.5)
                set_carb_setting(kit._carb_settings, rtx_mode + "/sceneDb/ambientLightIntensity", ambient_light_intensity)

                # Enable indirect diffuse GI (for ray tracing)
                set_carb_setting(kit._carb_settings, rtx_mode + "/indirectDiffuse/enabled", True)
                
                # Reset and delete all lights
                for light in self.scene_manager.lights:
                    prims.delete_prim(light.path)
                

                # Resample number of lights in viewport
                self.scene_manager.lights = []
                for grp_index, group in enumerate(self.scene_manager.sample("groups")):
                    # adjust ceiling light parameters
                    if group == "ceilinglights":
                      
                        for lightIndex, light in enumerate(self.scene_manager.ceilinglights):
                            
                            if lightIndex == 0:
                                new_intensity = light.sample("light_intensity")
                                if light.sample("light_temp_enabled"):
                                    new_temp = light.sample("light_temp")
                            
                            # change light intensity
                            light.attributes["intensity"] = new_intensity
                            light.prim.GetAttribute("intensity").Set(light.attributes["intensity"])

                            # change light temperature
                            if light.sample("light_temp_enabled"):
                                light.attributes["colorTemperature"] = new_temp
                                light.prim.GetAttribute("colorTemperature").Set(light.attributes["colorTemperature"])
                    
                    # adjust spherical light parameters
                    if group == "lights":
                        num_lights = self.scene_manager.sample("light_count", group=group)
                        for i in range(num_lights):
                            path = "{}/Lights/lights_{}".format( self.scene_manager.scene_path, len(self.scene_manager.lights))
                            light = Light(self.scene_manager.sim_app, self.scene_manager.sim_context, path, self.scene_manager.camera, group)
                            
                            # change light intensity
                            light.attributes["intensity"] = light.sample("light_intensity")
                            light.prim.GetAttribute("intensity").Set(light.attributes["intensity"])
                            
                            # change light temperature
                            if light.sample("light_temp_enabled"):
                                light.attributes["colorTemperature"] =light.sample("light_temp")
                                light.prim.GetAttribute("colorTemperature").Set(light.attributes["colorTemperature"])
                            
                            # change light coordinates
                            light_coord_w = camera_orbit_coord(r=random.uniform(lightHemisphereRadiusMin,lightHemisphereRadiusMax),tableTopHeight=roomTableHeight+0.2)
                            light.translate(light_coord_w)
                            light.coord, quaternion = light.xform_prim.get_world_pose()
                            light.coord = np.array(light.coord, dtype=np.float32)
                            self.scene_manager.lights.append(light)

                        print(f"Number of sphere lights in scene: {len(self.scene_manager.lights)}")

                # capture groundtruth of entire viewpoint
                groundtruth, img_index, ann_index, img_list, ann_list = \
                    self.output_manager.capture_amodal_groundtruth(self.index, 
                                                               self.scene_manager,
                                                               img_index, ann_index, view_id,
                                                               img_list, ann_list
                                                               )             

        else:
            self.scene_manager.update_scene()
            groundtruth = self.output_manager.capture_groundtruth(self.index)
        
        # finish the scene and reset prims in scene
        self.scene_manager.finish_scene()

        return groundtruth, img_index, ann_index, img_list, ann_list, regen_scene

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
    """ Determine output directory to store datasets. 
    """
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
    json_files = []
    if not params["overwrite"] and os.path.isdir(output_dir):
        # Check if annotation_final.json is present, continue from last scene index
        json_files = [pos_json for pos_json in os.listdir(output_dir) if pos_json.endswith('.json')]
        if len(json_files)>0:
            last_scene_index = -1
            last_json_path = ""
            for i in json_files:
                if i != "annotation_final.json":
                    json_index = int(i.split('_')[-1].split('.')[0])
                    if json_index >= last_scene_index:
                        last_scene_index = json_index
                        last_json_path = os.path.join(output_dir,i)

            # get current index
            index = last_scene_index + 1 
            # read latest json file
            f = open(last_json_path)
            data = json.load(f)

            last_img_index = max(data['images'][-1]['id'],-1)
            last_ann_index = max(data['annotations'][-1]['id'],-1)
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

    # Initialize composer
    composer = Composer(params, index, output_dir)
    metrics = Metrics(composer.log_dir, composer.content_log_path)
    if not params["overwrite"] and os.path.isdir(output_dir) and len(json_files) > 0:
        img_index, ann_index = last_img_index+1, last_ann_index+1
    else:
        img_index, ann_index = 1, 1
    img_list, ann_list = [],[]

    total_st = time.time()
    # Generate dataset
    while composer.index < params["num_scenes"]:
        # get the start time
        st = time.time()
        regen_scene = True
        while regen_scene:
            _, img_index, ann_index, img_list, ann_list, regen_scene = composer.generate_scene(img_index, ann_index,img_list,ann_list,regen_scene)

        # remove all images not are not saved in json/csv
        scene_no = composer.index
        if (scene_no % params["checkpoint_interval"]) == 0 and (scene_no != 0): # save every 2 generated scenes
            gc.collect() # Force the garbage collector for releasing an unreferenced memory
            
            date_created = str(datetime.datetime.now())
            # create annotation file
            coco_json = {
            "info": {
                "description": "SynTable",
                "url": "nil",
                "version": "0.1.0",
                "year": 2022,
                "contributor": "SynTable",
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
                    "supercategory": "shape"
                }
            ],
            "images":img_list,
            "annotations":ann_list}
            
            # if save background segmentation
            if params["save_background"]:
                coco_json["categories"].append({
                    "id": 0,
                    "name": "background",
                    "supercategory": "shape"
                })
            
            # save annotation dict
            with open(f'{output_dir}/annotation_{scene_no}.json', 'w') as write_file:
                json.dump(coco_json, write_file, indent=4)
            print(f"\n[Checkpoint] Finished scene {scene_no}, saving annotations to {output_dir}/annotation_{scene_no}.json")

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
        "description": "SynTable",
        "url": "nil",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "SynTable",
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
            "supercategory": "shape"
        }
    ],
    "images":img_list,
    "annotations":ann_list}
    
    # if save background segmentation
    if params["save_background"]:
        coco_json["categories"].append({
            "id": 0,
            "name": "background",
            "supercategory": "shape"
        })

    # save json
    with open(f'{output_dir}/annotation_{scene_no}.json', 'w') as write_file:
        json.dump(coco_json, write_file, indent=4)
    print(f"\n[End] Finished last scene {scene_no}, saving annotations to {output_dir}/annotation_{scene_no}.json")
    
    # reset lists to prevent out of memory (oom) error 
    del img_list
    del ann_list
    del coco_json
    gc.collect() # Force the garbage collector for releasing an unreferenced memory

    elapsed_time = time.time() - total_st
    print(f'\nExecution time for all {params["num_scenes"]} scenes * {params["num_views"]} views:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # Handle shutdown
    composer.output_manager.data_writer.stop_threads()
    composer.sim_context.clear_instance()
    composer.sim_app.close()

    # Output performance metrics
    metrics.output_performance_metrics()

    # concatenate all coco.json checkpoint files to final coco.json
    final_json_path = f'{output_dir}/annotation_final.json'
    json_files = [os.path.join(output_dir,pos_json) for pos_json in os.listdir(output_dir) if (pos_json.endswith('.json') and os.path.join(output_dir,pos_json) != final_json_path)]
    json_files = sorted(json_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

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
