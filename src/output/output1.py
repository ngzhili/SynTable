# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import numpy as np

import carb

from output import DisparityConverter, Logger
from sampling import Sampler
from omni.isaac.core.utils import prims
from output.writer1 import DataWriter

class OutputManager:
    """ For managing Composer outputs, including sending data to the data writer. """

    def __init__(self, sim_app, sim_context, scene_manager, output_data_dir, scene_units_in_meters):
        """ Construct OutputManager. Start data writer threads. """

        from omni.isaac.synthetic_utils import SyntheticDataHelper

        self.sim_app = sim_app
        self.sim_context = sim_context
        self.scene_manager = scene_manager
        self.output_data_dir = output_data_dir
        self.scene_units_in_meters = scene_units_in_meters

        self.camera = self.scene_manager.camera
        self.viewports = self.camera.viewports
        self.stage = self.sim_context.stage
        self.sample = Sampler().sample

        self.groundtruth_visuals = self.sample("groundtruth_visuals")
        self.label_to_class_id = self.get_label_to_class_id()
        max_queue_size = 500

        self.write_data = self.sample("write_data")
        if self.write_data:
            self.data_writer = DataWriter(self.output_data_dir, self.sample("num_data_writer_threads"), max_queue_size)
            self.data_writer.start_threads()

        self.sd_helper = SyntheticDataHelper()

        self.gt_list = []
        if self.sample("rgb") or (
            self.sample("bbox_2d_tight")
            or self.sample("bbox_2d_loose")
            or self.sample("bbox_3d")
            and self.groundtruth_visuals
        ):
            self.gt_list.append("rgb")
        if (self.sample("depth")) or (self.sample("disparity") and self.sample("stereo")):
            self.gt_list.append("depthLinear")
        if self.sample("instance_seg"):
            self.gt_list.append("instanceSegmentation")
        if self.sample("semantic_seg"):
            self.gt_list.append("semanticSegmentation")
        if self.sample("bbox_2d_tight"):
            self.gt_list.append("boundingBox2DTight")
        if self.sample("bbox_2d_loose"):
            self.gt_list.append("boundingBox2DLoose")
        if self.sample("bbox_3d"):
            self.gt_list.append("boundingBox3D")

        for viewport_name, viewport_window in self.viewports:
            self.sd_helper.initialize(sensor_names=self.gt_list, viewport=viewport_window)
            self.sim_app.update()

        self.carb_settings = carb.settings.acquire_settings_interface()

    def get_label_to_class_id(self):
        """ Get mapping of object semantic labels to class ids. """

        label_to_class_id = {}
        groups = self.sample("groups")
        for group in groups:
            class_id = self.sample("obj_class_id", group=group)
            label_to_class_id[group] = class_id

        label_to_class_id["[[scenario]]"] = self.sample("scenario_class_id")

        return label_to_class_id

    def capture_groundtruth(self, index, step_index=0, sequence_length=0):
        """ Capture groundtruth data from Isaac Sim. Send data to data writer. """

        depths = []
        all_viewport_data = []
        for i in range(len(self.viewports)):
            self.sim_context.render()
            self.sim_context.render()

            viewport_name, viewport_window = self.viewports[i]

            num_digits = len(str(self.sample("num_scenes") - 1))
            id = str(index)
            id = id.zfill(num_digits)

            if self.sample("sequential"):
                num_digits = len(str(sequence_length - 1))
                suffix_id = str(step_index)
                suffix_id = suffix_id.zfill(num_digits)
                id = id + "_" + suffix_id

            groundtruth = {
                "METADATA": {
                    "image_id": id,
                    "viewport_name": viewport_name,
                    "DEPTH": {},
                    "INSTANCE": {},
                    "SEMANTIC": {},
                    "BBOX2DTIGHT": {},
                    "BBOX2DLOOSE": {},
                    "BBOX3D": {},
                },
                "DATA": {},
            }

            # Collect Groundtruth
            self.sim_context.render()
            self.sim_context.render()
            gt = copy.deepcopy(self.sd_helper.get_groundtruth(self.gt_list, viewport_window, wait_for_sensor_data=0.2))

            # zhili added
            #print("groundtruth:\n")
            #print(gt)
            #print("\n")
            #for key,value in gt.items():
                #print(key, value)
                #print("\n\n")

            #for i in gt["instanceSegmentation"]:
                #print(i)
                #print("\n")
            
            #print( gt["instanceSegmentation"][1][7])
            #print( gt["instanceSegmentation"][1][7][1])
            #cube2_ref = prims.get_prim_at_path(gt["instanceSegmentation"][1][7][1])
            #prims.set_prim_visibility(cube2_ref, False)


            # RGB
            if "rgb" in gt["state"]:
                if gt["state"]["rgb"]:
                    groundtruth["DATA"]["RGB"] = gt["rgb"]

            # Depth (for Disparity)
            if "depthLinear" in gt["state"]:
                depth_data = copy.deepcopy(gt["depthLinear"]).squeeze()
                # Convert to scene units
                depth_data /= self.scene_units_in_meters
                depths.append(depth_data)

            if i == 0 or self.sample("groundtruth_stereo"):
                # Depth
                if "depthLinear" in gt["state"]:
                    if self.sample("depth"):
                        depth_data = gt["depthLinear"].squeeze()
                        # Convert to scene units
                        depth_data /= self.scene_units_in_meters
                        groundtruth["DATA"]["DEPTH"] = depth_data
                        groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["DEPTH"]["NPY"] = True

                # Instance Segmentation
                if "instanceSegmentation" in gt["state"]:
                    instance_data = gt["instanceSegmentation"][0]
                    groundtruth["DATA"]["INSTANCE"] = instance_data
                    groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
                    groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
                    groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["INSTANCE"]["NPY"] = True
                    
                    # zhili added
                    #print("gt", gt)
                    #print("instance_data:", instance_data)
                    #print("groundtruth:", groundtruth)

                # Semantic Segmentation
                if "semanticSegmentation" in gt["state"]:
                    semantic_data = gt["semanticSegmentation"]
                    semantic_data = self.sd_helper.get_mapped_semantic_data(
                        semantic_data, self.label_to_class_id, remap_using_base_class=True
                    )
                    semantic_data = np.array(semantic_data)
                    semantic_data[semantic_data == 65535] = 0  # deals with invalid semantic id
                    groundtruth["DATA"]["SEMANTIC"] = semantic_data
                    groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data.shape[1]
                    groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data.shape[0]
                    groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["SEMANTIC"]["NPY"] = True

                # 2D Tight BBox
                if "boundingBox2DTight" in gt["state"]:
                    groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
                    groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = True

                # 2D Loose BBox
                if "boundingBox2DLoose" in gt["state"]:
                    groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
                    groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = True

                # 3D BBox
                if "boundingBox3D" in gt["state"]:
                    groundtruth["DATA"]["BBOX3D"] = gt["boundingBox3D"]
                    groundtruth["METADATA"]["BBOX3D"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["BBOX3D"]["NPY"] = True

            all_viewport_data.append(groundtruth)

        # Wireframe
        if self.sample("wireframe"):
            self.carb_settings.set("/rtx/wireframe/mode", 2.0)
            # Need two updates for all viewports to have wireframe properly
            self.sim_context.render()
            self.sim_context.render()
            for i in range(len(self.viewports)):
                viewport_name, viewport_window = self.viewports[i]
                gt = copy.deepcopy(self.sd_helper.get_groundtruth(["rgb"], viewport_window))
                all_viewport_data[i]["DATA"]["WIREFRAME"] = gt["rgb"]
            self.carb_settings.set("/rtx/wireframe/mode", 0)
            self.sim_context.render()

        for i in range(len(self.viewports)):
            if self.write_data:
                self.data_writer.q.put(copy.deepcopy(all_viewport_data[i]))

        # Disparity
        if self.sample("disparity") and self.sample("stereo"):
            depth_l, depth_r = depths

            cam_intrinsics = self.camera.intrinsics[0]
            disp_convert = DisparityConverter(
                depth_l,
                depth_r,
                cam_intrinsics["fx"],
                cam_intrinsics["fy"],
                cam_intrinsics["cx"],
                cam_intrinsics["cy"],
                self.sample("stereo_baseline"),
            )
            disp_l, disp_r = disp_convert.compute_disparity()
            disparities = [disp_l, disp_r]

            for i in range(len(self.viewports)):
                if i == 0 or self.sample("groundtruth_stereo"):
                    viewport_name, viewport_window = self.viewports[i]
                    groundtruth = {
                        "METADATA": {"image_id": id, "viewport_name": viewport_name, "DISPARITY": {}},
                        "DATA": {},
                    }
                    disparity_data = disparities[i]
                    groundtruth["DATA"]["DISPARITY"] = disparity_data
                    groundtruth["METADATA"]["DISPARITY"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["DISPARITY"]["NPY"] = True

                    if self.write_data:
                        self.data_writer.q.put(copy.deepcopy(groundtruth))

        return groundtruth


    def capture_amodal_groundtruth(self, index, scene_manager, step_index=0, sequence_length=0):
        """ Capture groundtruth data from Isaac Sim. Send data to data writer. """
        num_objects = len(scene_manager.objs)
        objects = scene_manager.objs

        depths = []
        all_viewport_data = []
        for i in range(len(self.viewports)):
            # turn off visibility and physics of all objects
            for obj in objects:
                obj.off_physics_prim()
                #obj.print_instance_attributes()

            self.sim_context.render()
            self.sim_context.render()
            for obj in objects:
                obj.off_physics_prim()

            viewport_name, viewport_window = self.viewports[i]

            num_digits = len(str(self.sample("num_scenes") - 1))
            id = str(index)
            img_id = id.zfill(num_digits)

            if self.sample("sequential"):
                num_digits = len(str(sequence_length - 1))
                suffix_id = str(step_index)
                suffix_id = suffix_id.zfill(num_digits)
                img_id = id + "_" + suffix_id

            # get original RBG
            groundtruth = {
                    "METADATA": {
                        "image_id": img_id,
                        "viewport_name": viewport_name,
                        "DEPTH": {},
                        "INSTANCE": {},
                        "SEMANTIC": {},
                        "BBOX2DTIGHT": {},
                        "BBOX2DLOOSE": {},
                        "BBOX3D": {},
                    },
                    "DATA": {},
                }

            # Collect Groundtruth for original scene
            self.sim_context.render()
            self.sim_context.render()
            for obj in objects:
                obj.off_physics_prim()
            gt = copy.deepcopy(self.sd_helper.get_groundtruth(self.gt_list, viewport_window, wait_for_sensor_data=0.2))

            # RGB
            if "rgb" in gt["state"]:
                if gt["state"]["rgb"]:
                    groundtruth["DATA"]["RGB"] = gt["rgb"]
            # Depth (for Disparity)
            if "depthLinear" in gt["state"]:
                depth_data = copy.deepcopy(gt["depthLinear"]).squeeze()
                # Convert to scene units
                depth_data /= self.scene_units_in_meters
                depths.append(depth_data)

            if i == 0 or self.sample("groundtruth_stereo"):
                # Depth
                if "depthLinear" in gt["state"]:
                    if self.sample("depth"):
                        depth_data = gt["depthLinear"].squeeze()
                        # Convert to scene units
                        depth_data /= self.scene_units_in_meters
                        groundtruth["DATA"]["DEPTH"] = depth_data
                        groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["DEPTH"]["NPY"] = True
            
            all_viewport_data.append(groundtruth)
            """
            for i in range(len(self.viewports)):
                if self.write_data:
                    self.data_writer.q.put(copy.deepcopy(all_viewport_data[i]))
            """
            print("all_viewport_data = ",len(all_viewport_data))

            # Collect Amodal Groundtruth 

            # turn off visibility of all objects
            for obj in objects:
                obj.off_prim()
                obj.off_physics_prim()
                #print(obj.print_instance_attributes())
                
            obj.print_instance_attributes()
            # loop through objects and capture mask of each object
            for obj_index in range(num_objects):
                id = f"{img_id}_{obj_index}"

                obj = objects[obj_index]
                obj.on_prim() # turn on object
                obj.off_physics_prim()
                # capture instance segmentation of object j
                #groundtruth = self.output_manager.capture_amodal_groundtruth(self.index, obj_index=i)    
            
                groundtruth = {
                    "METADATA": {
                        "image_id": id,
                        "viewport_name": viewport_name,
                        "DEPTH": {},
                        "INSTANCE": {},
                        "SEMANTIC": {},
                        "BBOX2DTIGHT": {},
                        "BBOX2DLOOSE": {},
                        "BBOX3D": {},
                    },
                    "DATA": {},
                }

                # Collect Groundtruth
                #self.sim_context.render()
                #self.sim_context.render()
                gt = copy.deepcopy(self.sd_helper.get_groundtruth(self.gt_list, viewport_window, wait_for_sensor_data=0.2))

                # RGB
                if "rgb" in gt["state"]:
                    if gt["state"]["rgb"]:
                        groundtruth["DATA"]["RGB"] = gt["rgb"]

                # Depth (for Disparity)
                if "depthLinear" in gt["state"]:
                    depth_data = copy.deepcopy(gt["depthLinear"]).squeeze()
                    # Convert to scene units
                    depth_data /= self.scene_units_in_meters
                    depths.append(depth_data)

                if i == 0 or self.sample("groundtruth_stereo"):
                    # Depth
                    if "depthLinear" in gt["state"]:
                        if self.sample("depth"):
                            depth_data = gt["depthLinear"].squeeze()
                            # Convert to scene units
                            depth_data /= self.scene_units_in_meters
                            groundtruth["DATA"]["DEPTH"] = depth_data
                            groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = self.groundtruth_visuals
                            groundtruth["METADATA"]["DEPTH"]["NPY"] = True

                    # Instance Segmentation
                    if "instanceSegmentation" in gt["state"]:
                        instance_data = gt["instanceSegmentation"][0]
                        groundtruth["DATA"]["INSTANCE"] = instance_data
                        groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
                        groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
                        groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["INSTANCE"]["NPY"] = True
                        
                        # zhili added
                        #print("gt", gt)
                        #print("instance_data:", instance_data)
                        #print("groundtruth:", groundtruth)

                    # Semantic Segmentation
                    if "semanticSegmentation" in gt["state"]:
                        semantic_data = gt["semanticSegmentation"]
                        semantic_data = self.sd_helper.get_mapped_semantic_data(
                            semantic_data, self.label_to_class_id, remap_using_base_class=True
                        )
                        semantic_data = np.array(semantic_data)
                        semantic_data[semantic_data == 65535] = 0  # deals with invalid semantic id
                        groundtruth["DATA"]["SEMANTIC"] = semantic_data
                        groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data.shape[1]
                        groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data.shape[0]
                        groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["SEMANTIC"]["NPY"] = True

                    # 2D Tight BBox
                    if "boundingBox2DTight" in gt["state"]:
                        groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
                        groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = True

                    # 2D Loose BBox
                    if "boundingBox2DLoose" in gt["state"]:
                        groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
                        groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = True

                    # 3D BBox
                    if "boundingBox3D" in gt["state"]:
                        groundtruth["DATA"]["BBOX3D"] = gt["boundingBox3D"]
                        groundtruth["METADATA"]["BBOX3D"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["BBOX3D"]["NPY"] = True
                
                obj.off_prim() # turn off object
                all_viewport_data.append(groundtruth)
            
            obj.print_instance_attributes()
            print("all_viewport_data = ",len(all_viewport_data))



            # Wireframe
            if self.sample("wireframe"):
                self.carb_settings.set("/rtx/wireframe/mode", 2.0)
                # Need two updates for all viewports to have wireframe properly
                self.sim_context.render()
                self.sim_context.render()
                for i in range(len(self.viewports)):
                    viewport_name, viewport_window = self.viewports[i]
                    gt = copy.deepcopy(self.sd_helper.get_groundtruth(["rgb"], viewport_window))
                    all_viewport_data[i]["DATA"]["WIREFRAME"] = gt["rgb"]
                self.carb_settings.set("/rtx/wireframe/mode", 0)
                self.sim_context.render()

            #for i in range(len(self.viewports)):
            for j in range(len(all_viewport_data)):
                if self.write_data:
                    self.data_writer.q.put(copy.deepcopy(all_viewport_data[j]))




            # Disparity
            if self.sample("disparity") and self.sample("stereo"):
                depth_l, depth_r = depths

                cam_intrinsics = self.camera.intrinsics[0]
                disp_convert = DisparityConverter(
                    depth_l,
                    depth_r,
                    cam_intrinsics["fx"],
                    cam_intrinsics["fy"],
                    cam_intrinsics["cx"],
                    cam_intrinsics["cy"],
                    self.sample("stereo_baseline"),
                )
                disp_l, disp_r = disp_convert.compute_disparity()
                disparities = [disp_l, disp_r]

                for i in range(len(self.viewports)):
                    if i == 0 or self.sample("groundtruth_stereo"):
                        viewport_name, viewport_window = self.viewports[i]
                        groundtruth = {
                            "METADATA": {"image_id": id, "viewport_name": viewport_name, "DISPARITY": {}},
                            "DATA": {},
                        }
                        disparity_data = disparities[i]
                        groundtruth["DATA"]["DISPARITY"] = disparity_data
                        groundtruth["METADATA"]["DISPARITY"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["DISPARITY"]["NPY"] = True

                        if self.write_data:
                            self.data_writer.q.put(copy.deepcopy(groundtruth))
                


        return groundtruth



    # zhili added
    def capture_amodal_groundtruth1(self, index, obj_index=0):
        """ Capture groundtruth data from Isaac Sim. Send data to data writer. """

        depths = []
        all_viewport_data = []
        for i in range(len(self.viewports)):
            self.sim_context.render()
            self.sim_context.render()

            viewport_name, viewport_window = self.viewports[i]

            num_digits = len(str(self.sample("num_scenes") - 1))
            id = str(index)
            id = id.zfill(num_digits)

            id = f"{id}_{obj_index}"

            groundtruth = {
                "METADATA": {
                    "image_id": id,
                    "viewport_name": viewport_name,
                    "DEPTH": {},
                    "INSTANCE": {},
                    "SEMANTIC": {},
                    "BBOX2DTIGHT": {},
                    "BBOX2DLOOSE": {},
                    "BBOX3D": {},
                },
                "DATA": {},
            }

            # Collect Groundtruth
            self.sim_context.render()
            self.sim_context.render()
            gt = copy.deepcopy(self.sd_helper.get_groundtruth(self.gt_list, viewport_window, wait_for_sensor_data=0.2))

            # RGB
            #"""
            if "rgb" in gt["state"]:
                if gt["state"]["rgb"]:
                    groundtruth["DATA"]["RGB"] = gt["rgb"]
            #"""

            if i == 0 or self.sample("groundtruth_stereo"):

                # Instance Segmentation
                if "instanceSegmentation" in gt["state"]:
                    instance_data = gt["instanceSegmentation"][0]
                    groundtruth["DATA"]["INSTANCE"] = instance_data
                    groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
                    groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
                    groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["INSTANCE"]["NPY"] = True
                    
                    # zhili added
                    #print("gt", gt)
                    #print("instance_data:", instance_data)
                    #print("groundtruth:", groundtruth)

                # Semantic Segmentation
                if "semanticSegmentation" in gt["state"]:
                    semantic_data = gt["semanticSegmentation"]
                    semantic_data = self.sd_helper.get_mapped_semantic_data(
                        semantic_data, self.label_to_class_id, remap_using_base_class=True
                    )
                    semantic_data = np.array(semantic_data)
                    semantic_data[semantic_data == 65535] = 0  # deals with invalid semantic id
                    groundtruth["DATA"]["SEMANTIC"] = semantic_data
                    groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data.shape[1]
                    groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data.shape[0]
                    groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["SEMANTIC"]["NPY"] = True

                # 2D Tight BBox
                if "boundingBox2DTight" in gt["state"]:
                    groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
                    groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = True

                # 2D Loose BBox
                if "boundingBox2DLoose" in gt["state"]:
                    groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
                    groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = True

                # 3D BBox
                if "boundingBox3D" in gt["state"]:
                    groundtruth["DATA"]["BBOX3D"] = gt["boundingBox3D"]
                    groundtruth["METADATA"]["BBOX3D"]["COLORIZE"] = self.groundtruth_visuals
                    groundtruth["METADATA"]["BBOX3D"]["NPY"] = True

            all_viewport_data.append(groundtruth)

        for i in range(len(self.viewports)):
            if self.write_data:
                self.data_writer.q.put(copy.deepcopy(all_viewport_data[i]))

        return groundtruth