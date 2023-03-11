import os
import copy
import numpy as np
import cv2
import carb
import datetime

from output import DisparityConverter, Logger
# from sampling import Sampler
from sampling.sample1 import Sampler
# from omni.isaac.core.utils import prims
from output.writer1 import DataWriter

from helper_functions import compute_occluded_masks, GenericMask, bbox_from_binary_mask # Added
import pycocotools.mask as mask_util

class OutputManager:
    """ For managing Composer outputs, including sending data to the data writer. """

    def __init__(self, sim_app, sim_context, scene_manager, output_data_dir, scene_units_in_meters):
        """ Construct OutputManager. Start data writer threads. """

        from omni.isaac.synthetic_utils.syntheticdata import SyntheticDataHelper

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
        self.label_to_class_id = self.get_label_to_class_id1()
        max_queue_size = 500

        self.save_segmentation_data = self.sample("save_segmentation_data")

        self.write_data = self.sample("write_data")
        if self.write_data:
            self.data_writer = DataWriter(self.output_data_dir, self.sample("num_data_writer_threads"), self.save_segmentation_data, max_queue_size)
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

    def get_label_to_class_id1(self):
        """ Get mapping of object semantic labels to class ids. """

        label_to_class_id = {}
        groups = self.sample("groups")
        for group in groups:
            class_id = self.sample("obj_class_id", group=group)
            label_to_class_id[group] = class_id

        label_to_class_id["[[scenario]]"] = self.sample("scenario_class_id")
        return label_to_class_id

    def capture_amodal_groundtruth(self, index, scene_manager, img_index, ann_index,
                                  view_id, img_list, ann_list,
                                  step_index=0, sequence_length=0):
        """ Capture groundtruth data from Isaac Sim. Send data to data writer. """
        num_objects = len(scene_manager.objs) # get number of objects in scene
        objects = scene_manager.objs # get all objects in scene
        depths = []
        all_viewport_data = []
        
        for i in range(len(self.viewports)):
            viewport_name, viewport_window = self.viewports[i]
            num_digits = len(str(self.sample("num_scenes") - 1))
            img_id = str(index) + "_" + str(view_id)

            groundtruth = {
                    "METADATA": {
                        "image_id": img_id,
                        "viewport_name": viewport_name,
                        "RGB":{},
                        "DEPTH": {},
                        "INSTANCE": {},
                        "SEMANTIC": {},
                        "BBOX2DTIGHT": {},
                        "BBOX2DLOOSE": {},
                        "BBOX3D": {},
                    },
                    "DATA": {},
                }
            
            """ =================================================================
                ===== Collect Viewport's RGB/DEPTH and object visible masks =====
                ================================================================= """
            gt = copy.deepcopy(self.sd_helper.get_groundtruth(self.gt_list, viewport_window, wait_for_sensor_data=0.1))

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
                    semantics = list(self.label_to_class_id.keys())
                    instance_data, instance_mappings = self.sd_helper.sensor_helpers["instanceSegmentation"](
                        viewport_window, parsed=False, return_mapping=True)
                    instances_list = [(im[0], im[4], im["semanticLabel"]) for im in instance_mappings][::-1]
                    max_instance_id_list = max([max(il[1]) for il in instances_list])
                    max_instance_id = instance_data.max()
                    lut = np.zeros(max(max_instance_id, max_instance_id_list) + 1, dtype=np.uint32)

                    for uid, il, sem in instances_list:
                        if sem in semantics and sem != "[[scenario]]":
                            lut[np.array(il)] = uid


                    instance_data = np.take(lut, instance_data)
                    if self.save_segmentation_data:
                        groundtruth["DATA"]["INSTANCE"] = instance_data
                        groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
                        groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
                        groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["INSTANCE"]["NPY"] = True

                    # get visible instance segmentation of all objects in scene
                    instance_map = list(np.unique(instance_data))[1:]
                    org_instance_data_np = np.array(instance_data)
                    org_instance_data = instance_data

                    instance_mappings_dict ={}
                    for obj_prim in instance_mappings:
                        inst_id = obj_prim[0]
                        inst_path = obj_prim[1]
                        instance_mappings_dict[inst_path]= inst_id


            all_viewport_data.append(groundtruth)

            """ ==== define image info dict ==== """
            height, width, _ = gt["rgb"].shape

            date_captured = str(datetime.datetime.now())
            image_info = {
                "id": img_index,
                "file_name": f"data/mono/rgb/{img_id}.png",
                "depth_file_name": f"data/mono/depth/{img_id}.png",
                "occlusion_order_file_name": f"data/mono/occlusion_order/{img_id}.npy",
                "width": width,
                "height": height,
                "date_captured": date_captured,
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }

            """ =====================================
                ===== Collect Background Masks ======
                ===================================== """
            if self.sample("save_background"):
                groundtruth = {
                        "METADATA": {
                            "image_id": str(img_index) + "_background",
                            "viewport_name": viewport_name,
                            "DEPTH": {},
                            "INSTANCE": {},
                            "SEMANTIC": {},
                            "AMODAL": {},
                            "OCCLUSION": {},
                            "BBOX2DTIGHT": {},
                            "BBOX2DLOOSE": {},
                            "BBOX3D": {},
                        },
                        "DATA": {},
                    }

                ann_info = {
                        "id": ann_index,
                        "image_id": img_index,
                        "category_id": 0,
                        "bbox": [],
                        "height": height,
                        "width": width,
                        "object_name":"",
                        "iscrowd": 0,
                        "segmentation": {
                            "size": [
                                height,
                                width
                            ],
                            "counts": "",
                            "area": 0                      
                        },
                        "area": 0,
                        "visible_mask": {
                            "size": [
                                height,
                                width
                            ],
                            "counts": "",
                            "area": 0
                        },
                        "visible_bbox": [],
                        "occluded_mask": {
                            "size": [
                                height,
                                width
                            ],
                            "counts": "",
                            "area": 0
                        },
                        "occluded_rate": 0.0
                    }
                ann_info["object_name"] = "background"

                """ ===== extract visible mask ===== """
                curr_instance_data_np = org_instance_data_np.copy()
                # find pixels that belong to background class
                instance_id = 0 
                curr_instance_data_np[np.where(org_instance_data != instance_id)] = 0
                curr_instance_data_np[np.where(org_instance_data == instance_id)] = 1
                background_visible_mask = curr_instance_data_np.astype(np.uint8)
                """ ===== extract amodal mask ===== """ # background assumed to be binary mask of np.ones
                background_amodal_mask = np.ones(background_visible_mask.shape).astype(np.uint8) # get object amodal mask
                """ ===== calculate occlusion mask ===== """
                background_occ_mask = cv2.absdiff(background_amodal_mask, background_visible_mask)
                """ ===== calculate occlusion rate ===== """ # assumes binary mask (True == 1)
                background_occ_mask_pixel_count = background_occ_mask.sum()
                background_amodal_mask_pixel_count = background_amodal_mask.sum() 
                occlusion_rate = round(background_occ_mask_pixel_count / background_amodal_mask_pixel_count, 2)
                if occlusion_rate < 1: # fully occluded objects are not considered
                    if self.save_segmentation_data:
                        groundtruth["DATA"]["INSTANCE"] = background_visible_mask
                        groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = background_visible_mask.shape[1]
                        groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = background_visible_mask.shape[0]
                        groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["INSTANCE"]["NPY"] = True
                                            
                        groundtruth["DATA"]["AMODAL"] = background_amodal_mask
                        groundtruth["METADATA"]["AMODAL"]["WIDTH"] =  background_amodal_mask.shape[1]
                        groundtruth["METADATA"]["AMODAL"]["HEIGHT"] =  background_amodal_mask.shape[0]
                        groundtruth["METADATA"]["AMODAL"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["AMODAL"]["NPY"] = True

                    #if occlusion_rate > 0: # if object is occluded, save occlusion mask
                    if self.save_segmentation_data:
                        # print(background_occ_mask)
                        # print(background_occ_mask.shape)
                        groundtruth["DATA"]["OCCLUSION"] = background_occ_mask
                        groundtruth["METADATA"]["OCCLUSION"]["WIDTH"] = background_occ_mask.shape[1]
                        groundtruth["METADATA"]["OCCLUSION"]["HEIGHT"] = background_occ_mask.shape[0]
                        groundtruth["METADATA"]["OCCLUSION"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["OCCLUSION"]["NPY"] = True
                    
                    # Assign Mask to Generic Mask Class
                    background_amodal_mask_class = GenericMask(background_amodal_mask.astype("uint8"),height, width)
                    background_visible_mask_class = GenericMask(background_visible_mask.astype("uint8"),height, width)
                    background_occ_mask_class = GenericMask(background_occ_mask.astype("uint8"),height, width)

                    # Encode binary masks to bytes
                    background_amodal_mask= mask_util.encode(np.array(background_amodal_mask[:, :, None], order="F", dtype="uint8"))[0]
                    background_visible_mask= mask_util.encode(np.array(background_visible_mask[:, :, None], order="F", dtype="uint8"))[0]
                    background_occ_mask= mask_util.encode(np.array(background_occ_mask[:, :, None], order="F", dtype="uint8"))[0]

                    # append annotations to dict
                    ann_info["segmentation"]["counts"] = background_amodal_mask['counts'].decode('UTF-8') # amodal mask
                    ann_info["visible_mask"]["counts"] = background_visible_mask['counts'].decode('UTF-8') # obj_visible_mask
                    ann_info["occluded_mask"]["counts"] =background_occ_mask['counts'].decode('UTF-8') # obj_visible_mask
                    
                    ann_info["visible_bbox"] = list(background_visible_mask_class.bbox())
                    ann_info["bbox"] = list(background_visible_mask_class.bbox())

                    ann_info["segmentation"]["area"] = int(background_amodal_mask_class.area())
                    ann_info["visible_mask"]["area"] = int(background_visible_mask_class.area())
                    ann_info["occluded_mask"]["area"] = int(background_occ_mask_class.area())
                    ann_info["occluded_rate"] = occlusion_rate

                ann_index += 1
                all_viewport_data.append(groundtruth)
                ann_list.append(ann_info)
                img_list.append(image_info)

            """ =================================================
                ===== Collect Object Amodal/Occlusion Masks =====
                ================================================= """
            # turn off visibility of all objects
            for obj in objects:
                obj.off_prim()
         
            visible_obj_paths = instance_mappings_dict.keys()

            """ ======= START OBJ LOOP ======= """
            obj_visible_mask_list = []
            obj_occlusion_mask_list = []       
            # loop through objects and capture mask of each object
            for obj in objects:
                # turn on visibility of object
                obj.on_prim() 
                ann_info = {
                    "id": ann_index,
                    "image_id": img_index,
                    "category_id": 1,
                    "bbox": [],
                    "width": width,
                    "height": height,
                    "object_name":"",
                    "iscrowd": 0,
                    "segmentation": {
                        "size": [
                            height,
                            width
                        ],
                        "counts": "",
                        "area": 0                      
                    },
                    "area": 0,
                    "visible_mask": {
                        "size": [
                            height,
                            width
                        ],
                        "counts": "",
                        "area": 0
                    },
                    "visible_bbox": [],
                    "occluded_mask": {
                        "size": [
                            height,
                            width
                        ],
                        "counts": "",
                        "area": 0
                    },
                    "occluded_rate": 0.0
                }
                ann_info["object_name"] = obj.name

                """ ===== get object j index and attributes ===== """
                obj_path = obj.path
                obj_index = int(obj.path.split("/")[-1].split("_")[1])
                id = f"{img_id}_{obj_index}" #image id 
                obj_nested_prim_path = obj_path+"/nested_prim"
                if obj_nested_prim_path in instance_mappings_dict:
                    instance_id = instance_mappings_dict[obj_nested_prim_path]
                else:
                    print(f"{obj_nested_prim_path} does not exist")
                    instance_id = -1
                    print(f"instance_mappings_dict:{instance_mappings_dict}")

                """ ===== Check if Object j is visible from viewport ===== """
                # Remove Fully Occluded Objects from viewport
                if obj_path in visible_obj_paths and instance_id in instance_map: # if object is fully occluded
                    pass

                else: # object is not visible, skipping object
                    obj.off_prim()
                    continue
                
                groundtruth = {
                    "METADATA": {
                        "image_id": id,
                        "viewport_name": viewport_name,
                        "RGB":{},
                        "DEPTH": {},
                        "INSTANCE": {},
                        "SEMANTIC": {},
                        "AMODAL": {},
                        "OCCLUSION": {},
                        "BBOX2DTIGHT": {},
                        "BBOX2DLOOSE": {},
                        "BBOX3D": {},
                    },
                    "DATA": {},
                }

                """ ===== extract visible mask of object j ===== """
                curr_instance_data_np = org_instance_data_np.copy()
                if instance_id != 0: # find object instance segmentation
                    curr_instance_data_np[np.where(org_instance_data_np != instance_id)] = 0
                    curr_instance_data_np[np.where(org_instance_data_np == instance_id)] = 1
                    obj_visible_mask = curr_instance_data_np.astype(np.uint8)



                """ ===== extract amodal mask of object j ===== """
                # Collect Groundtruth
                gt = copy.deepcopy(self.sd_helper.get_groundtruth(self.gt_list, viewport_window, wait_for_sensor_data=0.01))
                
                obj.off_prim() # turn off visibility of object
                
                # RGB
                if self.save_segmentation_data:
                    if "rgb" in gt["state"]:
                        if gt["state"]["rgb"]:
                            groundtruth["DATA"]["RGB"] = gt["rgb"]

                if i == 0 or self.sample("groundtruth_stereo"):
                    # Instance Segmentation
                    if "instanceSegmentation" in gt["state"]:
                        semantics = list(self.label_to_class_id.keys())
                        instance_data, instance_mappings = self.sd_helper.sensor_helpers["instanceSegmentation"](
                            viewport_window, parsed=False, return_mapping=True)

                        instances_list = [(im[0], im[4], im["semanticLabel"]) for im in instance_mappings][::-1]
                        max_instance_id_list = max([max(il[1]) for il in instances_list])
                        max_instance_id = instance_data.max()
                        lut = np.zeros(max(max_instance_id, max_instance_id_list) + 1, dtype=np.uint32)
                        for uid, il, sem in instances_list:
                            if sem in semantics and sem != "[[scenario]]":
                                lut[np.array(il)] = uid

                        instance_data = np.take(lut, instance_data)

                        # get object amodal mask
                        obj_amodal_mask = instance_data.astype(np.uint8)
                        obj_amodal_mask[np.where(instance_data > 0)] = 1

                """ ===== calculate occlusion mask of object j ===== """
                obj_occ_mask = cv2.absdiff(obj_amodal_mask, obj_visible_mask)

                """ ===== calculate occlusion rate of object j ===== """ # assumes binary mask (True == 1)
                obj_occ_mask_pixel_count = obj_occ_mask.sum()
                obj_amodal_mask_pixel_count = obj_amodal_mask.sum() 
                occlusion_rate = round(obj_occ_mask_pixel_count / obj_amodal_mask_pixel_count, 2)
                
                """ ===== Save Segmentation Masks ==== """
                if occlusion_rate < 1: # fully occluded objects are not considered
                    # append visible and occlusion masks for generation of occlusion order matrix
                    obj_visible_mask_list.append(obj_visible_mask)
                    obj_occlusion_mask_list.append(obj_occ_mask)

                    if self.save_segmentation_data:
                        groundtruth["DATA"]["INSTANCE"] =  obj_visible_mask
                        groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = obj_visible_mask.shape[1]
                        groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = obj_visible_mask.shape[0]
                        groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["INSTANCE"]["NPY"] = True
                                            
                        groundtruth["DATA"]["AMODAL"] =  instance_data
                        groundtruth["METADATA"]["AMODAL"]["WIDTH"] = instance_data.shape[1]
                        groundtruth["METADATA"]["AMODAL"]["HEIGHT"] = instance_data.shape[0]
                        groundtruth["METADATA"]["AMODAL"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["AMODAL"]["NPY"] = True

                        # if occlusion_rate > 0: # if object is occluded, save occlusion mask
                        groundtruth["DATA"]["OCCLUSION"] = obj_occ_mask
                        groundtruth["METADATA"]["OCCLUSION"]["WIDTH"] = obj_occ_mask.shape[1]
                        groundtruth["METADATA"]["OCCLUSION"]["HEIGHT"] = obj_occ_mask.shape[0]
                        groundtruth["METADATA"]["OCCLUSION"]["COLORIZE"] = self.groundtruth_visuals
                        groundtruth["METADATA"]["OCCLUSION"]["NPY"] = True
                    
                    ann_info["visible_bbox"] = bbox_from_binary_mask(obj_visible_mask)
                    ann_info["bbox"] = ann_info["visible_bbox"]

                    """ ===== Add Segmentation Mask into COCO.JSON ===== """
                    instance_mask_class = GenericMask(instance_data.astype("uint8"),height, width)
                    obj_visible_mask_class = GenericMask(obj_visible_mask.astype("uint8"),height, width)
                    obj_occ_mask_class = GenericMask(obj_occ_mask.astype("uint8"),height, width)

                    # Encode binary masks to bytes
                    instance_data= mask_util.encode(np.array(instance_data[:, :, None], order="F", dtype="uint8"))[0]
                    obj_visible_mask= mask_util.encode(np.array(obj_visible_mask[:, :, None], order="F", dtype="uint8"))[0]
                    obj_occ_mask= mask_util.encode(np.array(obj_occ_mask[:, :, None], order="F", dtype="uint8"))[0]
                    
                    # append annotations to dict
                    ann_info["segmentation"]["counts"] = instance_data['counts'].decode('UTF-8') # amodal mask
                    ann_info["visible_mask"]["counts"] = obj_visible_mask['counts'].decode('UTF-8') # obj_visible_mask
                    ann_info["occluded_mask"]["counts"] = obj_occ_mask['counts'].decode('UTF-8') # obj_visible_mask
        
                    ann_info["segmentation"]["area"] = int(instance_mask_class.area())
                    ann_info["visible_mask"]["area"] = int(obj_visible_mask_class.area())
                    ann_info["occluded_mask"]["area"] = int(obj_occ_mask_class.area())

                    ann_info["occluded_rate"] = occlusion_rate

                    ann_index += 1
                    all_viewport_data.append(groundtruth)
                    ann_list.append(ann_info)
                    img_list.append(image_info)
                """ ======= END OBJ LOOP ======= """

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
            
            # turn on visibility of all objects (for next camera viewport)
            for obj in objects:
                obj.on_prim()
        
        # generate occlusion ordering for current viewport
        rows = cols = len(obj_visible_mask_list)
        occlusion_adjacency_matrix = np.zeros((rows,cols))
        
        # A(i,j), col j, row i. row i --> col j
        for i in range(0,len(obj_visible_mask_list)):
            visible_mask_i = obj_visible_mask_list[i] # occluder
            for j in range(0,len(obj_visible_mask_list)):
                if j != i:
                    occluded_mask_j = obj_occlusion_mask_list[j] # occludee
                    iou, _ = compute_occluded_masks(visible_mask_i,occluded_mask_j) 
                    if iou > 0: # object i's visible mask is overlapping object j's occluded mask
                        occlusion_adjacency_matrix[i][j] = 1
        
        data_folder = os.path.join(self.output_data_dir, viewport_name, "occlusion_order")
        os.makedirs(data_folder, exist_ok=True)
        filename = os.path.join(data_folder, f"{img_id}.npy")

        # save occlusion adjacency matrix
        np.save(filename, occlusion_adjacency_matrix)
        # increment img index (next viewport)
        img_index += 1
        
        return groundtruth, img_index, ann_index, img_list, ann_list

