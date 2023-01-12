# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import numpy as np

import carb

from scene.asset1 import Asset
from output import Logger
# from sampling import Sampler
from sampling.sample1 import Sampler

class Camera(Asset):
    """ For managing a camera in Isaac Sim. """

    def __init__(self, sim_app, sim_context, path, camera, group):
        """ Construct Camera. """

        self.sample = Sampler(group=group).sample
        self.stereo = self.sample("stereo")

        if self.stereo:
            name = "stereo_cams"
        else:
            name = "mono_cam"

        super().__init__(sim_app, sim_context, path, "camera", name, camera=camera, group=group)

        self.load_camera()

    def is_coord_camera_relative(self):
        return False

    def is_rot_camera_relative(self):
        return False

    def load_camera(self):
        """ Create a camera in Isaac Sim. """

        import omni
        from pxr import Sdf, UsdGeom
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils import prims

        self.prim = prims.create_prim(self.path, "Xform")
        self.xform_prim = XFormPrim(self.path)
        self.camera_rig = UsdGeom.Xformable(self.prim)

        camera_prim_paths = []
        if self.stereo:
            camera_prim_paths.append(self.path + "/LeftCamera")
            camera_prim_paths.append(self.path + "/RightCamera")
        else:
            camera_prim_paths.append(self.path + "/MonoCamera")

        self.cameras = [
            self.stage.DefinePrim(Sdf.Path(camera_prim_path), "Camera") for camera_prim_path in camera_prim_paths
        ]

        focal_length = self.sample("focal_length")
        focus_distance = self.sample("focus_distance")
        horiz_aperture = self.sample("horiz_aperture")
        vert_aperture = self.sample("vert_aperture")
        f_stop = self.sample("f_stop")

        for camera in self.cameras:
            camera = UsdGeom.Camera(camera)
            camera.GetFocalLengthAttr().Set(focal_length)
            camera.GetFocusDistanceAttr().Set(focus_distance)
            camera.GetHorizontalApertureAttr().Set(horiz_aperture)
            camera.GetVerticalApertureAttr().Set(vert_aperture)
            camera.GetFStopAttr().Set(f_stop)

        # Set viewports
        carb.settings.acquire_settings_interface().set_int("/app/renderer/resolution/width", -1)
        carb.settings.acquire_settings_interface().set_int("/app/renderer/resolution/height", -1)

        self.viewports = []
        for i in range(len(self.cameras)):
            if i == 0:
                viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().get_instance("Viewport")
            else:
                viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
            viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window.set_texture_resolution(self.sample("img_width"), self.sample("img_height"))
            viewport_window.set_active_camera(camera_prim_paths[i])

            if self.stereo:
                if i == 0:
                    viewport_name = "left"
                else:
                    viewport_name = "right"
            else:
                viewport_name = "mono"
            self.viewports.append((viewport_name, viewport_window))

        self.sim_context.render()
        self.sim_app.update()

        # Set viewport window size
        if self.stereo:
            left_viewport = omni.ui.Workspace.get_window("Viewport")
            right_viewport = omni.ui.Workspace.get_window("Viewport 2")
            right_viewport.dock_in(left_viewport, omni.ui.DockPosition.RIGHT)

        self.intrinsics = [self.get_intrinsics(camera) for camera in self.cameras]
        # print(self.intrinsics)

    def translate(self, coord):
        """ Translate each camera asset. Find stereo positions, if needed. """

        self.coord = coord

        if self.sample("stereo"):
            self.coords = self.get_stereo_coords(self.coord, self.rotation)
        else:
            self.coords = [self.coord]

        for i, camera in enumerate(self.cameras):
            viewport_name, viewport_window = self.viewports[i]
            viewport_window.set_camera_position(
                str(camera.GetPath()), self.coords[i][0], self.coords[i][1], self.coords[i][2], True
            )

    def rotate(self, rotation):
        """ Rotate each camera asset. """

        from pxr import UsdGeom

        self.rotation = rotation

        for i, camera in enumerate(self.cameras):
            offset_cam_rot = self.rotation + np.array((90, 0, 270), dtype=np.float32)
            UsdGeom.XformCommonAPI(camera).SetRotate(offset_cam_rot.tolist())

    def place_in_scene(self):
        """ Place camera in scene. """

        rotation = self.get_initial_rotation()
        self.rotate(rotation)

        coord = self.get_initial_coord()
        self.translate(coord)

        self.step(0)

    def get_stereo_coords(self, coord, rotation):
        """ Convert camera center coord and rotation and return stereo camera coords. """

        coords = []
        for i in range(len(self.cameras)):
            sign = 1 if i == 0 else -1
            theta = np.radians(rotation[0] + sign * 90)
            phi = np.radians(rotation[1])

            radius = self.sample("stereo_baseline") / 2

            # Add offset such that center of stereo cameras is at cam_coord
            x = coord[0] + radius * np.cos(theta) * np.cos(phi)
            y = coord[1] + radius * np.sin(theta) * np.cos(phi)
            z = coord[2] + radius * sign * np.sin(phi)

            coords.append(np.array((x, y, z)))

        return coords

    def get_intrinsics(self, camera):
        """ Compute, print, and return camera intrinsics. """

        from omni.syntheticdata import helpers

        width = self.sample("img_width")
        height = self.sample("img_height")

        aspect_ratio = width / height
        
        camera.GetAttribute("clippingRange").Set((0.01, 1000000)) # set clipping range
        near, far = camera.GetAttribute("clippingRange").Get()

        focal_length = camera.GetAttribute("focalLength").Get()
        horiz_aperture = camera.GetAttribute("horizontalAperture").Get()
        vert_aperture = camera.GetAttribute("verticalAperture").Get()

        horiz_fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        horiz_fov = np.degrees(horiz_fov)
        vert_fov = 2 * math.atan(vert_aperture / (2 * focal_length))
        vert_fov = np.degrees(vert_fov)

        fx = width * focal_length / horiz_aperture
        fy = height * focal_length / vert_aperture
        cx = width * 0.5
        cy = height * 0.5

        proj_mat = helpers.get_projection_matrix(np.radians(horiz_fov), aspect_ratio, near, far)

        with np.printoptions(precision=2, suppress=True):
            proj_mat_str = str(proj_mat)

        Logger.print("")
        Logger.print("Camera intrinsics")
        Logger.print("- width, height: {}, {}".format(round(width), round(height)))
        Logger.print("- focal_length: {}".format(focal_length, 2))
        Logger.print(
            "- horiz_aperture, vert_aperture: {}, {}".format(round(horiz_aperture, 2), round(vert_aperture, 2))
        )
        Logger.print("- horiz_fov, vert_fov: {}, {}".format(round(horiz_fov, 2), round(vert_fov, 2)))
        Logger.print("- focal_x, focal_y: {}, {}".format(round(fx, 2), round(fy, 2)))
        Logger.print("- proj_mat: \n {}".format(str(proj_mat_str)))
        Logger.print("")

        cam_intrinsics = {
            "width": width,
            "height": height,
            "focal_length": focal_length,
            "horiz_aperture": horiz_aperture,
            "vert_aperture": vert_aperture,
            "horiz_fov": horiz_fov,
            "vert_fov": vert_fov,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "proj_mat": proj_mat,
            "near":near, 
            "far":far
        }

        return cam_intrinsics
    
    # zhili added
    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)
    
    def translate_rotate(self,target=(0,0,0)):
        """ Translate each camera asset. Find stereo positions, if needed. """

        # self.coord = coord

        # if self.sample("stereo"):
        #     self.coords = self.get_stereo_coords(self.coord, self.rotation)
        # else:
        #     self.coords = [self.coord]

        for i, camera in enumerate(self.cameras):
            viewport_name, viewport_window = self.viewports[i]
            # viewport_window.set_camera_position(
            #     str(camera.GetPath()), self.coords[i][0], self.coords[i][1], self.coords[i][2], True
            # )
            viewport_window.set_camera_target(str(camera.GetPath()), target[0], target[1], target[2], True)

