# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from scene.object1 import Object
import numpy as np
import os

class RoomFace(Object):
    """ For managing an Xform asset in Isaac Sim. """

    def __init__(self, sim_app, sim_context, path, prefix, coord, rotation, scaling):
        """ Construct Object. """

        self.coord = coord
        self.rotation = rotation
        self.scaling = scaling

        super().__init__(sim_app, sim_context, "", path, prefix, None, None)

    def load_asset(self):
        """ Create asset from object parameters. """

        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import move_prim

        from pxr import PhysxSchema, UsdPhysics

        if self.prefix == "floor":
            # Create invisible ground plane
            path = "/World/Room/ground"
            planeGeom = PhysxSchema.Plane.Define(self.stage, path)
            planeGeom.CreatePurposeAttr().Set("guide")
            planeGeom.CreateAxisAttr().Set("Z")
            prim = self.stage.GetPrimAtPath(path)
            UsdPhysics.CollisionAPI.Apply(prim)

        # Create plane
        from omni.kit.primitive.mesh import CreateMeshPrimWithDefaultXformCommand

        CreateMeshPrimWithDefaultXformCommand(prim_type="Plane").do()
        move_prim(path_from="/Plane", path_to=self.path)

        self.prim = self.stage.GetPrimAtPath(self.path)
        self.xform_prim = XFormPrim(self.path)

    def place_in_scene(self):
        """ Scale, rotate, and translate asset. """

        self.translate(self.coord)
        self.rotate(self.rotation)
        self.scale(self.scaling)

    def step(self):
        """ Room Face does not update in a scene's sequence. """

        return



class RoomTable(Object):
    """ For managing an Xform asset in Isaac Sim. """
    def __init__(self, sim_app, sim_context, ref, path, prefix, camera, group):
        super().__init__(sim_app, sim_context, ref, path, prefix, camera, group, None)

        
    def load_asset(self):
        """ Create asset from object parameters. """

        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils import prims
        # print(self.path)
        # Create object
        self.prim = prims.create_prim(self.path, "Xform", semantic_label="[[scenario]]")
        self.xform_prim = XFormPrim(self.path)

        nested_path = os.path.join(self.path, "nested_prim")
        self.nested_prim = prims.create_prim(nested_path, "Xform", usd_path=self.ref, semantic_label="[[scenario]]")
        self.nested_xform_prim = XFormPrim(nested_path)

        self.add_material()
        self.add_collision()

