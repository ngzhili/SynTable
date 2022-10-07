# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os

from scene.asset import Asset


class Object(Asset):
    """ For managing an Xform asset in Isaac Sim. """

    def __init__(self, sim_app, sim_context, ref, path, prefix, camera, group):
        """ Construct Object. """

        self.ref = ref
        name = self.ref[self.ref.rfind("/") + 1 : self.ref.rfind(".")]

        super().__init__(sim_app, sim_context, path, prefix, name, camera=camera, group=group)

        self.load_asset()
        self.place_in_scene()

        if self.class_name != "RoomFace" and self.sample("obj_physics"):
            self.add_physics()

    def load_asset(self):
        """ Create asset from object parameters. """

        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils import prims

        # Create object
        self.prim = prims.create_prim(self.path, "Xform", semantic_label=self.label)
        self.xform_prim = XFormPrim(self.path)

        nested_path = os.path.join(self.path, "nested_prim")
        self.nested_prim = prims.create_prim(nested_path, "Xform", usd_path=self.ref, semantic_label=self.label)
        self.nested_xform_prim = XFormPrim(nested_path)

        self.add_material()

    def place_in_scene(self):
        """ Scale, rotate, and translate asset. """

        # Get asset dimensions
        min_bound, max_bound = self.get_bounds()
        size = max_bound - min_bound

        # Get asset scaling
        obj_size_is_enabled = self.sample("obj_size_enabled")
        if obj_size_is_enabled:
            obj_size = self.sample("obj_size")
            max_size = max(size)
            self.scaling = obj_size / max_size
        else:
            self.scaling = self.sample("obj_scale")

        # Offset nested asset
        obj_centered = self.sample("obj_centered")
        if obj_centered:
            offset = (max_bound + min_bound) / 2
            self.translate(-offset, xform_prim=self.nested_xform_prim)

        # Scale asset
        self.scaling = np.array([self.scaling, self.scaling, self.scaling])
        self.scale(self.scaling)

        # Get asset coord and rotation
        self.coord = self.get_initial_coord()
        self.rotation = self.get_initial_rotation()

        # Rotate asset
        self.rotate(self.rotation)

        # Place asset
        self.translate(self.coord)

    def get_bounds(self):
        """ Compute min and max bounds of an asset. """

        from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache, recompute_extents

        # recompute_extents(self.nested_prim)
        cache = create_bbox_cache()
        bound = compute_aabb(cache, self.path).tolist()

        min_bound = np.array(bound[:3])
        max_bound = np.array(bound[3:])

        return min_bound, max_bound

    def add_material(self):
        """ Add material to asset, if needed. """

        from pxr import UsdShade

        material = self.sample(self.concat("material"))
        color = self.sample(self.concat("color"))
        texture = self.sample(self.concat("texture"))
        texture_scale = self.sample(self.concat("texture_scale"))
        texture_rot = self.sample(self.concat("texture_rot"))
        reflectance = self.sample(self.concat("reflectance"))
        metallic = self.sample(self.concat("metallicness"))

        mtl_prim_path = None
        if self.is_given(material):
            # Load a material
            mtl_prim_path = self.load_material_from_nucleus(material)
        elif self.is_given(color) or self.is_given(texture):
            # Load a new material
            mtl_prim_path = self.create_material()

        if mtl_prim_path:
            # Update material properties and assign to asset
            mtl_prim = self.update_material(
                mtl_prim_path, color, texture, texture_scale, texture_rot, reflectance, metallic
            )
            UsdShade.MaterialBindingAPI(self.prim).Bind(mtl_prim, UsdShade.Tokens.strongerThanDescendants)

    def load_material_from_nucleus(self, material):
        """ Create material from Nucleus path. """

        from pxr import Sdf
        from omni.usd.commands import CreateMdlMaterialPrimCommand

        mtl_url = self.sample("nucleus_server") + material

        left_index = material.rfind("/") + 1 if "/" in material else 0
        right_index = material.rfind(".") if "." in material else -1
        mtl_name = material[left_index:right_index]

        left_index = self.path.rfind("/") + 1 if "/" in self.path else 0
        path_name = self.path[left_index:]

        mtl_prim_path = "/Looks/" + mtl_name + "_" + path_name
        mtl_prim_path = Sdf.Path(mtl_prim_path.replace("-", "_"))

        CreateMdlMaterialPrimCommand(mtl_url=mtl_url, mtl_name=mtl_name, mtl_path=mtl_prim_path).do()

        return mtl_prim_path

    def create_material(self):
        """ Create a OmniPBR material with provided properties and assign to asset. """

        from pxr import Sdf
        import omni
        from omni.isaac.core.utils.prims import move_prim
        from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary

        mtl_created_list = []
        CreateAndBindMdlMaterialFromLibrary(
            mdl_name="OmniPBR.mdl", mtl_name="OmniPBR", mtl_created_list=mtl_created_list
        ).do()

        mtl_prim_path = Sdf.Path(mtl_created_list[0])
        new_mtl_prim_path = omni.usd.get_stage_next_free_path(self.stage, "/Looks/OmniPBR", False)
        move_prim(path_from=mtl_prim_path, path_to=new_mtl_prim_path)
        mtl_prim_path = new_mtl_prim_path

        return mtl_prim_path

    def update_material(self, mtl_prim_path, color, texture, texture_scale, texture_rot, reflectance, metallic):
        """ Update properties of an existing material. """

        import omni
        from pxr import Sdf, UsdShade

        mtl_prim = UsdShade.Material(self.stage.GetPrimAtPath(mtl_prim_path))

        if self.is_given(color):
            color = tuple(color / 255)
            omni.usd.create_material_input(mtl_prim, "diffuse_color_constant", color, Sdf.ValueTypeNames.Color3f)
            omni.usd.create_material_input(mtl_prim, "diffuse_tint", color, Sdf.ValueTypeNames.Color3f)

        if self.is_given(texture):
            texture = self.sample("nucleus_server") + texture
            omni.usd.create_material_input(mtl_prim, "diffuse_texture", texture, Sdf.ValueTypeNames.Asset)

        if self.is_given(texture_scale):
            texture_scale = 1 / texture_scale
            omni.usd.create_material_input(
                mtl_prim, "texture_scale", (texture_scale, texture_scale), Sdf.ValueTypeNames.Float2
            )

        if self.is_given(texture_rot):
            omni.usd.create_material_input(mtl_prim, "texture_rotate", texture_rot, Sdf.ValueTypeNames.Float)

        if self.is_given(reflectance):
            roughness = 1 - reflectance
            omni.usd.create_material_input(
                mtl_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float
            )
        if self.is_given(metallic):
            omni.usd.create_material_input(mtl_prim, "metallic_constant", metallic, Sdf.ValueTypeNames.Float)

        return mtl_prim

    def add_physics(self):
        """ Make asset a rigid body to enable gravity and collision. """

        from omni.isaac.core.utils.prims import get_all_matching_child_prims, get_prim_at_path
        from omni.physx.scripts import utils
        from pxr import UsdPhysics

        def is_rigid_body(prim_path):
            prim = get_prim_at_path(prim_path)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return True
            return False

        has_physics_already = len(get_all_matching_child_prims(self.path, predicate=is_rigid_body)) > 0
        if has_physics_already:
            self.physics = True
            return

        utils.setRigidBody(self.prim, "convexHull", False)
        # Set mass to 1 kg
        mass_api = UsdPhysics.MassAPI.Apply(self.prim)
        mass_api.CreateMassAttr(1)
        self.physics = True

    # zhili added
    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)

    def off_physics_prim(self):
        """ Turn Object Physics """
        self.physics = False

    def off_prim(self):
        """ Turn Object Visibility off """
        from omni.isaac.core.utils import prims
        prims.set_prim_visibility(self.prim, False)

        #print("\nTurn off visibility of prim;",self.prim)
        #print("\n")
    
    def on_prim(self):
        """ Turn Object Visibility on """
        from omni.isaac.core.utils import prims
        prims.set_prim_visibility(self.prim, True)

        #print("\nTurn on visibility of prim;",self.prim)
        #print("\n")

    