
from sampling import Sampler
from scene.asset import Asset


class Light(Asset):
    """ For managing a light asset in Isaac Sim. """

    def __init__(self, sim_app, sim_context, path, camera, group):
        """ Construct Light. """

        self.sample = Sampler(group=group).sample
        self.distant = self.sample("light_distant")
        self.directed = self.sample("light_directed")

        if self.distant:
            name = "distant_light"
        elif self.directed:
            name = "directed_light"
        else:
            name = "sphere_light"

        super().__init__(sim_app, sim_context, path, "light", name, camera=camera, group=group)

        self.load_light()
        self.place_in_scene()

    def place_in_scene(self):
        """ Place light in scene. """

        self.coord = self.get_initial_coord()
        self.translate(self.coord)
        self.rotation = self.get_initial_rotation()
        self.rotate(self.rotation)

    def load_light(self):
        """ Create a light in Isaac Sim. """

        from pxr import Sdf
        from omni.usd.commands import ChangePropertyCommand
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils import prims

        intensity = self.sample("light_intensity")
        color = tuple(self.sample("light_color") / 255)
        temp_enabled = self.sample("light_temp_enabled")
        temp = self.sample("light_temp")
        radius = self.sample("light_radius")
        focus = self.sample("light_directed_focus")
        focus_softness = self.sample("light_directed_focus_softness")

        attributes = {}
        if self.distant:
            light_shape = "DistantLight"
        elif self.directed:
            light_shape = "DiskLight"
            attributes["radius"] = radius
        else:
            light_shape = "SphereLight"
            attributes["radius"] = radius

        attributes["intensity"] = intensity
        attributes["color"] = color
        if temp_enabled:
            attributes["enableColorTemperature"] = True
            attributes["colorTemperature"] = temp

        self.prim = prims.create_prim(self.path, light_shape, attributes=attributes)
        self.xform_prim = XFormPrim(self.path)

        if self.directed:
            ChangePropertyCommand(prop_path=Sdf.Path(self.path + ".shaping:focus"), value=focus, prev=0.0).do()
            ChangePropertyCommand(
                prop_path=Sdf.Path(self.path + ".shaping:cone:softness"), value=focus_softness, prev=0.0
            )
