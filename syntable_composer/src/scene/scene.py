
import time

from output import Logger
from sampling import Sampler
from scene import Camera, Light, Object, Room


class SceneManager:
    """ For managing scene set-up and generation. """

    def __init__(self, sim_app, sim_context):
        """ Construct SceneManager. Set-up scenario in Isaac Sim. """

        import omni

        self.sim_app = sim_app
        self.sim_context = sim_context

        self.stage = omni.usd.get_context().get_stage()

        self.sample = Sampler().sample

        self.scene_path = "/World/Scene"
        self.scenario_label = "[[scenario]]"
        self.play_frame = False
        self.objs = []
        self.lights = []

        self.setup_scenario()
        self.camera = Camera(self.sim_app, self.sim_context, "/World/CameraRig", None, group=None)

    def setup_scenario(self):
        """ Load in base scenario(s) """

        import omni
        from omni.isaac.core import SimulationContext
        from omni.isaac.core.utils import stage
        from omni.isaac.core.utils.stage import get_stage_units

        cached_physics_dt = self.sim_context.get_physics_dt()
        cached_rendering_dt = self.sim_context.get_rendering_dt()
        cached_stage_units = get_stage_units()

        self.room = None
        if self.sample("scenario_room_enabled"):
            # Generate a parameterizable room
            self.room = Room(self.sim_app, self.sim_context)
        else:
            # Load in a USD scenario
            self.load_scenario_model()

        # Re-initialize context after we open a stage
        self.sim_context = SimulationContext(
            physics_dt=cached_physics_dt, rendering_dt=cached_rendering_dt, stage_units_in_meters=cached_stage_units
        )

        self.stage = omni.usd.get_context().get_stage()

        # Set the up axis to the z axis
        stage.set_stage_up_axis("z")

        # Set scenario label to stage prims
        self.set_scenario_label()

        # Reset rendering settings
        self.sim_app.reset_render_settings()

    def set_scenario_label(self):
        """ Set scenario label to all prims in stage. """

        from pxr import Semantics

        for prim in self.stage.Traverse():
            path = prim.GetPath()
            if path == "/World":
                continue
            if not prim.HasAPI(Semantics.SemanticsAPI):
                sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
            else:
                sem = Semantics.SemanticsAPI.Get(prim, "Semantics")
                continue

            typeAttr = sem.GetSemanticTypeAttr()
            dataAttr = sem.GetSemanticDataAttr()
            typeAttr.Set("class")
            dataAttr.Set(self.scenario_label)

    def load_scenario_model(self):
        """ Load in a USD scenario. """

        from omni.isaac.core.utils.stage import open_stage

        # Load in base scenario from Nucleus
        if self.sample("scenario_model"):
            scenario_ref = self.sample("nucleus_server") + self.sample("scenario_model")
            open_stage(scenario_ref)

    def populate_scene(self):
        """ Populate a sample's scene a camera, objects, and lights. """

        # Update camera
        self.camera.place_in_scene()

        # Iterate through each group
        self.objs = []
        self.lights = []
        for group in self.sample("groups"):
            # Spawn objects
            num_objs = self.sample("obj_count", group=group)
            for i in range(num_objs):
                path = "{}/Objects/object_{}".format(self.scene_path, len(self.objs))
                ref = self.sample("nucleus_server") + self.sample("obj_model", group=group)
                obj = Object(self.sim_app, self.sim_context, ref, path, "obj", self.camera, group)
                self.objs.append(obj)

            # Spawn lights
            num_lights = self.sample("light_count", group=group)
            for i in range(num_lights):
                path = "{}/Lights/lights_{}".format(self.scene_path, len(self.lights))
                light = Light(self.sim_app, self.sim_context, path, self.camera, group)
                self.lights.append(light)

        # Update room
        if self.room:
            self.room.update()

        # Add skybox, if needed
        self.add_skybox()

    def update_scene(self, step_time=None, step_index=0):
        """ Update Omniverse after scene is generated. """

        from omni.isaac.core.utils.stage import is_stage_loading

        # Step positions of objs and lights
        if step_time:
            self.camera.step(step_time)

            for obj in self.objs:
                obj.step(step_time)

            for light in self.lights:
                light.step(step_time)

        # Wait for scene to finish loading
        while is_stage_loading():
            self.sim_context.render()

        # Determine if scene is played
        scene_assets = self.objs + self.lights
        self.play_frame = any([asset.physics for asset in scene_assets])

        # Play scene, if needed
        if self.play_frame and step_index == 0:
            Logger.print("physically simulating...")
            self.sim_context.play()
            render = not self.sample("headless")

            sim_time = self.sample("physics_simulate_time")
            frames_to_simulate = int(sim_time * 60) + 1
            for i in range(frames_to_simulate):
                self.sim_context.step(render=render)

        # Napping
        if self.sample("nap"):
            print("napping")
            while True:
                self.sim_context.render()

        # Update
        if step_index == 0:
            Logger.print("loading textures...")
            self.sim_context.render()

        # Pausing
        if step_index == 0:
            pause_time = self.sample("pause")

            start_time = time.time()
            while time.time() - start_time < pause_time:
                self.sim_context.render()

    def add_skybox(self):
        """ Add a DomeLight that creates a textured skybox, if needed. """

        from pxr import UsdGeom, UsdLux
        from omni.isaac.core.utils.prims import create_prim

        sky_texture = self.sample("sky_texture")
        sky_light_intensity = self.sample("sky_light_intensity")

        if sky_texture:
            create_prim(
                prim_path="{}/Lights/skybox".format(self.scene_path),
                prim_type="DomeLight",
                attributes={
                    UsdLux.Tokens.intensity: sky_light_intensity,
                    UsdLux.Tokens.specular: 1,
                    UsdLux.Tokens.textureFile: self.sample("nucleus_server") + sky_texture,
                    UsdLux.Tokens.textureFormat: UsdLux.Tokens.latlong,
                    UsdGeom.Tokens.visibility: "inherited",
                },
            )

    def prepare_scene(self, index):
        """ Scene preparation step. """

        self.valid_sample = True
        Logger.start_log_entry(index)
        Logger.print("Scene: " + str(index) + "\n")

    def finish_scene(self):
        """ Scene finish step. Clean-up variables, Isaac Sim stage. """

        from omni.isaac.core.utils.prims import delete_prim

        self.objs = []
        self.lights = []
        delete_prim(self.scene_path)
        delete_prim("/Looks")
        self.sim_context.stop()
        self.sim_context.render()
        self.play_frame = False
        Logger.finish_log_entry()
