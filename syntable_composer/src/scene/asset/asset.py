
from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.spatial.transform import Rotation

from output import Logger
from sampling import Sampler


class Asset(ABC):
    """ For managing an asset in Isaac Sim. """

    def __init__(self, sim_app, sim_context, path, prefix, name, group=None, camera=None):
        """ Construct Asset. """

        self.sim_app = sim_app
        self.sim_context = sim_context
        self.path = path
        self.camera = camera
        self.name = name
        self.prefix = prefix

        self.stage = self.sim_context.stage
        self.sample = Sampler(group=group).sample

        self.class_name = self.__class__.__name__

        if self.class_name != "RoomFace":
            self.vel = self.sample(self.concat("vel"))
            self.rot_vel = self.sample(self.concat("rot_vel"))

            self.accel = self.sample(self.concat("accel"))
            self.rot_accel = self.sample(self.concat("rot_accel"))
        self.label = group

        self.physics = False

    @abstractmethod
    def place_in_scene(self):
        """ Place asset in scene. """
        pass

    def is_given(self, param):
        """ Is a parameter value is given. """

        if type(param) in (np.ndarray, list, tuple, str):
            return len(param) > 0
        elif type(param) is float:
            return not math.isnan(param)
        else:
            return param is not None

    def translate(self, coord, xform_prim=None):
        """ Translate asset. """

        if xform_prim is None:
            xform_prim = self.xform_prim
        xform_prim.set_world_pose(position=coord)

    def scale(self, scaling, xform_prim=None):
        """ Scale asset uniformly across all axes. """

        if xform_prim is None:
            xform_prim = self.xform_prim
        xform_prim.set_local_scale(scaling)

    def rotate(self, rotation, xform_prim=None):
        """ Rotate asset. """

        from omni.isaac.core.utils.rotations import euler_angles_to_quat

        if xform_prim is None:
            xform_prim = self.xform_prim
        xform_prim.set_world_pose(orientation=euler_angles_to_quat(rotation.tolist(), degrees=True))

    def is_coord_camera_relative(self):
        return self.sample(self.concat("coord_camera_relative"))

    def is_rot_camera_relative(self):
        return self.sample(self.concat("rot_camera_relative"))

    def concat(self, parameter_suffix):
        """ Concatenate the parameter prefix and suffix. """

        return self.prefix + "_" + parameter_suffix

    def get_initial_coord(self):
        """ Get coordinates of asset across 3 axes. """

        if self.is_coord_camera_relative():
            cam_coord = self.camera.coords[0]
            cam_rot = self.camera.rotation
            horiz_fov = -1 * self.camera.intrinsics[0]["horiz_fov"]
            vert_fov = self.camera.intrinsics[0]["vert_fov"]

            radius = self.sample(self.concat("distance"))
            theta = horiz_fov * self.sample(self.concat("horiz_fov_loc")) / 2
            phi = vert_fov * self.sample(self.concat("vert_fov_loc")) / 2

            # Convert from polar to cartesian
            rads = np.radians(cam_rot[2] + theta)
            x = cam_coord[0] + radius * np.cos(rads)
            y = cam_coord[1] + radius * np.sin(rads)

            rads = np.radians(cam_rot[0] + phi)
            z = cam_coord[2] + radius * np.sin(rads)

            coord = np.array([x, y, z])
        else:
            coord = self.sample(self.concat("coord"))

        pretty_coord = tuple([round(v, 1) for v in coord.tolist()])
        Logger.print("adding {} {} at coords{}".format(self.prefix.upper(), self.name, pretty_coord))

        return coord

    def get_initial_rotation(self):
        """ Get rotation of asset across 3 axes. """

        rotation = self.sample(self.concat("rot"))
        rotation = np.array(rotation)

        if self.is_rot_camera_relative():
            cam_rot = self.camera.rotation
            rotation += cam_rot

        return rotation

    def step(self, step_time):
        """ Step asset forward in its sequence. """

        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        if self.class_name != "Camera":
            self.coord, quaternion = self.xform_prim.get_world_pose()
            self.coord = np.array(self.coord, dtype=np.float32)
            self.rotation = np.degrees(quat_to_euler_angles(quaternion))

        vel_vector = self.vel
        accel_vector = self.accel
        if self.sample(self.concat("movement") + "_" + self.concat("relative")):
            radians = np.radians(self.rotation)
            direction_cosine_matrix = Rotation.from_rotvec(radians).as_matrix()
            vel_vector = direction_cosine_matrix.dot(vel_vector)
            accel_vector = direction_cosine_matrix.dot(accel_vector)

        self.coord += vel_vector * step_time + 0.5 * accel_vector * step_time ** 2
        self.translate(self.coord)

        self.rotation += self.rot_vel * step_time + 0.5 * self.rot_accel * step_time ** 2
        self.rotate(self.rotation)
