# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

from sampling.sample1 import Sampler
from scene.room_face1 import RoomFace, RoomTable


class Room:
    """ For managing a parameterizable rectangular prism centered at the origin. """

    def __init__(self, sim_app, sim_context):
        """ Construct Room. Generate room in Isaac SIM. """

        self.sim_app = sim_app
        self.sim_context = sim_context

        self.stage = self.sim_context.stage

        self.sample = Sampler().sample

        self.room = self.scenario_room()

    def scenario_room(self):
        """ Generate and return assets creating a rectangular prism at the origin. """

        wall_height = self.sample("wall_height")
        floor_size = self.sample("floor_size")
        self.room_faces = []

        faces = []
        coords = []
        scalings = []
        rotations = []
        if self.sample("floor"):
            faces.append("floor")
            coords.append((0, 0, 0))
            scalings.append((floor_size / 100, floor_size / 100, 1))
            rotations.append((0, 0, 0))

        if self.sample("wall"):
            faces.extend(4 * ["wall"])
            coords.append((floor_size / 2, 0, wall_height / 2))
            coords.append((0, floor_size / 2, wall_height / 2))
            coords.append((-floor_size / 2, 0, wall_height / 2))
            coords.append((0, -floor_size / 2, wall_height / 2))
            scalings.extend(4 * [(floor_size / 100, wall_height / 100, 1)])
            rotations.append((90, 0, 90))
            rotations.append((90, 0, 0))
            rotations.append((90, 0, 90))
            rotations.append((90, 0, 0))

        if self.sample("ceiling"):
            faces.append("ceiling")
            coords.append((0, 0, wall_height))
            scalings.append((floor_size / 100, floor_size / 100, 1))
            rotations.append((0, 0, 0))

        room = []

        for i, face in enumerate(faces):
            coord = np.array(coords[i])
            rotation = np.array(rotations[i])
            scaling = np.array(scalings[i])
            path = "/World/Room/{}_{}".format(face, i)
            room_face = RoomFace(self.sim_app, self.sim_context, path, face, coord, rotation, scaling)
            room.append(room_face)

        return room

    def update(self):
        """ Update room components. """

        for room_face in self.room:
            room_face.add_material()
