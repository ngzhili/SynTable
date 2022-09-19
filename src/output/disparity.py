# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np


class DisparityConverter:
    """ For converting stereo depth maps to stereo disparity maps. """

    def __init__(self, depth_l, depth_r, fx, fy, cx, cy, baseline):
        """ Construct DisparityConverter. """

        self.depth_l = np.array(depth_l, dtype=np.float32)
        self.depth_r = np.array(depth_r, dtype=np.float32)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

    def compute_disparity(self):
        """ Computes a disparity map from left and right depth maps. """

        # List all valid depths in the depth map
        (y, x) = np.nonzero(np.invert(np.isnan(self.depth_l)))
        depth_l = self.depth_l[y, x]
        depth_r = self.depth_r[y, x]

        # Compute disparity maps
        disp_lr = self.depth_to_disparity(x, depth_l, self.baseline)
        disp_rl = self.depth_to_disparity(x, depth_r, -self.baseline)

        # Use numpy vectorization to get pixel coordinates
        disp_l, disp_r = np.zeros(self.depth_l.shape), np.zeros(self.depth_r.shape)

        disp_l[y, x] = np.abs(disp_lr)
        disp_r[y, x] = np.abs(disp_rl)

        disp_l = np.array(disp_l, dtype=np.float32)
        disp_r = np.array(disp_r, dtype=np.float32)

        return disp_l, disp_r

    def depth_to_disparity(self, x, depth, baseline_offset):
        """ Convert depth map to disparity map. """

        # Backproject image to 3D world
        x_est = (x - self.cx) * (depth / self.fx)
        # Add baseline offset to 3D world position
        x_est += baseline_offset
        # Project to the other stereo image domain
        x_pt = self.cx + (x_est / depth * self.fx)
        # Compute disparity with the x-axis only since the left and right images are rectified
        disp = x_pt - x

        return disp
