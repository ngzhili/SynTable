"""
Zhili's Replicator Composer Helper Functions
"""

import numpy as np

def compute_occluded_masks(mask1, mask2):
    """Computes occlusions between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # intersections and union
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection_mask = np.logical_and(mask1, mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection/(mask1_area+mask2_area-intersection)

    return iou, intersection_mask.astype(float)


import pycocotools.mask as mask_util
import cv2


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        try:
            # print(self.polygons)
            p = mask_util.frPyObjects(self.polygons, self.height, self.width)
            # print(p)
            p = mask_util.merge(p)
            # print(p)
            bbox = mask_util.toBbox(p)
            # print(bbox)
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
        except:
            print(f"Encountered error while generating bounding boxes from mask polygons: {self.polygons}")
            print("self.polygons:",self.polygons)

            bbox = np.array([0,0,0,0])
        return bbox

# def camera_orbit_coord(r = 12,z = 10):
#     """
#     constraints camera loc to a circular orbit around world origin
#     """
#     circle_x,circle_y = 0,0
#     a = random.uniform(0,2 * math.pi)
#     x = r * math.cos(a) + circle_x
#     y = r * math.sin(a) + circle_y
#     return np.array([x,y,z])

# def normalize(Vector):
#     return Vector / np.linalg.norm(Vector)
# def camera_orbit_rot(cam_coord_w, target_coord_w=np.array([0,0,8])):
#     """
#     constraints camera pose to lookat world origin
#     """
#     forwardVector = cam_coord_w - target_coord_w
#     forwardVector = normalize(forwardVector) #normalize

#     tempUpVector = np.array([0,1,0])
#     rightVector = np.cross(tempUpVector,forwardVector)
#     rightVector = normalize(rightVector)

#     upVector = np.cross(forwardVector,rightVector)
#     upVector = normalize(upVector)

#     # all 3 vectors are orthonormal
#     translationX = np.dot(cam_coord_w, rightVector)
#     translationY = np.dot(cam_coord_w, upVector)
#     translationZ = np.dot(cam_coord_w, forwardVector)

#     # build transformation matrix (Lookat matrix), transform from world-space to camera-space
#     lookAt = np.array([[rightVector[0],rightVector[1],rightVector[2],translationX],
#                         [upVector[0],upVector[1],upVector[2],translationY],
#                         [forwardVector[0],forwardVector[1],forwardVector[2],translationZ],
#                         [0,0,0,1]                                 
#                         ])

#     # x = random.uniform(-20,-40)
#     # y = 0
#     # z = 0 
#     return np.array([translationX,translationY,translationZ])
