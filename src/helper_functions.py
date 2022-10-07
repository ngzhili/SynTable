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