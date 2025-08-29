from __future__ import annotations
import numpy as np


def create_circlular_mask(shape, centroid, radius, shrink_ratio=1, scale=1):
    inv_scale = 1/scale
    c_x = np.round(inv_scale*centroid[0])
    c_y = np.round(inv_scale*centroid[1])

    nx = np.linspace(-c_x, shape[0] - c_x - 1, shape[0])
    ny = np.linspace(-c_y, shape[1] - c_y - 1, shape[1])
    mesh_x, mesh_y = np.meshgrid(nx, ny)
    c_mask = mesh_x ** 2 + mesh_y ** 2 <= (shrink_ratio*inv_scale*radius) ** 2

    return c_mask

def mask_and_crop(img_rgb, cxcy_r, dish_offset: float):
    """
    img_rgb: uint8 RGB image (H,W,3)
    Returns: crop_rgb, crop_orig_rgb, mask_thresh_crop (bool), (x_min,y_min)
    """
    cx, cy, r = cxcy_r
    r_off = r - dish_offset

    mask = create_circlular_mask(img_rgb.shape[1::-1], (cx, cy), r_off)
    mask_thresh = create_circlular_mask(img_rgb.shape[1::-1], (cx, cy), r_off - 8)

    out = img_rgb.copy()
    out[~mask] = 0

    x_min = max(int(cy - r_off), 0); x_max = min(int(cy + r_off), img_rgb.shape[0])
    y_min = max(int(cx - r_off), 0); y_max = min(int(cx + r_off), img_rgb.shape[1])

    crop = out[x_min:x_max, y_min:y_max]
    crop_orig = img_rgb[x_min:x_max, y_min:y_max]
    mask_thresh_crop = mask_thresh[x_min:x_max, y_min:y_max]
    return crop, crop_orig, mask_thresh_crop, (x_min, y_min)
