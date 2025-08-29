from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_local
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def opencv_like_gray_u8(img_rgb):
    g = (0.299*img_rgb[...,0] + 0.587*img_rgb[...,1] + 0.114*img_rgb[...,2])
    return np.rint(g).astype(np.uint8)

def opencv_like_gaussian_u8(gray_u8, ksize=7):
    sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8   # OpenCV implicit sigma when sigmaX=0
    truncate = (ksize-1)/(2*sigma)
    g = gaussian(gray_u8, sigma=sigma, truncate=truncate, mode='nearest', preserve_range=True)
    return np.rint(g).astype(np.uint8)

def opencv_like_adapt_mean_inv_u8(blur_u8, block_size=9, C=2):
    T = threshold_local(blur_u8, block_size=block_size, method='mean', offset=C, mode='nearest')
    return (blur_u8 <= T)  # THRESH_BINARY_INV

def preprocess_and_segment(img_rgb_crop, mask_thresh_crop, close_radius=1, open_radius=1):
    """
    Returns dict with 'img_gray', 'img_mask_artifacts', 'closing', 'opening', 'dist', 'labels'
    """
    img_gray_u8 = opencv_like_gray_u8(img_rgb_crop)
    img_blur_u8 = opencv_like_gaussian_u8(img_gray_u8, ksize=7)
    img_threshold = opencv_like_adapt_mean_inv_u8(img_blur_u8, block_size=9, C=2)

    img_mask_artifacts = img_threshold.copy()
    img_mask_artifacts[~mask_thresh_crop] = 0

    if close_radius > 0:
        closing = binary_closing(img_mask_artifacts, disk(close_radius))
    else:
        closing = img_mask_artifacts

    if open_radius > 0:
        opening = binary_opening(closing, disk(open_radius))
    else:
        opening = closing


    dist = np.empty(opening.shape, dtype=float)
    ndi.distance_transform_edt(opening, distances=dist)

    coords = peak_local_max(dist, footprint=np.ones((3, 3)), labels=opening)
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers = ndi.label(peak_mask)[0]

    labels = watershed(dist, markers, mask=opening, connectivity=1, compactness=0)

    return {
        "img_gray": img_gray_u8,
        "img_mask_artifacts": img_mask_artifacts,
        "closing": closing,
        "opening": opening,
        "dist": dist,
        "labels": labels,
    }
