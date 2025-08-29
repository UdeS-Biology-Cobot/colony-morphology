from __future__ import annotations
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from colony_morphology import regionprops_util as cb
from .metric import compactness as compute_compactness
from .metric import axes_closness as compute_axes_closness

def compute_region_properties(img_gray_u8, labels):
    extra = (
        cb.compactness,
        cb.nn_collision_distance,
        cb.nn_centroid_distance,
        cb.cell_quality,
        cb.discarded,
        cb.discarded_description,
        cb.axes_closness,
    )
    props = regionprops(labels, intensity_image=img_gray_u8, extra_properties=extra)

    # fill robust values & filter degenerate
    filtered = []
    for p in props:
        if p.perimeter <= 0:
            continue
        p.compactness = compute_compactness(p.area, p.perimeter) if p.perimeter else 0.0
        if p.axis_major_length == 0.0 or p.axis_minor_length == 0.0:
            p.axes_closness = 0.0
        else:
            p.axes_closness = compute_axes_closness(p.axis_major_length, p.axis_minor_length)
        filtered.append(p)
    return filtered

def compute_nn_metrics(properties, nn_query_size: int):
    centroids = [p["centroid"] for p in properties]
    if not centroids:
        return
    tree = cKDTree(centroids)
    k = min(nn_query_size, len(centroids))
    for i, centroid in enumerate(centroids):
        dd, ii = tree.query(centroid, k)
        p = properties[i]
        if len(dd) > 1:
            p.nn_centroid_distance = dd[1]
        radius = p.equivalent_diameter_area / 2.0

        prev_nn_diam = float("-inf")
        prev_collision = float("+inf")
        for idx in range(1, len(ii)):
            pnn = properties[ii[idx]]
            nn_d = pnn.equivalent_diameter_area
            if nn_d > prev_nn_diam:
                prev_nn_diam = nn_d
                nn_radius = nn_d / 2.0
                collision = dd[idx] - (radius + nn_radius)
                if collision < prev_collision:
                    prev_collision = collision
                    p.nn_collision_distance = collision
