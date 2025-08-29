import numpy as np
import shapely
from skimage import measure


# https://forum.image.sc/t/calculating-distance-to-nearest-neighboring-cells-edge-not-centroid/78075/2
def prepare_box_for_contours(box, shape, pad=3):
    """Marginally pads a bounding box so that object boundaries
    are not on cropped image patch edges.
    """
    for i in range(2):
        box[i] = min(0, box[i] - pad)
        box[i+2] = max(shape[i], box[i] + pad)

    slices = tuple([slice(box[i], box[i+2]) for i in range(2)])
    top_left = np.array(box[:2])[None] # (1, 2)
    return slices, top_left

def make_polygons_from_mask(mask):
    """Constructs a polygon for each object in a mask. Returns
    a dict where each key is a label id and values are shapely polygons.
    """
    polygons = {}
    for rp in measure.regionprops(mask):
        # Faster to compute contours on small cell tiles than the whole image
        box_slices, box_top_left = prepare_box_for_contours(list(rp.bbox), mask.shape)
        label_mask = mask[box_slices] == rp.label

        label_cnts = np.concatenate(
            measure.find_contours(label_mask), axis=0
        )

        polygons[rp.label] = shapely.Polygon(label_cnts + box_top_left)

    return polygons

def pairwise_polygon_distance(polygons_dict):
    """Computes pairwise distance between all polygons in
    a dictionary. Returns a dictionary of distances.
    """
    distances = {l: {} for l in polygons_dict.keys()}
    for i in polygons_dict.keys():
        for j in polygons_dict.keys():
            # nested loop is slow but we cache results
            # to eliminate duplicate work
            if i != j and distances[i].get(j) is None:
                distances[i][j] = shapely.distance(polygons_dict[i], polygons_dict[j])

    return distances

def get_nn_distance(key, distances_dict):
    """Returns the nearest neighbor for a polygon
    along with the distance.
    """
    return min(distances_dict[key].items(), key=lambda x: x[1])


# https://stackoverflow.com/a/70023212 
def intersects(box1, box2):
    """Checks if 2 bounding boxes overlaps with each other
    """
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

