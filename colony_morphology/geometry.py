import numpy as np
import shapely
from skimage import measure
from skimage.draw import disk
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks


def create_circlular_mask(shape, centroid, radius, shrink_ratio=1, scale=1):
    inv_scale = 1/scale
    c_x = np.round(inv_scale*centroid[0])
    c_y = np.round(inv_scale*centroid[1])

    nx = np.linspace(-c_x, shape[0] - c_x - 1, shape[0])
    ny = np.linspace(-c_y, shape[1] - c_y - 1, shape[1])
    mesh_x, mesh_y = np.meshgrid(nx, ny)
    c_mask = mesh_x ** 2 + mesh_y ** 2 <= (shrink_ratio*inv_scale*radius) ** 2

    return c_mask


# https://github.com/morris-lab/Colony-counter
def detect_circle_by_canny(image_bw, radius=395, n_peaks=20):
    edges = canny(image_bw, sigma=2)
    hough_res = hough_circle(edges, [radius])
    accums, cx, cy, radii = hough_circle_peaks(hough_res, [radius],
                                               total_num_peaks=n_peaks)

    label = np.zeros_like(image_bw)
    ind = 1
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = disk((center_y, center_x), radius,
                                        shape=image_bw.shape)
        label[circy, circx] = ind
        ind += 1

    return label.astype(int)

def _get_radius(bbox):
    minr, minc, maxr, maxc = bbox
    r = np.array([[maxr - minr], [maxc - minc]]).mean()/2
    return r

def _bbox_to_center(bbox):
    minr, minc, maxr, maxc = bbox
    c = int(np.array([minc, maxc]).mean())
    r = int(np.array([minr, maxr]).mean())
    return r, c

def make_circle_label(bb_list, img_shape):

    # get radious
    radius = np.median([_get_radius(i) for i in bb_list])

    # draw circles in an image
    label = np.zeros(img_shape)
    id = 1
    for bb in bb_list:
        # get centroid
        r, c = _bbox_to_center(bb)

        # draw circle
        rr, cc = disk((r, c), radius)
        label[rr, cc] = id
        id += 1

    return label.astype(int)

def crop_circle(image, shrinkage_ratio=0.95):

    x, y = image.shape
    center = (int(x/2), int(y/2))
    diameter = min(center)

    threshold = (diameter*shrinkage_ratio)**2

    # initialize mask
    mask = np.zeros_like(image)

    # crop as circle.
    for x_ in range(x):
        for y_ in range(y):
            dist = (x_ - center[0])**2 + (y_ - center[1])**2
            mask[x_, y_] = (dist < threshold)

    return image*mask



def detect_area_by_canny(image, n_samples=None, radius=395, n_peaks=20):
    """
    The method detects sample area in input image.
    Large, white and circle-like object in the input image will be
    detected as sample area.

    Args:

    """
    # 1. Segmentation
    bw = image.copy()

    # detect circles by canny method
    labeled = detect_circle_by_canny(bw, radius=radius, n_peaks=n_peaks)


    # 2. region props
    props = np.array(measure.regionprops(label_image=labeled, intensity_image=image))
    bboxs = np.array([prop.bbox for prop in props])
    areas = np.array([prop.area for prop in props])
    cordinates = np.array([prop.centroid for prop in props])
    eccentricities = np.array([prop.eccentricity for prop in props])
    intensity = np.array([prop.intensity_image.mean() for prop in props])


    # 3. filter object

    selected = (areas >= np.percentile(areas, 90)) & (eccentricities < 0.3)


    # update labels
    labeled = make_circle_label(bb_list=bboxs[selected], img_shape=image.shape)

    # Region props again
    props = np.array(measure.regionprops(label_image=labeled, intensity_image=image))
    bboxs = np.array([prop.bbox for prop in props])
    areas = np.array([prop.area for prop in props])
    cordinates = np.array([prop.centroid for prop in props])
    eccentricities = np.array([prop.eccentricity for prop in props])
    intensity = np.array([prop.intensity_image.mean() for prop in props])

    if not n_samples is None:
        ind = np.argsort(intensity)[-n_samples:]
        props = props[ind]
        bboxs = bboxs[ind]
        areas = areas[ind]
        cordinates = cordinates[ind]
        eccentricities = eccentricities[ind]



    # sort by cordinate y
    idx = np.argsort(cordinates[:, 0])

    # self._props = props[idx]
    # self.props["bboxs"] = bboxs[idx]
    # self.props["areas"] = areas[idx]
    # self.props["cordinates"] = cordinates[idx]
    # self.props["eccentricities"] = eccentricities[idx]
    # self.props["names"] = [f"sample_{i}" for i in range(len(self.props["areas"]))]

    return props[idx]


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

