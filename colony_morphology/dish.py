from __future__ import annotations
import numpy as np
from scipy.optimize import leastsq
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, rescale

from skimage.draw import disk, line
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks


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



# Fit Circle Through New Points
def circle_residuals(params, points):
    x0, y0, r = params
    return np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r

def find_brightest_with_dark_neighbor(image, points, center, threshold=200, search_distance=20):
    """
    Find the brightest pixel along the line from each perimeter point to the center and in the opposite direction.

    Args:
        image (ndarray): Grayscale image.
        points (ndarray): Array of (x, y) coordinates of perimeter points.
        center (tuple): (x, y) coordinates of the center point.
        threshold (int, optional): Brightness threshold to identify bright pixels. Default is 200.
        search_distance (int, optional): Maximum distance to search along the line.

    Returns:
        list: List of tuples (x, y) of valid bright pixels.
    """
    bright_points = []  # Store valid bright pixels

    for x, y in points:
        # Get the line between the perimeter point and the center
        rr_to_center, cc_to_center = line(int(y), int(x), int(center[1]), int(center[0]))

        # Calculate the opposite direction (away from center)
        away_x = int(x + (x - center[0]))
        away_y = int(y + (y - center[1]))
        rr_away, cc_away = line(int(y), int(x), int(away_y), int(away_x))

        found_points = []  # Store potential bright points
        for rr, cc in [(rr_to_center, cc_to_center), (rr_away, cc_away)]:
            # Limit the search to the specified distance
            search_limit = min(len(rr), search_distance)

            # Iterate through the line points up to search_distance
            for i in range(search_limit):
                ny, nx = rr[i], cc[i]

                # Check if within valid bounds and meets the threshold
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and image[ny, nx] >= threshold:
                    found_points.append((nx, ny))
                    break  # Stop searching after finding the first valid pixel

        # Select the point furthest from the center if both directions found points
        if len(found_points) == 2:
            dist1 = np.sqrt((found_points[0][0] - center[0]) ** 2 + (found_points[0][1] - center[1]) ** 2)
            dist2 = np.sqrt((found_points[1][0] - center[0]) ** 2 + (found_points[1][1] - center[1]) ** 2)

            # Choose the one further away from the center
            chosen_point = found_points[0] if dist1 > dist2 else found_points[1]
            bright_points.append(chosen_point)

        # If only one point is found, use it
        elif len(found_points) == 1:
            bright_points.append(found_points[0])

    return bright_points



def detect_dish_circle(img_gray_u8, dish_diameter: float, scale: float):
    """
    Return adjusted (cx, cy, r) in original pixels using canny+hough then refinement.
    img_gray_u8: uint8 grayscale (H,W)
    """
    # small rescale for fast Hough
    rescaled = rescale(img_gray_u8, scale=scale, anti_aliasing=True)
    edges = canny(rescaled, sigma=2)

    r = (dish_diameter / 2.0) * scale
    hough_radii = np.arange(r - r*0.04, r + r*0.04, 2)
    hough_res = hough_circle(edges, hough_radii)
    _, cx, cy, rr = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)

    i = int(np.argmax(rr))
    cx_f, cy_f, r_f = cx[i] / scale, cy[i] / scale, rr[i] / scale

    # refine circle via your brightest-edge helper
    thetas = np.linspace(0, 2*np.pi, 9)[:-1]
    pts = np.array([[cx_f + r_f*np.cos(t), cy_f + r_f*np.sin(t)] for t in thetas])
    adj = find_brightest_with_dark_neighbor(img_gray_u8, pts, (cx_f, cy_f), threshold=50, search_distance=5)
    if len(adj) != len(pts):
        adj = find_brightest_with_dark_neighbor(img_gray_u8, pts, (cx_f, cy_f), threshold=50, search_distance=30)

    params0 = [cx_f, cy_f, r_f]
    new_params, _ = leastsq(circle_residuals, params0, args=(np.array(adj),), maxfev=2000)
    cx_a, cy_a, r_a = new_params
    return cx_a, cy_a, r_a
