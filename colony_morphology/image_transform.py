import cv2 as cv
import numpy as np

from scipy import ndimage
from skimage import morphology

def resize_image(image, pixel_threshold=1280*1280):
    """
    Args:
      image:           image to resize
      pixel_threshold: max number of pixels before resizing

    Returns:
      scaling_factor
      img_resize
    """
    img_resize = image

    scale = 1.0
    max_x = image.shape[1]
    max_y = image.shape[0]

    if(image.size > pixel_threshold):
        pixels = max_x * max_y
        scale = 1.0 / (pixels / pixel_threshold);
        point = (int(scale * max_x), int(scale * max_y))

        img_resize = cv.resize(image, point, interpolation= cv.INTER_LINEAR)

    return (scale, img_resize)


def crop_image_to_circle(image, centroid, diameter, shrinkage_ratio=0.95):

    x, y, _ = image.shape
    center = (centroid[1], centroid[0])

    threshold = (diameter*shrinkage_ratio)**2

    # initialize mask
    mask = np.zeros_like(image)

    # crop as circle.
    for x_ in range(x):
        for y_ in range(y):
            dist = (x_ - center[0])**2 + (y_ - center[1])**2
            mask[x_, y_] = (dist < threshold)

    return image*mask



def background_subtraction(image, sigma=1):

    image_ = ndimage.gaussian_filter(image, sigma)
    #seed = np.copy(image_)
    #seed[1:-1, 1:-1] = image_.min()
    #image_ = image.copy()
    seed = image_ - 0.4
    mask = image_
    dilated = morphology.reconstruction(seed, mask, method='dilation')
    return image_ - dilated
