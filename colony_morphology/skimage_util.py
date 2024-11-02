import numpy as np


# Callbacks that can be added to skimage.measure regionprops extra_properties parameter
#      At this stage, these metrics cannot be computed as they rely on other metrics.
#      It is up to the user to overwrite these fields afterwards.

def compactness(regionmask):
    return 0

def min_distance_nn(regionmask):
    return 0

def cell_quality(region_mask):
    return 0

def custom1(regionmask):
    return 0

def custom2(regionmask):
    return 0

def custom3(regionmask):
    return 0

def custom4(regionmask):
    return 0

def custom5(regionmask):
    return 0

# Excluding Intensity metrics
RegionPropertyNames =(
        'area',  
        'area_bbox',  
        'area_convex',  
        'area_filled',  
        'axis_major_length',  
        'axis_minor_length',  
        'bbox',  
        'centroid',  
        'centroid_local',  
        'coords',  
        'eccentricity',  
        'equivalent_diameter_area',  
        'euler_number',  
        'extent',  
        'feret_diameter_max',  
        'image',  
        'image_convex',  
        'image_filled',
        'inertia_tensor',  
        'inertia_tensor_eigvals',   
        'label',  
        'moments',  
        'moments_central',  
        'moments_hu',  
        'moments_normalized',  
        'orientation',  
        'perimeter',  
        'perimeter_crofton',  
        'slice',  
        'solidity',)


