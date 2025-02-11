import numpy as np
import inspect

# Callbacks that can be added to skimage.measure regionprops extra_properties parameter
#      At this stage, these metrics cannot be computed as they rely on other metrics.
#      It is up to the user to overwrite these fields afterwards.

def compactness(regionmask):
    return 0.0

def nn_centroid_distance(regionmask):    # nearest neighbor wrt. centroids
    return 0.0

def nn_collision_distance(regionmask):   # distance between edges of radius, taken from the equivalent diameter area property
    return 0.0

def cell_quality(region_mask):
    return 0.0

def axes_closness(region_mask):
    return 0.0

def custom1(regionmask):
    return 0.0

def custom2(regionmask):
    return 0.0

def custom3(regionmask):
    return 0.0

def custom4(regionmask):
    return 0.0

def custom5(regionmask):
    return 0.0

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


# This function is private in skimage
# Commit: cffc0e1d44061097e89ef0e2235ac780b43f7d7e
# https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops.py
# All values in this PROPS dict correspond to current scikit-image property
# names. The keys in this PROPS dict correspond to older names used in prior
# releases. For backwards compatibility, these older names will continue to
# work, but will not be documented.
PROPS = {
    'Area': 'area',
    'BoundingBox': 'bbox',
    'BoundingBoxArea': 'area_bbox',
    'bbox_area': 'area_bbox',
    'CentralMoments': 'moments_central',
    'Centroid': 'centroid',
    'ConvexArea': 'area_convex',
    'convex_area': 'area_convex',
    # 'ConvexHull',
    'ConvexImage': 'image_convex',
    'convex_image': 'image_convex',
    'Coordinates': 'coords',
    'Eccentricity': 'eccentricity',
    'EquivDiameter': 'equivalent_diameter_area',
    'equivalent_diameter': 'equivalent_diameter_area',
    'EulerNumber': 'euler_number',
    'Extent': 'extent',
    # 'Extrema',
    'FeretDiameter': 'feret_diameter_max',
    'FeretDiameterMax': 'feret_diameter_max',
    'FilledArea': 'area_filled',
    'filled_area': 'area_filled',
    'FilledImage': 'image_filled',
    'filled_image': 'image_filled',
    'HuMoments': 'moments_hu',
    'Image': 'image',
    'InertiaTensor': 'inertia_tensor',
    'InertiaTensorEigvals': 'inertia_tensor_eigvals',
    'IntensityImage': 'image_intensity',
    'intensity_image': 'image_intensity',
    'Label': 'label',
    'LocalCentroid': 'centroid_local',
    'local_centroid': 'centroid_local',
    'MajorAxisLength': 'axis_major_length',
    'major_axis_length': 'axis_major_length',
    'MaxIntensity': 'intensity_max',
    'max_intensity': 'intensity_max',
    'MeanIntensity': 'intensity_mean',
    'mean_intensity': 'intensity_mean',
    'MinIntensity': 'intensity_min',
    'min_intensity': 'intensity_min',
    'std_intensity': 'intensity_std',
    'MinorAxisLength': 'axis_minor_length',
    'minor_axis_length': 'axis_minor_length',
    'Moments': 'moments',
    'NormalizedMoments': 'moments_normalized',
    'Orientation': 'orientation',
    'Perimeter': 'perimeter',
    'CroftonPerimeter': 'perimeter_crofton',
    # 'PixelIdxList',
    # 'PixelList',
    'Slice': 'slice',
    'Solidity': 'solidity',
    # 'SubarrayIdx'
    'WeightedCentralMoments': 'moments_weighted_central',
    'weighted_moments_central': 'moments_weighted_central',
    'WeightedCentroid': 'centroid_weighted',
    'weighted_centroid': 'centroid_weighted',
    'WeightedHuMoments': 'moments_weighted_hu',
    'weighted_moments_hu': 'moments_weighted_hu',
    'WeightedLocalCentroid': 'centroid_weighted_local',
    'weighted_local_centroid': 'centroid_weighted_local',
    'WeightedMoments': 'moments_weighted',
    'weighted_moments': 'moments_weighted',
    'WeightedNormalizedMoments': 'moments_weighted_normalized',
    'weighted_moments_normalized': 'moments_weighted_normalized',
}

COL_DTYPES = {
    'area': float,
    'area_bbox': float,
    'area_convex': float,
    'area_filled': float,
    'axis_major_length': float,
    'axis_minor_length': float,
    'bbox': int,
    'centroid': float,
    'centroid_local': float,
    'centroid_weighted': float,
    'centroid_weighted_local': float,
    'coords': object,
    'coords_scaled': object,
    'eccentricity': float,
    'equivalent_diameter_area': float,
    'euler_number': int,
    'extent': float,
    'feret_diameter_max': float,
    'image': object,
    'image_convex': object,
    'image_filled': object,
    'image_intensity': object,
    'inertia_tensor': float,
    'inertia_tensor_eigvals': float,
    'intensity_max': float,
    'intensity_mean': float,
    'intensity_min': float,
    'intensity_std': float,
    'label': int,
    'moments': float,
    'moments_central': float,
    'moments_hu': float,
    'moments_normalized': float,
    'moments_weighted': float,
    'moments_weighted_central': float,
    'moments_weighted_hu': float,
    'moments_weighted_normalized': float,
    'num_pixels': int,
    'orientation': float,
    'perimeter': float,
    'perimeter_crofton': float,
    'slice': object,
    'solidity': float,
}

OBJECT_COLUMNS = [col for col, dtype in COL_DTYPES.items() if dtype == object]

def _infer_number_of_required_args(func):
    """Infer the number of required arguments for a function

    Parameters
    ----------
    func : callable
        The function that is being inspected.

    Returns
    -------
    n_args : int
        The number of required arguments of func.
    """
    argspec = inspect.getfullargspec(func)
    n_args = len(argspec.args)
    if argspec.defaults is not None:
        n_args -= len(argspec.defaults)
    return n_args


def _infer_regionprop_dtype(func, *, intensity, ndim):
    """Infer the dtype of a region property calculated by func.

    If a region property function always returns the same shape and type of
    output regardless of input size, then the dtype is the dtype of the
    returned array. Otherwise, the property has object dtype.

    Parameters
    ----------
    func : callable
        Function to be tested. The signature should be array[bool] -> Any if
        intensity is False, or *(array[bool], array[float]) -> Any otherwise.
    intensity : bool
        Whether the regionprop is calculated on an intensity image.
    ndim : int
        The number of dimensions for which to check func.

    Returns
    -------
    dtype : NumPy data type
        The data type of the returned property.
    """
    mask_1 = np.ones((1,) * ndim, dtype=bool)
    mask_1 = np.pad(mask_1, (0, 1), constant_values=False)
    mask_2 = np.ones((2,) * ndim, dtype=bool)
    mask_2 = np.pad(mask_2, (1, 0), constant_values=False)
    propmasks = [mask_1, mask_2]

    rng = np.random.default_rng()

    if intensity and _infer_number_of_required_args(func) == 2:

        def _func(mask):
            return func(mask, rng.random(mask.shape))

    else:
        _func = func
    props1, props2 = map(_func, propmasks)
    if (
        np.isscalar(props1)
        and np.isscalar(props2)
        or np.array(props1).shape == np.array(props2).shape
    ):
        dtype = np.array(props1).dtype.type
    else:
        dtype = np.object_
    return dtype

def regionprops_to_dict(regions, properties=('label', 'bbox'), separator='-'):
    """Convert image region properties list into a column dictionary.

    Parameters
    ----------
    regions : (K,) list
        List of RegionProperties objects as returned by :func:`regionprops`.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).

        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.

    Returns
    -------
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows.

    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.

    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.

    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.

    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key.

    Examples
    --------
    >>> from skimage import data, util, measure
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> proplist = regionprops(label_image, image)
    >>> props = _props_to_dict(proplist, properties=['label', 'inertia_tensor',
    ...                                              'inertia_tensor_eigvals'])
    >>> props  # doctest: +ELLIPSIS +SKIP
    {'label': array([ 1,  2, ...]), ...
     'inertia_tensor-0-0': array([  4.012...e+03,   8.51..., ...]), ...
     ...,
     'inertia_tensor_eigvals-1': array([  2.67...e+02,   2.83..., ...])}

    The resulting dictionary can be directly passed to pandas, if installed, to
    obtain a clean DataFrame:

    >>> import pandas as pd  # doctest: +SKIP
    >>> data = pd.DataFrame(props)  # doctest: +SKIP
    >>> data.head()  # doctest: +SKIP
       label  inertia_tensor-0-0  ...  inertia_tensor_eigvals-1
    0      1         4012.909888  ...                267.065503
    1      2            8.514739  ...                  2.834806
    2      3            0.666667  ...                  0.000000
    3      4            0.000000  ...                  0.000000
    4      5            0.222222  ...                  0.111111

    """

    out = {}
    n = len(regions)
    for prop in properties:
        r = regions[0]
        # Copy the original property name so the output will have the
        # user-provided property name in the case of deprecated names.
        orig_prop = prop
        # determine the current property name for any deprecated property.
        prop = PROPS.get(prop, prop)
        rp = getattr(r, prop)
        if prop in COL_DTYPES:
            dtype = COL_DTYPES[prop]
        else:
            func = r._extra_properties[prop]
            dtype = _infer_regionprop_dtype(
                func,
                intensity=r._intensity_image is not None,
                ndim=r.image.ndim,
            )

        # scalars and objects are dedicated one column per prop
        # array properties are raveled into multiple columns
        # for more info, refer to notes 1
        if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
            column_buffer = np.empty(n, dtype=dtype)
            for i in range(n):
                column_buffer[i] = regions[i][prop]
            out[orig_prop] = np.copy(column_buffer)
        else:
            # precompute property column names and locations
            modified_props = []
            locs = []
            for ind in np.ndindex(np.shape(rp)):
                modified_props.append(separator.join(map(str, (orig_prop,) + ind)))
                locs.append(ind if len(ind) > 1 else ind[0])

            # fill temporary column data_array
            n_columns = len(locs)
            column_data = np.empty((n, n_columns), dtype=dtype)
            for k in range(n):
                # we coerce to a numpy array to ensure structures like
                # tuple-of-arrays expand correctly into columns
                rp = np.asarray(regions[k][prop])
                for i, loc in enumerate(locs):
                    column_data[k, i] = rp[loc]

            # add the columns to the output dictionary
            for i, modified_prop in enumerate(modified_props):
                out[modified_prop] = column_data[:, i]
    return out
