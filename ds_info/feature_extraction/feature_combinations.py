

from pandas import array
from ds_info.utils.io_utils import get_img_label, get_arrays_from_img_label
from ds_info.feature_extraction.img_features import resolution, voxel_spacing, intensity_mean_median
from ds_info.feature_extraction.label_features import skimg_props, connected_components, relative_area, relative_bounding_boxes
from ds_info.utils.label_utils import divide_label_maps
import numpy as np
import warnings
from collections import OrderedDict
import numbers

def some_features(dataset_path, file_name, img_mode=0):
    # Fetch data
    x_sitk, y_sitk = get_img_label(dataset_path, file_name)
    x, y = get_arrays_from_img_label(x_sitk, y_sitk, img_mode=img_mode)
    props = OrderedDict()
    # Image properties
    props['intensity_mean'], props['intensity_median'] = intensity_mean_median(x)
    props['resolution'] = resolution(x)
    props['spacing'] = voxel_spacing(x_sitk)
    # Properties for each label
    label_maps = divide_label_maps(y)
    for label, label_map in label_maps.items():
        label_props = skimg_props(label_map, x)
        for key, value in label_props.items():
            props[str(label)+'_'+key] = value
        # Add relative area
        props[str(label)+'_area-rel'] = relative_area(label_map, 1)
        props[str(label)+'_bbox-rel'] = relative_bounding_boxes(props['resolution'], props[str(label)+'_bbox'])
        # Properties for the largest connected component of each label
        labeled_image, nr_components = connected_components(label_map, connectivity=2)
        props[str(label)+'_CC'] = nr_components
        cc_maps = divide_label_maps(labeled_image).values()
        # Sometimes, convex hull cannot be calculated for small components
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            largest_map = max(cc_maps, key=lambda m: skimg_props(m)['area'])
        largest_cc_props = skimg_props(largest_map, x)
        for key, value in largest_cc_props.items():
            props[str(label)+'_CC_'+key] = value
        # Add area relative to whole label
        props[str(label)+'_CC_area-rel-{}'.format(label)] = relative_area(largest_map, 1, vals_b=[1], array_b=label_map)
    return props

def reduced_features(dataset_path, file_name, img_mode=0):
    # Fetch data
    x_sitk, y_sitk = get_img_label(dataset_path, file_name)
    x, y = get_arrays_from_img_label(x_sitk, y_sitk, img_mode=img_mode)
    props = OrderedDict()
    # Image properties
    props['intensity_mean'], _ = intensity_mean_median(x)
    props['resolution'] = resolution(x)
    props['spacing'] = voxel_spacing(x_sitk)
    # Properties for each label
    label_maps = divide_label_maps(y)
    for label, label_map in label_maps.items():
        # Add relative area
        props[str(label)+'_area-rel'] = relative_area(label_map, 1)
        # Properties for the largest connected component of each label
        labeled_image, nr_components = connected_components(label_map, connectivity=2)
        props[str(label)+'_CC'] = nr_components
    return props

def props_mean_std(props, round=2):
    ordered_vals = OrderedDict()
    for subject_props in props.values():
        for key, value in subject_props.items():
            if key not in ordered_vals:
                ordered_vals[key] = []
            ordered_vals[key].append(value)
    means, stds = OrderedDict(), OrderedDict()
    for key, val in ordered_vals.items():
        means[key], stds[key] = np.mean(val, axis=0), np.std(val, axis=0)
        if round is not None:
            means[key], stds[key] = np.round_(means[key], round), np.round_(stds[key], round)
    return means, stds

def props_range(props):
    ordered_vals = OrderedDict()
    for subject_props in props.values():
        for key, value in subject_props.items():
            if key not in ordered_vals:
                ordered_vals[key] = []
            ordered_vals[key].append(value)
    mins, maxs = OrderedDict(), OrderedDict()
    for key, val in ordered_vals.items():
        mins[key], maxs[key] = np.min(val, axis=0), np.max(val, axis=0)
    return mins, maxs

def in_range(x, min, max):
    if isinstance(min, numbers.Number):
        assert isinstance(x, numbers.Number)
        in_range = x >= min and x <= max 
    else:
        x = list(x)
        assert len(x) == len(min) == len(max)
        in_range = all(x[i]>= min[i] and x[i]<= max[i] for i in range(len(x)))
    return in_range

def select_subjects(props, prop_key, prop_min, prop_max):
    return [s for s in props.keys() if 
        in_range(props[s][prop_key], prop_min, prop_max)]

def select_k_nearest_subjects(props, prop_key, value, k=None, ix=None):
    """Returns an ordered list of the specified length, order according to the
    closeness to value.
    """
    if isinstance(value, numbers.Number):
        sorted_lst = sorted(list(props.keys()), key=
            lambda k: abs(props[k][prop_key] - value))
    else:
        assert ix is not None
        sorted_lst = sorted(list(props.keys()), key=
            lambda k: abs(props[k][prop_key][ix] - value[ix]))
    if k is None:
        return sorted_lst
    else:
        return sorted_lst[:k]