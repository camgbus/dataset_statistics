import numpy as np

from skimage.measure import label,regionprops

def skimg_props(y, x=None, props_y=['area', 'area_convex', 'bbox', 'centroid'], 
    props_x_y=['intensity_mean']):
    """Extract properties from a label map and optional intensity image. For 
    other region properties, see: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

    Parameters:
    y (numpy.ndarray): label map
    x (numpy.ndarray): img with same dimensions as y but continuous values
    props_y (lst(str)): properties calculated from y
    props_x_y (lst(str)): properties calculated from y and x (if x not None)

    Returns:
    dict(str -> Any): property dictionary
    """
    props = regionprops(label_image=y, intensity_image=x)
    assert len(props) == 1 # Only one label
    props_dict = {key: props[0][key] for key in props_y}
    if x is not None and props_x_y:
        for key in props_x_y:
            props_dict[key] = props[0][key]
    return props_dict

def connected_components(y, connectivity=2):
    """Calculate the connected components of a label map.

    Parameters:
    y (numpy.ndarray): label map
    connectivity (int): maximum number of orthogonal hops to consider a 
        pixel/voxel a neighbor (the smaller the 'connectivity' attribute, the 
        more connected components).

    Returns:
    (numpy.ndarray, int): an array where each component is assigned a new label,
        ant the number of connected components
    """
    labeled_image, nr_components = label(y, return_num=True, connectivity=1)
    return labeled_image, nr_components

def relative_area(array_a, val_a, vals_b='all', array_b=None):
    """Returns the area of val_a in array_a relative to the summed area of 
    vals_b in array_b.
    """
    if array_b is None:
        array_b = array_a
    
    present_values_a = list(np.unique(array_a))
    assert val_a in present_values_a, "{} not in array a".format(v)
    present_values_b = list(np.unique(array_b))
    if vals_b == 'all':
        vals_b = present_values_b
    else:
        for v in vals_b:
            assert v in present_values_b, "{} not in array b".format(v)
    # Area of val_a in array_a
    area_val_a = (array_a == val_a).sum()
    # Summed area of vals_b in array_b
    area_vals_b = sum((array_b == v).sum() for v in vals_b)
    return area_val_a/area_vals_b

def relative_bounding_boxes(resolution, bb):
    assert len(resolution)==3 and len(bb)==6
    rel_values = [bb[0]/resolution[0], bb[1]/resolution[1], bb[2]/resolution[2],
        bb[3]/resolution[0], bb[4]/resolution[1], bb[5]/resolution[2]]
    return tuple(round(v, 3) for v in rel_values)