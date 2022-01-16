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