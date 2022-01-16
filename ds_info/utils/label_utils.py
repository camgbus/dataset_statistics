import numpy as np
from collections import OrderedDict

def map_label_voxels(y, mappings=[(2,0)]):
    """Change the value of certain labels, i.e. for each item of mappings, turn
        the values equal to key to value. Keep in mind that the order of the 
        mappings is relevant.

    Parameters:
    y (numpy.ndarray): label map
    mappings (lst(int, int)): mappings as (from, to)

    Returns:
    numpy.ndarray: a label map with a new class label mapping
    """
    for key, value in mappings:
        y = np.where(y==key, value, y) 
    return y

def divide_label_maps(y):
    """Turn a map with multiple labels into a list of maps with each label 1.

    Parameters:
    y (numpy.ndarray): label map

    Returns:
    OrderedDict(key -> numpy.ndarray): a dictionary mapping labels to label maps,
        ordered wrt the label value
    """
    labels = sorted(list(np.unique(y)))
    labels.pop(0)
    divided_label_maps = OrderedDict()
    for label in labels:
        other_labels = [l for l in labels if l != label]
        new_y = map_label_voxels(y, mappings=[(key, 0) for key in other_labels]+[(label, 1)])
        divided_label_maps[label] = new_y
    return divided_label_maps

