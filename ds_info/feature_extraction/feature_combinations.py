

from ds_info.utils.io_utils import get_img_label, get_arrays_from_img_label
from ds_info.feature_extraction.img_features import resolution, voxel_spacing, intensity_mean_median
from ds_info.feature_extraction.label_features import skimg_props, connected_components
from ds_info.utils.label_utils import divide_label_maps

def some_features(dataset_path, file_name, img_mode=0):
    x_sitk, y_sitk = get_img_label(dataset_path, file_name)
    x, y = get_arrays_from_img_label(x_sitk, y_sitk, img_mode=img_mode)

    props = dict()
    mean, median = intensity_mean_median(x)
    props['intensity_mean'], props['intensity_median'] = intensity_mean_median(x)
    props['resolution'] = resolution(x_sitk)
    props['spacing'] = voxel_spacing(x_sitk)
    # Properties for each label
    '''
    label_maps = divide_label_maps(y)
    for label, label_map in label_maps.items():
        label_props = skimg_props(label_map, x)
        for key, value in label_props.items():
            props[str(label)+'_'+key] = value
        # Properties for the largest connected component on each label
        labeled_image, nr_components = connected_components(label_map)
        props[str(label)+'_CC'] = nr_components
        cc_maps = divide_label_maps(label_map).values()
        largest_map = max(cc_maps, key=lambda m: skimg_props(m)['area'])

        largest_cc_props = skimg_props(largest_map, x)
        for key, value in largest_cc_props.items():
            props[str(label)+'_CC_'+key] = value
    '''
    return props