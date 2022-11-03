import os
import SimpleITK as sitk

def list_files(dataset_path):
    """List files in a dataset directory with the format of the Medical
    Segmentation Decathlon.

    Parameters:
    dataset_path (str): path to a dataset

    Returns:
    lst(str): full file names
    """
    train_img_path = os.path.join(dataset_path, 'imagesTr')
    file_names = [f for f in os.listdir(train_img_path) if 
        os.path.isfile(os.path.join(train_img_path, f))]
    return file_names

def get_img_label(dataset_path, file_name):
    """Load an image or label map.

    Parameters:
    dataset_path (str): path to a dataset with the Medical Segmentation 
        Decathlon structure
    file_name (str): name of the file, including ending

    Returns:
    (SimpleITK.SimpleITK.Image, SimpleITK.SimpleITK.Image): image and label map
    """
    img_path = os.path.join(dataset_path, 'imagesTr', file_name)
    # Remove the mode, e.g. _0000, _0001
    for possible_mode in range(5):
        file_name = file_name.replace('_000{}.nii'.format(possible_mode), '.nii')
    label_path = os.path.join(dataset_path, 'labelsTr', file_name)
    return sitk.ReadImage(img_path), sitk.ReadImage(label_path)

def get_arrays_from_img_label(img, label, img_mode=None):
    """Transform a SimpleITK image and label map into numpy arrays, and 
        optionally select a channel.

    Parameters:
    img (SimpleITK.SimpleITK.Image): image
    label (SimpleITK.SimpleITK.Image): label map
    img_mode (int or None): optional mode channel, so output is 3D

    Returns:
    (numpy.ndarray, numpy.ndarray): image and label in numpy format
    """
    img_np = sitk.GetArrayFromImage(img)
    if img_mode is not None:
        img_np = img_np[img_mode]
    label_np = sitk.GetArrayFromImage(label)
    return img_np, label_np.astype(int)

# PICKLE
import pickle
def pkl_dump(obj, name, path='obj'):
    """Saves an object in pickle format."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    pickle.dump(obj, open(path, 'wb'))

def pkl_load(name, path='obj'):
    """Restores an object from a pickle file."""
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    try:
        obj = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        obj = None
    return obj

# NUMPY
from numpy import save, load

def np_dump(obj, name, path='obj'):
    """Saves an object in npy format."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    save(path, obj)

def np_load(name, path='obj'):
    """Restores an object from a npy file."""
    if '.npy' not in name:
        name = name + '.npy'
    path = os.path.join(path, name)
    try:
        obj = load(path)
    except FileNotFoundError:
        obj = None
    return obj

# JSON
import json
def save_json(dict_obj, path, name):
    """Saves a dictionary in json format."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(path, name):
    """Restores a dictionary from a json file."""
    if '.json' not in name:
        name += '.json'
    with open(os.path.join(path, name), 'r') as json_file:
        return json.load(json_file)
