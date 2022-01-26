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