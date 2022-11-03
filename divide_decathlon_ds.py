import os
import numpy as np
import torch
import monai
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
import seaborn as sns 
sns.set_style("whitegrid")
import pandas as pd
from ds_info.utils.io_utils import pkl_dump, pkl_load, list_files
from ds_info.utils.io_utils import get_img_label, get_arrays_from_img_label
from tqdm import tqdm

def min_dims_dataset(task_name):
    min_x, min_y, min_z = 1e5, 1e5, 1e5
    dataset_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task_name)
    file_names = list_files(dataset_path)
    ending = '_000{}.nii.gz'.format('0')
    file_names = [file_name for file_name in file_names if ending in file_name]
    for file_name in tqdm(file_names):
        x_sitk, y_sitk = get_img_label(dataset_path, file_name)
        x, y = get_arrays_from_img_label(x_sitk, y_sitk, img_mode=None) # Arg. is for slicing, do not use here
        shape = np.array(x).shape
        min_x, min_y, min_z = min(min_x, shape[0]), min(min_y, shape[1]), min(min_z, shape[2])
    return min_x, min_y, min_z

def center_roi_all_subjects(task_name, roi_size, img_mode=0):
    dataset_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', task_name)
    X = []
    file_names = list_files(dataset_path)
    ending = '_000{}.nii.gz'.format(img_mode)
    file_names = [file_name for file_name in file_names if ending in file_name]
    ids = [file_name.replace(ending, '') for file_name in file_names]
    for file_name in tqdm(file_names):
        x_sitk, y_sitk = get_img_label(dataset_path, file_name)
        x, y = get_arrays_from_img_label(x_sitk, y_sitk, img_mode=None) # Arg. is for slicing, do not use here
        x = np.array(x)
        x = torch.tensor(x)
        x = torch.flatten(center_crop_3d(x, roi_size=roi_size))
        X.append(x.detach().numpy())
    return np.array(X), ids

def center_crop_3d(x, roi_size):
    """Center-crops a tensor so it has the specified dimension."""
    cropper = monai.transforms.CenterSpatialCrop(roi_size)
    if len(x.shape) < 4:
        x = x[None, :]
        x = cropper(x)
        return torch.tensor(x[0])
    return torch.tensor(cropper(x))

def pca(X, nr_components=1):
    pca = PCA(n_components=nr_components)
    pca.fit(X)
    print("{} samples with {} features each".format(pca.n_samples_, pca.n_features_))
    print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))
    X = pca.transform(X)
    return X, pca.explained_variance_ratio_

def plot_clusters(X, labels, file_path, file_name):
    df = pd.DataFrame(X, columns=["x", "y"])
    color_palette = {0: "#4059AD", 1: "#F48942", 2: "#97D8C4", 3: "#F4B942", 4: "#EFF2F1", 5: "#6B9AC4"}
    df["cluster"] = labels
    plot = sns.scatterplot(data=df, x="x", y="y", hue=df["cluster"], palette=color_palette)
    plot.figure.savefig(os.path.join(file_path, "{}.png".format(file_name)))

def save_clustering(root_path, task_name, nr_components, nr_clusters, roi_size):
    clustering_name = "{}_{}_{}_{}".format(task_name, nr_components, nr_clusters, "-".join([str(x) for x in roi_size]))
    X, ids = center_roi_all_subjects(task_name, roi_size)
    print('Data is prepared')
    X, explained_variance_ratio = pca(X, nr_components)
    print('PCA done')
    centers, labels, sum_sq_dist = k_means(X, nr_clusters)
    print('K-means done')
    clustering = {'X': X, 'centers': centers, 'labels': labels, 'sum_sq_dist': sum_sq_dist, 'ids': ids, 'explained_variance_ratio': explained_variance_ratio}
    pkl_dump(clustering, clustering_name, path=root_path)
    return clustering_name

def plot_clustering(root_path, clustering=None, clustering_name=None):
    if clustering is None:
        clustering = pkl_load(name=clustering_name, path=root_path)
    plot_clusters(clustering['X'], clustering['labels'], file_path=root_path, file_name=clustering_name)

# Obtained with min_dims_dataset
roi_sizes = {'Task001_BrainTumour': (155, 240, 240), 'Task006_Lung': (112, 512, 512), 'Task007_Pancreas': (37, 512, 512), 'Task008_HepaticVessel': (24, 512, 512), 'Task010_Colon': (37, 512, 512)}

root_path = os.path.join(os.environ['IPMI23'], 'clusters')
task_name = 'Task001_BrainTumour'
nr_components = 2
nr_clusters = 5
roi_size = roi_sizes[task_name]

clustering_name = save_clustering(root_path, task_name, nr_components, nr_clusters, roi_size)
plot_clustering(root_path, clustering_name=clustering_name)
