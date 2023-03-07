#   Create a new dataset in MSD format.

import os
import shutil
import SimpleITK as sitk
from ds_info.utils.io_utils import pkl_load
from nnunet.dataset_conversion.utils import generate_dataset_json
from ds_info.ds_division.define_new_splits import create_new_splits, split_to_new_cases, get_cases_from_splits

def create_ds_from_nnunet_dss(new_task_full_name, new_cases, 
    label_names, mode_names):
    """For several existing nnunet datasets, determine a subset of case names. 
    All these files are stored as a new 'task'. The names remain the same but 
    are prefaced by the old task name

    param new_cases: dict(<full task name> -> list(<case names>))
    """

    # Create new directories
    store_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')
    dataset_path = os.path.join(store_path, new_task_full_name)
    new_img_path = os.path.join(dataset_path, 'imagesTr')
    test_img_path = os.path.join(dataset_path, 'imagesTs')
    new_labels_path = os.path.join(dataset_path, 'labelsTr')
    os.mkdir(dataset_path)
    os.mkdir(new_img_path)
    os.mkdir(test_img_path)
    os.mkdir(new_labels_path)

    # Copy files from relevant cases
    for old_task, case_names in new_cases.items():
        old_task_path = os.path.join(store_path, old_task)
        old_img_path = os.path.join(old_task_path, 'imagesTr')
        old_labels_path = os.path.join(old_task_path, 'labelsTr')
        for file_name in [f for f in os.listdir(old_labels_path) if 
            os.path.isfile(os.path.join(old_labels_path, f)) and 
            any(case_name in f for case_name in case_names)]:
            shutil.copyfile(os.path.join(old_labels_path, file_name),
                os.path.join(new_labels_path, '['+old_task+']'+file_name))
        for file_name in [f for f in os.listdir(old_img_path) if 
            os.path.isfile(os.path.join(old_img_path, f)) and 
            any(case_name in f for case_name in case_names)]:
            shutil.copyfile(os.path.join(old_img_path, file_name),
                os.path.join(new_img_path, '['+old_task+']'+file_name))

    # Generate json
    generate_dataset_json(output_file=os.path.join(dataset_path, 'dataset.json'),
        imagesTr_dir=new_img_path, imagesTs_dir=test_img_path, modalities=mode_names, 
        labels=label_names, dataset_name=new_task_full_name)

def case_names_from_cluster(old_task, cluster_ix, cluster_name, cluster_path):
    clustering = pkl_load(name=cluster_name, path=cluster_path)
    ids = [id for label, id in zip(clustering['labels'], clustering['ids']) if label==cluster_ix]
    return ids

def generate_joint_tasks(new_task_name, old_ordered_tasks, 
    label_names, mode_names, nr_folds=5):
    """From a list of task names, generate a new task joining that data, 
    maintaining previous splits.
    """
    # Generate splits_final.pkl and store
    new_task_split_path = os.path.join(os.environ['nnUNet_preprocessed'], new_task_name)
    old_task_split_paths = [os.path.join(os.environ['nnUNet_preprocessed'], task) for task in old_ordered_tasks]
    new_task_ratios = [[1] for old_task in old_ordered_tasks]
    create_new_splits([new_task_name], [new_task_split_path], old_ordered_tasks, 
    old_task_split_paths, new_task_ratios, nr_folds=nr_folds)
    # From splits, change to new_cases format and generate new data
    store_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')
    new_cases = split_to_new_cases(get_cases_from_splits(new_task_split_path))
    create_ds_from_nnunet_dss(new_task_name, new_cases, label_names, mode_names)


# Create new (splitted) tasks
old_to_new_tasks = {'Task001_BrainTumour': ['Task011_BrainTumourC1', 'Task012_BrainTumourC2', 'Task013_BrainTumourC3', 'Task014_BrainTumourC4', 'Task015_BrainTumourC5'],
'Task006_Lung': ['Task061_LungC1', 'Task062_LungC2', 'Task063_LungC3', 'Task064_LungC4', 'Task065_LungC5'],
'Task007_Pancreas': ['Task071_PancreasC1', 'Task072_PancreasC2', 'Task073_PancreasC3', 'Task074_PancreasC4', 'Task075_PancreasC5'],
'Task008_HepaticVessel': ['Task081_HepaticVesselC1', 'Task082_HepaticVesselC2', 'Task083_HepaticVesselC3', 'Task084_HepaticVesselC4', 'Task085_HepaticVesselC5'],
'Task010_Colon': ['Task091_ColonC1', 'Task092_ColonC2', 'Task093_ColonC3', 'Task094_ColonC4', 'Task095_ColonC5']}
task_labels = {'Task001_BrainTumour': {0: "background", 1: "edema", 2: "non-enhancing tumor", 3: "enhancing tumour"},
    'Task006_Lung': {0: "background", 1: "cancer"},
    'Task007_Pancreas': {0: "background", 1: "pancreas", 2: "cancer"},
    'Task008_HepaticVessel': {0: "background", 1: "Vessel", 2: "Tumour"},
    'Task010_Colon': {0: "background", 1: "colon cancer primaries"}}
    
task_modes = {'Task001_BrainTumour': ('FLAIR', 'T1w', 't1gd', 'T2w'), 'Task006_Lung': ('CT',), 'Task007_Pancreas': ('CT',), 'Task008_HepaticVessel': ('CT',), 'Task010_Colon': ('CT',)}

clusters = {'Task001_BrainTumour': 'Task001_BrainTumour_2_5_155-240-240', 'Task006_Lung': 'Task006_Lung_2_5_112-512-512', 
    'Task007_Pancreas': 'Task007_Pancreas_2_5_37-512-512', 'Task008_HepaticVessel': 'Task008_HepaticVessel_2_5_24-512-512', 
    'Task010_Colon': 'Task010_Colon_2_5_37-512-512'}

clusters_path = os.path.join(os.environ['IPMI23'], 'clusters')

"""
for old_task, new_tasks in old_to_new_tasks.items():
    for cluster_ix, new_task_full_name in enumerate(new_tasks):
        print('New task: {}'.format(new_task_full_name))
        new_cases = {old_task: case_names_from_cluster(old_task, cluster_ix, clusters[old_task], clusters_path)}
        create_ds_from_nnunet_dss(new_task_full_name, new_cases, task_labels[old_task], task_modes[old_task])
"""

# Join clusters
individual_clusters = {'Task001_BrainTumour': 'Task013_BrainTumourC3', 'Task006_Lung': 'Task062_LungC2', 'Task007_Pancreas': 'Task073_PancreasC3', 'Task008_HepaticVessel': 'Task085_HepaticVesselC5', 'Task010_Colon': 'Task093_ColonC3'}
joint_clusters = {old_task: [t for t in old_to_new_tasks[old_task] if t != leave_out_task] for old_task, leave_out_task in individual_clusters.items()}
new_task_names = {'Task001_BrainTumour': 'Task913_BrainTumourNC3', 'Task006_Lung': 'Task962_LungNC2', 'Task007_Pancreas': 'Task973_PancreasNC3', 'Task008_HepaticVessel': 'Task985_HepaticVesselNC5', 'Task010_Colon': 'Task993_ColonNC3'}

for old_task, new_task_name in new_task_names.items():
    if old_task != 'Task001_BrainTumour':
        old_ordered_tasks = joint_clusters[old_task]
        generate_joint_tasks(new_task_name, old_ordered_tasks, task_labels[old_task], 
            task_modes[old_task], nr_folds=5)
