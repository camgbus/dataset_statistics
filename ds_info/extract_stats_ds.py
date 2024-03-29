#   Extract statistics of a dataset with the MSD format.

import os, argparse
from ds_info.utils.io_utils import list_files
from ds_info.utils.format_utils import zeros_in_format
from ds_info.feature_extraction.feature_combinations import reduced_features, props_mean_std
from ds_info.visualization.plots import props_to_pandas

import time
from tqdm import tqdm

def extract_ds_stats(dataset_path, img_mode = None, case_names=None):
    # Extract properties for each subject
    if case_names is None:
        # All cases
        file_names = list_files(dataset_path)
    else:
        file_names = ["{}.nii.gz".format(n) for n in case_names]
    props = dict()
    all_props_rep = dict()
    for file_name in tqdm(file_names):
        subject_name = file_name.split('.')[0]
        props[subject_name] = reduced_features(dataset_path, file_name, img_mode=img_mode)
        for key, value in props[subject_name].items():
            all_props_rep[key] = value
    # Check that all prop. are extracted for all subjects, and fill. This is 
    # needed for cases where all labels are not represented in all cases
    for subject, subject_props in props.items():
        for key, value in  all_props_rep.items():
            if key not in subject_props:
                subject_props[key] = zeros_in_format(value)
    return props

def save_ds_stats(dataset_path, save_df_path, ds_name, img_mode=None, case_names=None):
    props = extract_ds_stats(dataset_path=dataset_path, img_mode=img_mode, case_names=case_names)
    means, stds = props_mean_std(props)
    props['MEAN'] = means
    props['STD'] = stds
    df = props_to_pandas(props, columns=['intensity_mean', 'resolution', 'spacing', '1_area-rel', '1_CC'])
    csv_name = "{}_stats.csv".format(ds_name)
    df.to_csv(os.path.join(save_df_path, csv_name))    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("save_path")
    args = parser.parse_args()
    dataset_path = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', args.task)
    save_df_path = args.save_path
    start = time.time()
    save_ds_stats(dataset_path, save_df_path, ds_name=args.task, img_mode=None, case_names=None)
    time_passed = time.time() - start
    print('Finished {} time passed: {:.2f} s.'.format(args.task, time_passed))
import numpy as np
if __name__ == "__main__":
    main()
