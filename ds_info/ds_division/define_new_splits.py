# Define splits_final for combined datasets based on other existing splits from the individual sets.

import os
import math
import numpy as np
import pickle
import random
from collections import OrderedDict

def save_splits_from_cases(save_path, new_splits):
    for fold_ix in range(len(new_splits)):
        """Converts list of cases into numpy array, as the original format, 
        and saves.
        """
        new_splits[fold_ix] = OrderedDict([('train', np.array(new_splits[fold_ix]['train'])), 
            ('val', np.array(new_splits[fold_ix]['val']))])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'splits_final.pkl'), 'wb') as f:
        pickle.dump(new_splits, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_cases_from_splits(splits_path):
    """Loads the contents of splits_final.pkl from a directory. These have the
    form of a list with the number of folds (typically 5). In each list there is
    an OrderedDict with the keys 'train' and 'val'. The values are numpy arrays 
    the name of cases (without endings) as numpy strings.

    param splits_path (str): a path without including the file name.
    """
    with open(os.path.join(splits_path, 'splits_final.pkl'), 'rb') as f:
        per_fold_split_cases = pickle.load(f)
        return per_fold_split_cases

def distribute_total(ratios, total_nr):
    """Receiving a tuple of ratios, e.g. (0.8, 0.2), distribute the total so 
    there are no remaining, overweighting smaller ratios but not assigning
    any cases if the ratio is = 0.

    param ratios (tuple): A tuple of ratios that must add to 1.
    param total_nr (int): total number of cases
    returns (list(int)): Number of cases, with the same lenght as 'ratios'.
    """
    assert abs(sum(ratios) - 1) < 0.001
    assert total_nr > 0
    dist_total = [math.ceil(ratio*total_nr) for ratio in ratios]
    for ratio in sorted(ratios, reverse=True):
        if sum(dist_total) > total_nr:
            dist_total[ratios.index(ratio)] -= 1
    return dist_total

def create_new_splits(new_task_names, new_task_split_paths, old_ordered_tasks, 
    old_task_split_paths, new_task_ratios, nr_folds=5):
    """Creates new split files by distributing old cases into new tasks. Saves
    these new splits in the specified directories. Note that in the splits, 
    the case name is preceded by [old task name] to show where the case came from.

    param new_task_names (list(str)): a list with new task names
    param new_task_split_paths (list(str)): a list of directories where these 
        split files will be stored. If these don't exist, they are created
    param old_ordered_tasks (list(str)): a list with old task names
    param old_task_split_paths (list(str)): a list of paths, one per old task, 
        where splits_final.pkl files are stored
    param new_task_ratios: for each old task, what raio of cases will be part of 
        each of the new tasks?
    """
    assert len(old_ordered_tasks) == len(old_task_split_paths) == len(new_task_ratios)
    assert len(new_task_names) == len(new_task_split_paths) == len(new_task_ratios[0])

    new_task_splits = {new_task: [OrderedDict([('train', []), ('val', [])]) 
        for _ in range(nr_folds)] for new_task in new_task_names}

    for task_ix, task in enumerate(old_ordered_tasks):
        per_fold_split_cases = get_cases_from_splits(old_task_split_paths[task_ix])
        # Select validation cases per fold. The train cases for that fold are 
        # the val cases for all other folds. The total cases for that new task 
        # are the val cases for all folds.
        assert nr_folds == len(per_fold_split_cases)
        for fold_ix, fold_cases in enumerate(per_fold_split_cases):
            all_val_cases = list(fold_cases['val'])
            # How many of the old val cases in this task fold?
            nr_val_new_task = distribute_total(new_task_ratios[task_ix], len(all_val_cases))
            # For each new task, select a subset of cases and remove from 
            # the array where remaining cases are stored. These are distributed
            # among all new tasks.
            for new_task_ix, new_cases_val in enumerate(nr_val_new_task):
                new_val_cases = random.sample(all_val_cases, new_cases_val)
                all_val_cases = [x for x in all_val_cases if x not in new_val_cases]
                # Add new task precedence
                new_val_cases = ['['+task+']'+case for case in new_val_cases]
                new_task_splits[new_task_names[new_task_ix]][fold_ix]['val'] += new_val_cases
        
    # Now set the training cases for each fold
    for new_task in new_task_names:
        for fold_ix in range(nr_folds):
            for fold_ix_val in range(nr_folds):
                if fold_ix != fold_ix_val:
                    new_task_splits[new_task][fold_ix]['train'] += new_task_splits[new_task][fold_ix_val]['val']

    # Assert split integrity
    for fold_ix in range(nr_folds):
        # For the same split, different tasks should not share training samples
        all_train_samples = []
        for new_task in new_task_names:
            new_train_samples = new_task_splits[new_task][fold_ix]['train']
            assert not any(x in all_train_samples for x in new_train_samples)
            all_train_samples += new_train_samples
            # For no fold should a sample be in the train and val split
            assert not any(x in new_train_samples for x in new_task_splits[new_task][fold_ix]['val'])

    # Create splits_final and save for each task
    for new_task_name, new_task_split_path in zip(new_task_names, new_task_split_paths):
        if not os.path.exists(new_task_split_path):
            os.makedirs(new_task_split_path)
        save_splits_from_cases(new_task_split_path, new_task_splits[new_task_name])

def create_splits_from_splits(new_task_names, new_task_split_paths, old_task, 
    old_task_split_path, nr_folds=5):
    """Creates new split files by distributing ald cases into new tasks. Saves
    these new splits in the specified directories. Note that in the splits, 
    the case name is preceded by [old task name] to show where the case came from.

    param new_task_names (list(str)): a list with new task names
    param new_task_split_paths (list(str)): a list of directories where these 
        split files will be stored. If these don't exist, they are created
    param old_tasks (str): old task name
    param old_task_split_path (str): path to old task split
    """
    new_task_splits = {new_task: [OrderedDict([('train', []), ('val', [])]) 
        for _ in range(nr_folds)] for new_task in new_task_names}
    per_fold_split_cases = get_cases_from_splits(old_task_split_path)
    assert len(per_fold_split_cases) == len(new_task_names) == len(new_task_split_paths)
    assert nr_folds == len(per_fold_split_cases)
    for fold_cases, new_task_name in zip(per_fold_split_cases, new_task_names):
        # The entire training data for the task are the validation cases for the respective fold
        all_val_cases = list(fold_cases['val'])
        # How many case on each new fold?
        nr_val_new_task = distribute_total([1/nr_folds for _ in range(nr_folds)], len(all_val_cases))
        for fold_ix, new_cases_val in enumerate(nr_val_new_task):
            new_val_cases = random.sample(all_val_cases, new_cases_val)
            all_val_cases = [x for x in all_val_cases if x not in new_val_cases]
            new_val_cases = ['['+old_task+']'+case for case in new_val_cases]
            new_task_splits[new_task_name][fold_ix]['val'] += new_val_cases

    # Now set the training cases for each fold
    for new_task in new_task_names:
        for fold_ix in range(nr_folds):
            for fold_ix_val in range(nr_folds):
                if fold_ix != fold_ix_val:
                    new_task_splits[new_task][fold_ix]['train'] += new_task_splits[new_task][fold_ix_val]['val']

    # Assert split integrity
    for fold_ix in range(nr_folds):
        # For the same split, different tasks should not share training samples
        all_train_samples = []
        for new_task in new_task_names:
            new_train_samples = new_task_splits[new_task][fold_ix]['train']
            assert not any(x in all_train_samples for x in new_train_samples)
            all_train_samples += new_train_samples
            # For no fold should a sample be in the train and val split
            assert not any(x in new_train_samples for x in new_task_splits[new_task][fold_ix]['val'])

    # Create splits_final and save for each task
    for new_task_name, new_task_split_path in zip(new_task_names, new_task_split_paths):
        save_splits_from_cases(new_task_split_path, new_task_splits[new_task_name])

def create_new_splits_by_attribute(new_task_names, new_task_split_paths, old_task, 
    old_task_split_path, old_task_info, info_key, boundaries, nr_folds=5):
    """Divides instances of one task based on certain task information.

    param new_task_names (list(str)): a list with new task names
    param new_task_split_paths (list(str)): a list of directories where these 
        split files will be stored. If these don't exist, they are created
    param old_task (str): old task name
    param old_task_split_path (str): where splits_final.pkl is stored
    param old_task_info: a dictionary of dictionaries, where the first key is 
        the case name and the second key, the properties
    param info_key: the property that is considered
    param boundaries: the values that will divide the instance + an upped bound,
        or string values if working with categorical data.
    """
    categorical = False
    if isinstance(boundaries[0], str):
        assert len(new_task_names) == len(new_task_split_paths) == len(boundaries)
        categorical = True
    else:
        assert len(new_task_names) == len(new_task_split_paths) == len(boundaries) - 1

    new_task_splits = {new_task: [OrderedDict([('train', []), ('val', [])]) 
        for _ in range(nr_folds)] for new_task in new_task_names}

    per_fold_split_cases = get_cases_from_splits(old_task_split_path)
    assert nr_folds == len(per_fold_split_cases)
    for fold_ix, fold_cases in enumerate(per_fold_split_cases):
        all_train_cases = list(fold_cases['train'])
        all_val_cases = list(fold_cases['val'])
        # For each new task, add the cases that correspond to the criteria
        for new_task_ix, new_task in enumerate(new_task_names):
            if categorical:
                train_cases = [c for c in all_train_cases if boundaries[new_task_ix] ==  old_task_info[c][info_key]]
                val_cases = [c for c in all_val_cases if boundaries[new_task_ix] ==  old_task_info[c][info_key]]
            else:
                train_cases = [c for c in all_train_cases if boundaries[new_task_ix] 
                    <= old_task_info[c][info_key] < boundaries[new_task_ix+1]]
                val_cases = [c for c in all_val_cases if boundaries[new_task_ix] 
                    <= old_task_info[c][info_key] < boundaries[new_task_ix+1]]
            train_cases = ['['+old_task+']'+case for case in train_cases]
            val_cases = ['['+old_task+']'+case for case in val_cases]
            new_task_splits[new_task][fold_ix]['train'] = train_cases
            new_task_splits[new_task][fold_ix]['val'] = val_cases

    # Assert split integrity
    for fold_ix in range(nr_folds):
        # For the same split, different tasks should not share training samples
        all_train_samples = []
        for new_task in new_task_names:
            new_train_samples = new_task_splits[new_task][fold_ix]['train']
            assert not any(x in all_train_samples for x in new_train_samples)
            all_train_samples += new_train_samples
            # For no fold should a sample be in the train and val split
            assert not any(x in new_train_samples for x in new_task_splits[new_task][fold_ix]['val'])

    # Create splits_final and save for each task
    for new_task_name, new_task_split_path in zip(new_task_names, new_task_split_paths):
        save_splits_from_cases(new_task_split_path, new_task_splits[new_task_name])


def split_to_new_cases(splits):
    new_cases = dict()
    cases = list(splits[0]['train']) + list(splits[0]['val'])
    # Make sure all splits have the same cases
    for fold_splits in splits:
        assert set(cases) == set(list(fold_splits['train']) + list(fold_splits['val']))
    # Make sure there are no repeated cases
    assert len(set(cases)) == len(cases)
    for case in cases:
        # If this throws an error, there were ']' chars in the original cases
        if case.count(']')>1:
            splitted_case = case.split(']')
            orig_task = splitted_case[0]
            orig_case = ']'.join(splitted_case[1:])
        else:
            orig_task, orig_case = case.split(']')
        orig_task = orig_task[1:] # Remove "["
        if orig_task not in new_cases:
            new_cases[orig_task] = []
        new_cases[orig_task].append(orig_case)
    return new_cases