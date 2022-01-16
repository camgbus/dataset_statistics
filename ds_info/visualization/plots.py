import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def props_to_pandas(values, columns=[]):
    """Turn a dictionary of dictionaries into a pandas data frame.

    Parameters:
    values dict(str -> dict(str -> value)): dict(name -> dict(prop -> value))
    columns (lst(str)): columns considered for the df, from 'prop' entries

    Returns:
    pd.core.frame.DataFrame: data frame
    """
    data = []
    for key, props in values.items():
        data.append([key]+[props[c] for c in columns])
    return pd.DataFrame(data, columns=['subject']+columns)


def histogram(data, x=None, nr_bins=20, kde=True, hue=None, color=None, figsize=(30,10)):
    """Wrapper for a seaborn histogram that can receive a df or numpy array"""
    is_df = isinstance(data, pd.core.frame.DataFrame)
    if not is_df:
        if isinstance(data, np.ndarray):
            data = data.flatten()
    # Ticks centered around bins
    min_val = min(data)
    max_val = max(data)
    val_width = max_val - min_val
    bin_width = val_width/nr_bins
    val_width = max_val - min_val
    # Plot historam
    plt.figure(figsize=figsize)
    sns.histplot(data=data, x=x, bins=nr_bins, binrange=(min_val, max_val), kde=kde, hue=hue, color=color)
    plt.xticks([round(x, 1) for x in np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width)])