
import numbers
import numpy as np

def zeros_in_format(format_rep):
    if isinstance(format_rep, numbers.Number):
        return 0.0
    if isinstance(format_rep, list):
        return [0]*len(format_rep)
    if isinstance(format_rep, tuple):
        return tuple([0]*len(format_rep))
    if isinstance(format_rep, np.ndarray):
        return np.zeros_like(format_rep)