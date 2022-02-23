import numpy as np


def check_type(array):

    if not array.dtype == np.float64:
        return array.astype(np.float64)

    return array


def check_contig(array):

    if not array.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(array)

    return array
