# -*- coding: utf-8 -*-
"""This module houses the Whittaker smoothing algorithm wrappers.

The wrappers below call the Rust written library that runs the Whittaker
smoother. There are multithreaded functions denoted by the prefix "multiple". 
More will be added here in future, but for now this is sufficient for most EO 
applications.

"""

import numpy as np
from EOkit.EOkit import lib
from EOkit.array_utils import check_type, check_contig
from cffi import FFI

ffi = FFI()

# Todo put references in Doc style.
def single_whittaker(y_input, weights_input, lambda_, d):
    """Run a single Whittaker smoother on 1D data.

    The Whittaker smoother is based on penalized least squares. The main paper
    is behind a pay-wall, though the supporting information is available and
    contains extra details about the implementation of this algorithm [1].

    Parameters
    ----------
    y_input : ndarray of type float, size (N)
        The inputs that are to be smoothed.
    weights_input : ndarray of type float, size (N)
        The weight that should be given to each input, where 0. ignores a given
        point (useful for interpolation) and 1. applies the full weight.
    lambda_ : float
        Smoothing coefficient. Larger = smoother.
    d : float
        Order of the smoothing/interpolation. 1 = linear and so on.

    Returns
    -------
    ndarray of type float, size (N)
        Smoothed data at y inputs.

    Examples
    --------
    Below is a simple example of how to use the Whittaker smoother.

    >>> data_len = 1000
    >>> vci = (np.sin(np.arange(0, data_len, 1., dtype=float))
    >>>           + np.random.standard_normal(data_len) * 2))
    >>> weights = np.full(vci.size, 1.0, dtype=np.float64)
    >>> rust_smoothed_data = whittaker.single_whittaker(vci, weights, 5, 3)

    References
    ----------
    .. [1] Eilers, Paul H. C. "A Perfect Smoother", Analytical Chemistry,
           2003, 75 (14), 3631-3636, 10.1021/ac034173t,
           https://pubs.acs.org/doi/pdf/10.1021/ac034173t

    """
    data_len = len(y_input)

    result = np.empty(data_len, dtype=np.float64)
    result = check_contig(result)

    y_input = check_contig(y_input)
    weights_input = check_contig(weights_input)

    y_input = check_type(y_input)
    weights_input = check_type(weights_input)

    y_input_ptr = ffi.cast("double *", y_input.ctypes.data)
    weights_input_ptr = ffi.cast("double *", weights_input.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data)

    lib.rust_single_whittaker(
        y_input_ptr,
        weights_input_ptr,
        result_ptr,
        result.size,
        lambda_,
        d,
    )

    return result


def multiple_whittakers(y_inputs, weights_inputs, lambda_, d, n_threads=-1):
    """Run many Whittaker smoothers on 1D data in a multithreaded manner.

    This runs an identical algorithm to the single_whittaker function. However,
    this functions takes a list of y inputs and corresponding list of weights.
    Rust is then used to multithread each Whittaker as a task leading to faster
    computations for pixel-based problems!

    I have used a list here as apposed to a 2-D array so that arrays of different
    lengths can be supplied.

    Parameters
    ----------
    y_inputs : list of ndarrays of type float, size (N)
        A list of numpy arrays containing the values to be smoothed.
    weights_inputs : list of ndarrays of type float, size (N)
        A list of numpy arrays containing the weights for the values to be
        smoothed. 0. ignores a given point (for interpolation) whereas 1.
        takes the point into full consideration.
    lambda_ : float
        Smoothing coefficient. Larger = smoother.
    d : float
        Order of smoothing. 1. for linear.
    n_threads : int, optional
        Amount of worker threads spawned to complete the task. The default is -1
        which uses all logical processor cores. To tone this down, use something
        between 1 and the number of processor cores you have. Setting this value
        to a number larger than the amount of logical cores you have will most
        likely degreade performance.

    Returns
    -------
    list of ndarrays of type float, size (N)
        A list of numpy arrays containing the smoothed data at y_inputs.
    """

    index_runner = 0

    start_indices = [0]

    for y_input in y_inputs[:-1]:
        length_of_input = len(y_input)
        index_runner += length_of_input
        start_indices.append(index_runner)

    start_indices = np.array(start_indices, dtype=np.uint64)

    y_input_array = np.concatenate(y_inputs).ravel().astype(np.float64)
    weight_input_array = np.concatenate(weights_inputs).ravel().astype(np.float64)

    result = np.empty(y_input_array.size, dtype=np.float64)

    y_input_array = check_contig(y_input_array)
    weight_input_array = check_contig(weight_input_array)
    result = check_contig(result)
    start_indices = check_contig(start_indices)

    y_input_ptr = ffi.cast("double *", y_input_array.ctypes.data)
    weights_input_ptr = ffi.cast("double *", weight_input_array.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data)
    start_indices_ptr = ffi.cast("uintptr_t *", start_indices.ctypes.data)

    lib.rust_multiple_whittakers(
        y_input_ptr,
        weights_input_ptr,
        start_indices_ptr,
        start_indices.size,
        result_ptr,
        result.size,
        lambda_,
        d,
        n_threads,
    )

    results = []

    for i in range(0, len(start_indices)):

        if i + 1 >= len(start_indices):

            single_result = result[start_indices[int(i)] :]
        else:

            single_result = result[start_indices[int(i)] : int(start_indices[i + 1])]

        results.append(single_result)

    return results
