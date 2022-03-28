import numpy as np
from EOkit.EOkit import lib
from EOkit.array_utils import check_type, check_contig
from cffi import FFI


ffi = FFI()


def single_sav_golay(y_input, window_size, order, deriv=0, delta=1):
    """Run a single Savitzky-golay filter on 1D data.

    The Savitzky-golay smoother fits a polynomial to sliding windows of data
    using a least squares fit. The implementation I have used is similar to that
    found in the SciPy cookbook: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay.

    Parameters
    ----------
    y_input : ndarray of type float, size (N)
        The inputs that are to be smoothed.
    window_size : int
        The size of the sliding window. Generally, the larger the window of data
        points, the smoother the resultant data.
    order : int
        Order of polynomial to fit the data with. Needs to be less than
        window_size - 1.
    deriv : int, optional
        Order of the derivative to smooth, by default 0
    delta : int, optional
       The spacing of the samples to which the filter is applied, by default 1

    Returns
    -------
    ndarray of type float, size (N)
        Smoothed data at y inputs.

    Examples
    --------
    Below is a simple example of how to use the Savitzky-golay smoother.

    >>> data_len = 1000
    >>> vci = (np.sin(np.arange(0, data_len, 1., dtype=float))
    >>>           + np.random.standard_normal(data_len) * 2))
    >>> rust_smoothed_data = sav_golay.single_sav_golay(vci, 7, 2, 0, 1)

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.

    """
    # TODO! Condense all this stuff into a function accross the smoothers.
    data_len = len(y_input)

    result = np.empty(data_len, dtype=np.float64)
    result = check_contig(result)

    y_input = check_contig(y_input)

    y_input_ptr = ffi.cast("double *", y_input.ctypes.data)

    result_ptr = ffi.cast("double *", result.ctypes.data)

    lib.rust_single_sav_golay(
        y_input_ptr, result_ptr, result.size, window_size, order, deriv, delta
    )

    return result


def multiple_sav_golays(y_inputs, window_size, order, deriv=0, delta=1, n_threads=-1):
    """Run many Savitzky-golay smoothers on 1D data in a multithread manner.

    This runs an identical algorithm to the single_sav_golay function. However,
    this functions takes a list of y_inputs. Rust is then used under the hood
    to multithread each Savitzky-golay smoother as a task leading to faster
    computation for pixel-based problems!

    I have used a list here as apposed to a 2-D array so that arrays of different
    lengths can be supplied.

    Parameters
    ----------
    y_inputs : list of ndarrays of type float, size (N)
        A list of numpy arrays containing the values to be smoothed.
    window_size : int
        The size of the sliding window. Generally, the larger the window of data
        points, the smoother the resultant data.
    order : int
        Order of polynomial to fit the data with. Needs to be less than
        window_size - 1.
    deriv : int, optional
        Order of the derivative to smooth, by default 0
    delta : int, optional
        The spacing of the samples to which the filter is applied, by default 1
    n_threads : int, optional
        Amount of worker threads spawned to complete the task. The default is -1
        which uses all logical processor cores. To tone this down, use something
        between 1 and the number of processor cores you have. Setting this value
        to a number larger than the amount of logical cores you have will most
        likely degreade performance, by default -1

    Returns
    -------
    list of ndarrays of type float, size (N)
        A list of numpy arrays containing the smoothed data at y_inputs.

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.

    """

    index_runner = 0

    start_indices = [0]

    for y_input in y_inputs[:-1]:
        length_of_input = len(y_input)
        index_runner += length_of_input
        start_indices.append(index_runner)

    start_indices = np.array(start_indices, dtype=np.uint64)

    y_input_array = np.concatenate(y_inputs).ravel().astype(np.float64)

    result = np.empty(y_input_array.size, dtype=np.float64)

    y_input_array = check_contig(y_input_array)
    result = check_contig(result)
    start_indices = check_contig(start_indices)

    y_input_ptr = ffi.cast("double *", y_input_array.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data)
    start_indices_ptr = ffi.cast("uintptr_t *", start_indices.ctypes.data)

    lib.rust_multiple_sav_golays(
        y_input_ptr,
        start_indices_ptr,
        start_indices.size,
        result_ptr,
        result.size,
        window_size,
        order,
        deriv,
        delta,
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
