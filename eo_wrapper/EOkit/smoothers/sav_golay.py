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
       Tbe spacing of the samples to which the filter is applied, by default 1

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


def multiple_sav_golay(y_inputs, window_size, order, deriv=0, delta=1, n_threads=-1):
    pass
