# -*- coding: utf-8 -*-
"""This module houses Gaussian Processes wrappers.

The wrappers below call the Rust written library that runs Gaussian processes
smoothing and forecasting. There are multithreaded functions denoted by the
prefix "multiple". More will be added here in future, but for now this is 
sufficient for most EO applications.

"""

import numpy as np
from EOkit.array_utils import check_contig
from .EOkit import lib
from cffi import FFI

ffi = FFI()


def single_gp(
    x_input,
    y_input,
    forecast_spacing,
    forecast_amount,
    length_scale=50.0,
    amplitude=0.5,
    noise=0.01,
):
    """Run a single RBF kernel GP on 1D data.

    This is a wrapper function that lets you specify RBF kernel parameters to
    run a GP smoother. The code also allows for forecasting, simply set the
    forecasting spacing as desired. For example, if x_input is in days and
    a weekly forecast is needed, forecast_spacing would be 7. Then set the
    forecasting amount - this is how many forecasting of forecast_spacing
    are needed.

    Notes
    -----
    NO NANS/INFS SHOULD ENTER THIS FUNCTION.

    Parameters
    ----------
    x_input : ndarray of type float, size (N)
        The x_inputs, in most cases this will be time of some format. It's best
        to subtract the start value from this array to have it start from 0.
    y_input : ndarray of type float, size (N)
        The y_inputs (the variable to be forecast/smoothed). Remove NaNs first.
    forecast_spacing : float
        The spacing of the forecast. E.g. the temporal resolution of the
        forecast.
    forecast_amount : float
        The amount of forecasts of resultion forecast_spacing. Set to 0 for no
        forecasts (just smoothing).
    length_scale : float, optional
        The lengthscale of the RBF kernel. Larger = Smoother, by default 50.0
    amplitude : float, optional
        The amplitude of the RBF kernel, by default 0.5
    noise : float, optional
        Noise of the GP regresion, by default 0.01

    Returns
    -------
    ndarray of type float, size (N)
        A numpy array containing the smoothed/forecasted values. In the future
        this may also include the X variable for ease.

    Examples
    --------
    Below is a simple example of how to use Gaussian Processes.

    >>> data_len = 1000
    >>> days = np.arange(0, data_len, 1., dtype=float)
    >>> vci = np.sin(days) + np.random.standard_normal(data_len) * 2))
    >>> rust_smoothed_data = gaussian_processes.single_gp(days, vci, 0, 0)

    """
    result = np.empty(x_input.size + forecast_amount, dtype=np.float64)

    # If data is not contiguous, using sending a pointer of the NumPy arrays
    # to the Rust library will not work! So good to check.
    x_input = check_contig(x_input)

    y_input = check_contig(y_input)

    result = check_contig(result)

    x_input = x_input.astype(np.float64)

    y_input_mean_removed = (y_input - np.mean(y_input)).astype(np.float64)

    x_input_ptr = ffi.cast("double *", x_input.ctypes.data)
    y_input_ptr = ffi.cast("double *", y_input_mean_removed.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data)

    lib.rust_single_gp(
        x_input_ptr,
        y_input_ptr,
        y_input.size,
        result_ptr,
        result.size,
        forecast_spacing,
        forecast_amount,
        length_scale,
        amplitude,
        noise,
    )

    return result + np.mean(y_input)


def multiple_gps(
    x_inputs,
    y_inputs,
    forecast_spacing,
    forecast_amount,
    length_scale=30,
    amplitude=0.5,
    noise=0.1,
    n_threads=-1,
):
    """Run multiple RBF kernel GPs on 1D data.

    This is a wrapper function that lets you specify RBF kernel parameters to
    run GP smoothers. The code also allows for forecasting, simply set the
    forecasting spacing as desired. For example, if x_input is in days and
    a weekly forecast is needed, forecast_spacing would be 7. Then set the
    forecasting amount - this is how many forecasting of forecast_spacing
    are needed.

    A list of NumPy arrays should be used as inputs. A list was used here so
    that the inputs can be of different lengths, which makes it easier when
    removing cloud cover or other types of null value.

    Notes
    -----
    NO NANS/INFS SHOULD ENTER THIS FUNCTION.

    Parameters
    ----------
    x_inputs : list of ndarrays of type float, size (N)
        A list of NumPy arrays containing the x_input variable.
    y_inputs : list of ndarrays of type float, size (N)
        A list of NumPy arrays containing the y input (the variable to be
        forecast/smoothed). Remove NaNs first.
    forecast_spacing : float
        The spacing of the forecast. E.g. the temporal resolution of the
        forecast.
    forecast_amount : float
        The amount of forecasts of resultion forecast_spacing. Set to 0 for no
        forecasts (just smoothing).
    length_scale : float, optional
        The lengthscale of the RBF kernel. Larger = Smoother, by default 50.0
    amplitude : float, optional
        The amplitude of the RBF kernel, by default 0.5
    noise : float, optional
        Noise of the GP regresion, by default 0.01
    n_threads : int, optional
        Amount of worker threads spawned to complete the task. The default is -1
        which uses all logical processor cores. To tone this down, use something
        between 1 and the number of processor cores you have. Setting this value
        to a number larger than the amount of logical cores you have will most
        likely degreade performance.

    Returns
    -------
    list of ndarrays of type float, size (N)
        A list of numpy arrays containing the smoothed/forecasted values.
        In the future this may also include the X variable for ease.

    """
    number_of_inputs = len(x_inputs)

    index_runner = 0

    start_indices = [0]

    for x_input in x_inputs[:-1]:
        length_of_input = len(x_input)
        index_runner += length_of_input
        start_indices.append(index_runner)

    start_indices = np.array(start_indices, dtype=np.uint64)

    y_inputs_list, y_inputs_means = [], []

    for y in y_inputs:
        mean = y.mean()
        y_inputs_list.append(y - mean)
        y_inputs_means.append(mean)

    y_input_array = np.concatenate(y_inputs_list).ravel().astype(np.float64)
    x_input_array = np.concatenate(x_inputs).ravel().astype(np.float64)

    result = np.empty(
        x_input_array.size + (forecast_amount * number_of_inputs),
        dtype=np.float64,
    )

    x_input_array = check_contig(x_input_array)

    y_input_array = check_contig(y_input_array)

    result = check_contig(result)

    x_input_ptr = ffi.cast("double *", x_input_array.ctypes.data)
    y_input_ptr = ffi.cast("double *", y_input_array.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data)
    start_indices_ptr = ffi.cast("uintptr_t *", start_indices.ctypes.data)

    lib.rust_multiple_gps(
        x_input_ptr,
        y_input_ptr,
        x_input_array.size,
        start_indices_ptr,
        start_indices.size,
        result_ptr,
        result.size,
        forecast_spacing,
        forecast_amount,
        length_scale,
        amplitude,
        noise,
        n_threads,
    )

    results = []

    start_indices[1:] += (
        np.arange(1, len(start_indices[1:]) + 1) * forecast_amount
    ).astype(np.uint64)

    for i in range(0, len(start_indices)):

        if i + 1 >= len(start_indices):

            single_result = result[start_indices[int(i)] :]
        else:

            single_result = result[start_indices[int(i)] : int(start_indices[i + 1])]

        results.append(single_result + y_inputs_means[i])

    return results
