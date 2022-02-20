import numpy as np
from EOkit.array_utils import check_contig
from .EOkit import lib
from cffi import FFI
ffi = FFI()

def rust_run_single_gp(
    x_input,
    y_input,
    forecast_spacing,
    forecast_amount,
    length_scale=50.0,
    amplitude=0.5,
    noise=0.01,
):
    """Wrapper function to run a single GP through the Rust library.
    Simply pass the x and y you want to fit, a forecast of spacing and amount,
    and this wrapper will call the Rust function. The y input in this case
    should not have the mean removed and any Nans/infs etc should be removed
    from the dataset.


    Args:
        x_input ([float64]): The x-axis input. For VCI forcasting, time in days.
        y_input ([float64]): The y-axis input. For VCI forecasting, the VCI.
        forecast_spacing (int): The spacing between the forecast. For weekly, 7.
        forecast_amount (int): The amount of forecasts. 10 would yield 10 forecasts of forecasting_spacing.
        length_scale (float, optional): Lengthscale of the squared-exp Kernel. Defaults to 50.
        amplitude (float, optional): Amplitude of the squared-exp Kernel. Defaults to 0.5.
        noise (float, optional): Noise of the GP regression. Defaults to 0.01.

    Returns:
        [float64]: The result of the GP regression sampled at each input as well as
        the requested forecasts.
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


def run_multiple_gps(
    x_inputs,
    y_inputs_mean_removed,
    forecast_spacing,
    forecast_amount,
    length_scale=30,
    amplitude=0.5,
    noise=0.1,
):
    """Wrapper function to run multiple GPs multithreaded through the Rust library.

    This function is a little more complex compared to the single GP function.
    The inputs/outputs should be a list of numpy arrays that you want to forecast.
    All of the arrays are then combined and the indicies of where each one starts
    is saved. This is then all passed to the rust function and a train/forecast
    cycle is performed on each one. This wrapper function then unpacks the results
    back into a list of arrays.

    Note: The y inputs should have their mean removed. This is handled in the
    single GP function, but here, it is more effiecent for the user to do it.

    Args:
        x_inputs ([[float]]): A list of numpy arrays containing the x-axis input.
        y_inputs_mean_removed ([[float]]): A list of numpy arrays containing the y-axis input.
            These arrays should all have a mean of zero! Remove the mean before calling it.
        forecast_spacing (int): The spacing between the forecast. For weekly, 7.
        forecast_amount (int): The amount of forecasts. 10 would yield 10 forecasts of forecasting_spacing.
        length_scale (float, optional): Lengthscale of the squared-exp Kernel. Defaults to 50.
        amplitude (float, optional): Amplitude of the squared-exp Kernel. Defaults to 0.5.
        noise (float, optional): Noise of the GP regression. Defaults to 0.01.

    Returns:
        [[float64]]: A list of arrays containing the results of each GP forecast.
    """

    number_of_inputs = len(x_inputs)

    index_runner = 0

    start_indices = [0]

    for x_input in x_inputs[:-1]:
        length_of_input = len(x_input)
        index_runner += length_of_input
        start_indices.append(index_runner)

    start_indices = np.array(start_indices, dtype=np.uint64)

    y_input_array = np.concatenate(y_inputs_mean_removed).ravel().astype(np.float64)
    x_input_array = np.concatenate(x_inputs).ravel().astype(np.float64)

    result = np.empty(
        x_input_array.size + (forecast_amount * number_of_inputs), dtype=np.float64
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

        results.append(single_result)

    return results
