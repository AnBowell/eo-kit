import numpy as np
from ctypes import c_double, c_int64
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
    if not x_input.flags["C_CONTIGUOUS"]:
        x_input = np.ascontiguousarray(x_input)

    if not y_input.flags["C_CONTIGUOUS"]:
        y_input = np.ascontiguousarray(y_input)

    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)
        
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


