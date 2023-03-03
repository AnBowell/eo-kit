mod gaussian_processes;
pub mod math_utils;
mod ndvi;
pub mod smoothers;

use gaussian_processes::gp::{multiple_gps, single_gp};
use smoothers::{
    sav_golay::{multiple_sav_golays, single_sav_golay},
    whittaker::{multiple_whittakers, single_whittaker},
};

#[no_mangle]
pub extern "C" fn rust_multiple_gps(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    input_size: usize,
    input_indices_ptr: *mut usize,
    input_indices_size: usize,
    output_ptr: *mut f64,
    output_size: usize,
    forecast_spacing: i64,
    forecast_amount: i64,
    length_scale: f64,
    amplitude: f64,
    noise: f64,
    n_threads: i64,
) {
    multiple_gps(
        x_input_ptr,
        y_input_ptr,
        input_size,
        input_indices_ptr,
        input_indices_size,
        output_ptr,
        output_size,
        forecast_spacing,
        forecast_amount,
        length_scale,
        amplitude,
        noise,
        n_threads,
    );
}

#[no_mangle]
pub extern "C" fn rust_single_gp(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    input_size: usize,
    output_ptr: *mut f64,
    output_size: usize,
    forecast_spacing: i64,
    forecast_amount: i64,
    length_scale: f64,
    amplitude: f64,
    noise: f64,
) {
    single_gp(
        x_input_ptr,
        y_input_ptr,
        input_size,
        output_ptr,
        output_size,
        forecast_spacing,
        forecast_amount,
        length_scale,
        amplitude,
        noise,
    );
}

#[no_mangle]
pub extern "C" fn rust_multiple_whittakers(
    y_input_ptr: *mut f64,
    weights_input_ptr: *mut f64,
    input_indices_ptr: *mut usize,
    input_indices_size: usize,
    output_ptr: *mut f64,
    data_length: usize,
    lambda: f64,
    d: i64,
    njobs: i64,
) {
    multiple_whittakers(
        y_input_ptr,
        weights_input_ptr,
        input_indices_ptr,
        input_indices_size,
        output_ptr,
        data_length,
        lambda,
        d,
        njobs,
    );
}

#[no_mangle]
pub extern "C" fn rust_single_whittaker(
    x_input_ptr: *mut f64,
    y_input_ptr: *mut f64,
    weights_input_ptr: *mut f64,
    output_ptr: *mut f64,
    data_length: usize,
    lambda: f64,
    d: i64,
) {
    single_whittaker(
        x_input_ptr,
        y_input_ptr,
        weights_input_ptr,
        output_ptr,
        data_length,
        lambda,
        d,
    );
}

#[no_mangle]
pub extern "C" fn rust_single_sav_golay(
    y_input_ptr: *mut f64,
    output_ptr: *mut f64,
    data_length: usize,
    window_size: i64,
    order: i64,
    deriv: i64,
    delta: f64,
) {
    single_sav_golay(
        y_input_ptr,
        output_ptr,
        data_length,
        window_size,
        order,
        deriv,
        delta,
    )
}

#[no_mangle]
pub extern "C" fn rust_multiple_sav_golays(
    y_input_ptr: *mut f64,
    input_indices_ptr: *mut usize,
    input_indices_size: usize,
    output_ptr: *mut f64,
    data_length: usize,
    window_size: i64,
    order: i64,
    deriv: i64,
    delta: f64,
    n_threads: i64,
) {
    multiple_sav_golays(
        y_input_ptr,
        input_indices_ptr,
        input_indices_size,
        output_ptr,
        data_length,
        window_size,
        order,
        deriv,
        delta,
        n_threads,
    )
}
