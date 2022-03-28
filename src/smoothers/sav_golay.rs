use nalgebra::DMatrix;

use crate::math_utils::convolve::{convolve_1d, ConvType};

use tokio::runtime::{self};
use tokio::task::JoinHandle;

pub fn multiple_sav_golays(
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
    let rt = if n_threads < 0 {
        runtime::Builder::new_multi_thread()
            .build()
            .expect("Could not build Tokio runtime.")
    } else {
        runtime::Builder::new_multi_thread()
            .worker_threads(n_threads as usize)
            .build()
            .expect(
                format!(
                    "Could not build Tokio runtime with {} threads",
                    n_threads
                )
                .as_str(),
            )
    };

    let y_input: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(y_input_ptr, data_length)
    };

    let input_indices: &mut [usize] = unsafe {
        assert!(!input_indices_ptr.is_null());
        std::slice::from_raw_parts_mut(input_indices_ptr, input_indices_size)
    };

    let output: &mut [f64] = unsafe {
        assert!(!output_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, data_length)
    };

    let mut handles: Vec<JoinHandle<Vec<f64>>> =
        Vec::with_capacity(data_length);

    for i in 0..input_indices_size {
        let y_input_slice = if i + 1_usize >= input_indices_size {
            &y_input[input_indices[i]..]
        } else {
            &y_input[input_indices[i]..input_indices[i + 1]]
        };

        handles.push(rt.spawn(async move {
            let slice_length = y_input_slice.len();

            let half_window =
                ((window_size as f64 - 1_f64) / 2_f64).floor() as i64;

            let mut b_vec = Vec::with_capacity(
                (((half_window * 2) + 1) * (order + 1)) as usize,
            );

            for k in -half_window..half_window + 1 {
                for i in 0..(order + 1) {
                    b_vec.push((k as f64).powf(i as f64));
                }
            }

            let b = DMatrix::from_vec(
                (order + 1) as usize,
                ((half_window * 2) + 1) as usize,
                b_vec,
            )
            .transpose();

            let inverse_b = b.pseudo_inverse(1e-15).unwrap();

            let mut row = (inverse_b.row(deriv as usize)
                * (delta.powf(deriv as f64))
                * factorial(deriv) as f64)
                .as_slice()
                .to_vec();

            let mut first_vals: Vec<f64> = y_input_slice
                [(1 as usize)..((half_window + 1) as usize)]
                .iter()
                .rev()
                .map(|x| y_input_slice[0] - (x - y_input_slice[0]).abs())
                .collect();

            let last_vals = y_input_slice
                [(slice_length - half_window as usize - 1)..slice_length - 1]
                .iter()
                .rev()
                .map(|x| {
                    y_input_slice[slice_length - 1]
                        + (x - y_input_slice[slice_length - 1]).abs()
                });

            first_vals.extend(y_input_slice.iter());
            first_vals.extend(last_vals);

            row.reverse();

            let result = convolve_1d(&row, &first_vals, ConvType::Valid);

            return result;
        }));
    }

    // TODO There will probably be a more efficient way of doing this.
    for i in 0..input_indices_size {
        let result = rt.block_on(&mut handles[i]).unwrap();

        let this_index = input_indices[i];
        for result_index in 0..result.len() {
            output[this_index + result_index] = result[result_index];
        }
    }
}

pub fn single_sav_golay(
    y_input_ptr: *mut f64,
    output_ptr: *mut f64,
    data_length: usize,
    window_size: i64,
    order: i64,
    deriv: i64,
    delta: f64,
) {
    let y_input: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(y_input_ptr, data_length)
    };
    let output: &mut [f64] = unsafe {
        assert!(!output_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, data_length)
    };

    let half_window = ((window_size as f64 - 1_f64) / 2_f64).floor() as i64;

    let mut b_vec =
        Vec::with_capacity((((half_window * 2) + 1) * (order + 1)) as usize);

    for k in -half_window..half_window + 1 {
        for i in 0..(order + 1) {
            b_vec.push((k as f64).powf(i as f64));
        }
    }

    let b = DMatrix::from_vec(
        (order + 1) as usize,
        ((half_window * 2) + 1) as usize,
        b_vec,
    )
    .transpose();

    let inverse_b = b.pseudo_inverse(1e-15).unwrap();

    let mut row = (inverse_b.row(deriv as usize)
        * (delta.powf(deriv as f64))
        * factorial(deriv) as f64)
        .as_slice()
        .to_vec();

    let mut first_vals: Vec<f64> = y_input
        [(1 as usize)..((half_window + 1) as usize)]
        .iter()
        .rev()
        .map(|x| y_input[0] - (x - y_input[0]).abs())
        .collect();

    let last_vals = y_input
        [(data_length - half_window as usize - 1)..data_length - 1]
        .iter()
        .rev()
        .map(|x| {
            y_input[data_length - 1] + (x - y_input[data_length - 1]).abs()
        });

    first_vals.extend(y_input.iter());
    first_vals.extend(last_vals);

    row.reverse();

    let result = convolve_1d(&row, &first_vals, ConvType::Valid);
    for i in 0..data_length {
        output[i] = result[i]
    }
}

fn factorial(num: i64) -> i64 {
    (1..=num).product()
}
