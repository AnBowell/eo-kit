use sprs::{CsMat, TriMatBase};
use sprs::{DontCheckSymmetry, FillInReduction::ReverseCuthillMcKee};
use sprs_ldl::Ldl;

use tokio::runtime::{self};
use tokio::task::JoinHandle;

pub fn multiple_whittakers(
    y_input_ptr: *mut f64,
    weights_input_ptr: *mut f64,
    input_indices_ptr: *mut usize,
    input_indices_size: usize,
    output_ptr: *mut f64,
    data_length: usize,
    lambda: f64,
    d: i64,
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

    let weights_input: &mut [f64] = unsafe {
        assert!(!weights_input_ptr.is_null());
        std::slice::from_raw_parts_mut(weights_input_ptr, data_length)
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
        let (y_input_slice, weights_input_slice) =
            if i + 1_usize >= input_indices_size {
                (
                    &y_input[input_indices[i]..],
                    &weights_input[input_indices[i]..],
                )
            } else {
                (
                    &y_input[input_indices[i]..input_indices[i + 1]],
                    &weights_input[input_indices[i]..input_indices[i + 1]],
                )
            };

        handles.push(rt.spawn(async move {
            let slice_length = y_input_slice.len();
            let e: CsMat<f64> = CsMat::eye(slice_length);

            let diff_mat = diff(&e, d);

            let diags = (0..slice_length).collect::<Vec<usize>>();

            let weights_matrix = TriMatBase::from_triplets(
                (slice_length, slice_length),
                diags.clone(),
                diags,
                weights_input_slice.to_vec(),
            )
            .to_csc();

            let to_solve: CsMat<f64> = &weights_matrix
                + &(&(&diff_mat.transpose_view() * &diff_mat) * lambda);

            let ldl = Ldl::new()
                .fill_in_reduction(ReverseCuthillMcKee)
                .check_symmetry(DontCheckSymmetry)
                .numeric(to_solve.view())
                .expect("Could not create solver.");

            let smoothed_y = ldl.solve(
                weights_input_slice
                    .iter()
                    .zip(y_input_slice)
                    .map(|(a, b)| *a * *b)
                    .collect::<Vec<f64>>(),
            );

            return smoothed_y;
        }));
    }

    // TODO There will probably be a more efficient way of doing this.
    for i in 0..input_indices_size {
        let result = rt.block_on(&mut handles[i]).unwrap().to_vec();

        let this_index = input_indices[i];
        for result_index in 0..result.len() {
            output[this_index + result_index] = result[result_index];
        }
    }
}

pub fn single_whittaker(
    y_input_ptr: *mut f64,
    weights_input_ptr: *mut f64,
    output_ptr: *mut f64,
    data_length: usize,
    lambda: f64,
    d: i64,
) {
    let y_input: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(y_input_ptr, data_length)
    };

    let weights: &mut [f64] = unsafe {
        assert!(!weights_input_ptr.is_null());
        std::slice::from_raw_parts_mut(weights_input_ptr, data_length)
    };

    let output: &mut [f64] = unsafe {
        assert!(!y_input_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, data_length)
    };

    let e: CsMat<f64> = CsMat::eye(data_length as usize);

    let diff_mat = diff(&e, d);

    let diags = (0..data_length).collect::<Vec<usize>>();
    let weights_matrix = TriMatBase::from_triplets(
        (data_length, data_length),
        diags.clone(),
        diags,
        weights.to_vec(),
    )
    .to_csc();

    let to_solve: CsMat<f64> = &weights_matrix
        + &(&(&diff_mat.transpose_view() * &diff_mat) * lambda);

    let ldl = Ldl::new()
        .fill_in_reduction(ReverseCuthillMcKee)
        .check_symmetry(DontCheckSymmetry)
        .numeric(to_solve.view())
        .expect("Could not create solver.");

    let smoothed_y = ldl.solve(
        weights
            .iter()
            .zip(y_input)
            .map(|(a, b)| *a * *b)
            .collect::<Vec<f64>>(),
    );

    for i in 0..data_length {
        output[i] = smoothed_y[i]
    }
}

fn diff(e: &CsMat<f64>, d: i64) -> CsMat<f64> {
    if d == 0 {
        return e.clone();
    } else {
        let e1 = e.slice_outer(0..e.rows() - 1);
        let e2 = e.slice_outer(1..e.rows());
        return diff(&(&e2 - &e1), d - 1);
    }
}
