use nalgebra::DMatrix;

fn factorial(num: i64) -> i64 {
    (1..=num).product()
}

pub fn single_sav_golay(
    y_input_ptr: *mut f64,
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

    let half_window = ((window_size as f64 - 1_f64) / 2_f64).floor() as i64;

    let mut b_vec =
        Vec::with_capacity(((half_window * 2) * (order + 1)) as usize);

    for k in -half_window..half_window {
        for i in 0..order + 1 {
            b_vec.push((k as f64).powf(i as f64));
        }
    }

    let b = DMatrix::from_vec(
        (half_window * 2) as usize,
        (order + 1) as usize,
        b_vec,
    );
    let inverse_b = b.pseudo_inverse(1e-15).unwrap();

    let row = inverse_b.row(deriv as usize)
        * (delta.powf(deriv as f64))
        * factorial(deriv) as f64;

    // let first_vals = y_input[0] - (y_input[1..half_window + 1] - y_input[0]);
    // let last_vals = y_input[data_length - 1]
    //     - (y_input[-half_window - 1..-1] - y_input[-1]);

    println!("Array: b {}", row);
}
