use ndarray::arr2;

pub fn single_sav_golay(
    y_input_ptr: *mut f64,
    data_length: usize,
    window_size: i64,
    order: i64,
) {
    let order_range = 0..order + 1;
    let half_window = (window_size as f64 - 1_f64 / 2_f64).floor();
}
