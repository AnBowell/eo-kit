use EOkit::smoothers::sav_golay::single_sav_golay;

#[test]
fn test_sav_golay_filter() {
    let mut input_y: Vec<f64> = Vec::with_capacity(20);
    for i in 0..20 {
        input_y.push((i as f64).powf(2.));
    }

    let input_y_ptr = input_y.as_mut_ptr();

    let data_length = input_y.len();

    single_sav_golay(input_y_ptr, data_length, 4, 3, 0, 1.)
}
