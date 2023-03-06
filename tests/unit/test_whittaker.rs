use EOkit::smoothers::whittaker::single_whittaker;

#[test]
fn test_whittaker_smoother() {
    let mut input_x: Vec<f64> = Vec::with_capacity(20);
    let mut input_y: Vec<f64> = Vec::with_capacity(20);
    let mut output_y: Vec<f64> = Vec::with_capacity(20);

    let mut weights: Vec<f64> = Vec::with_capacity(20);

    for i in 0..20 {
        input_x.push(i as f64);
        input_y.push((i as f64).powf(2.));
        weights.push(1.0);
    }

    let input_x_ptr = input_x.as_mut_ptr();

    let input_y_ptr = input_y.as_mut_ptr();
    let output_ptr = output_y.as_mut_ptr();
    let weights_ptr = weights.as_mut_ptr();

    let data_length = input_y.len();

    single_whittaker(
        input_x_ptr,
        input_y_ptr,
        weights_ptr,
        output_ptr,
        data_length,
        5.0,
        3,
    );

    let whittaker_output: Vec<f64> = unsafe {
        assert!(!output_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, data_length)
    }
    .to_vec();

    println!("Whittaker_output: {:?}", whittaker_output);
}
