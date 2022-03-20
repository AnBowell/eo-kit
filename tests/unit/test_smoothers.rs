use EOkit::smoothers::sav_golay::single_sav_golay;

#[test]
fn test_sav_golay_filter() {
    let mut input_y: Vec<f64> = Vec::with_capacity(20);
    let mut output_y: Vec<f64> = Vec::with_capacity(20);

    for i in 0..20 {
        input_y.push((i as f64).powf(2.));
    }

    let input_y_ptr = input_y.as_mut_ptr();
    let output_ptr = output_y.as_mut_ptr();

    let data_length = input_y.len();

    single_sav_golay(input_y_ptr, output_ptr, data_length, 4, 3, 0, 1.);

    let sav_gol_output: Vec<f64> = unsafe {
        assert!(!output_ptr.is_null());
        std::slice::from_raw_parts_mut(output_ptr, data_length)
    }
    .to_vec();

    let scipy_res = vec![
        -1.38777878e-16,
        1.00000000e+00,
        4.00000000e+00,
        9.00000000e+00,
        1.60000000e+01,
        2.50000000e+01,
        3.60000000e+01,
        4.90000000e+01,
        6.40000000e+01,
        8.10000000e+01,
        1.00000000e+02,
        1.21000000e+02,
        1.44000000e+02,
        1.69000000e+02,
        1.96000000e+02,
        2.25000000e+02,
        2.56000000e+02,
        2.89000000e+02,
        3.24000000e+02,
        3.61000000e+02,
    ];

    for (res, sci) in sav_gol_output.iter().zip(scipy_res) {
        assert!((res - sci).abs() < 1e-8)
    }
}
