use EOkit::math_utils::convolve::{convolve_1d, ConvType};

#[test]
fn test_1d_convolve() {
    let x: Vec<f64> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let y: Vec<f64> = vec![1., 2., 3., 4.];

    let z_valid = convolve_1d(&x, &y, ConvType::Valid);
    let z_full = convolve_1d(&x, &y, ConvType::Full);

    let actual_z_valid = vec![20., 30., 40., 50., 60., 70., 80.];
    let actual_z_full = vec![
        1., 4., 10., 20., 30., 40., 50., 60., 70., 80., 79., 66., 40.,
    ];

    assert!(actual_z_valid.iter().eq(z_valid.iter()));
    assert!(actual_z_full.iter().eq(z_full.iter()));
}
