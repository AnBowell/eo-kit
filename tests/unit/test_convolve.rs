use EOkit::math_utils::convolve::{convolve_1d, ConvType};

#[test]
fn test_full_1d_convolve() {
    let x: Vec<f64> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let y: Vec<f64> = vec![1., 2., 3., 4.];

    let z = convolve_1d(&x, &y, ConvType::Valid);

    let actual_z = vec![20., 30., 40., 50., 60., 70., 80.];

    println!("z: {:?}", z);

    assert!(actual_z.iter().eq(z.iter()))
}
