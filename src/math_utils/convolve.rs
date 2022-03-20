pub enum ConvType {
    Valid,
    Full,
}

pub fn convolve_1d(a: &Vec<f64>, v: &Vec<f64>, conv: ConvType) -> Vec<f64> {
    let na = a.len();
    let nv = v.len();

    match conv {
        ConvType::Full => {
            let n = na + nv - 1;
            let mut out_vec: Vec<f64> = vec![0.; n];

            for i in 0..n {
                let jmn = if i >= nv - 1 { i - (nv - 1) } else { 0 };
                let jmx = if i < na - 1 { i } else { na - 1 };

                for j in jmn..jmx + 1 {
                    out_vec[i] += a[j] * v[i - j]
                }
            }
            return out_vec;
        }
        ConvType::Valid => {
            let min_v = if na < nv { a } else { v };
            let max_v = if na < nv { v } else { a };

            let n = std::cmp::max(na, nv) - std::cmp::min(na, nv) + 1;

            let mut out_vec: Vec<f64> = vec![0.; n];

            for i in 0..n {
                for (k, j) in (0..(min_v.len())).rev().enumerate() {
                    out_vec[i] += min_v[j] * max_v[k + i];
                }
            }

            return out_vec;
        }
    }
}
