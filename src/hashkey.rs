use constants::*;
use nalgebra;
use num::{cast, Float};
use std::f64::consts::PI;
use std::fmt::{Debug, Display};

type WeightsType = nalgebra::Matrix<
    f_t,
    GradientVectorSizeType,
    GradientVectorSizeType,
    nalgebra::MatrixArray<f_t, GradientVectorSizeType, GradientVectorSizeType>,
>;

fn sobel_filter(input: &ImagePatch) -> (GradientBlock, GradientBlock) {
    let horiz_filter =
        nalgebra::Matrix3::new(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0).transpose();
    let vertical_filter =
        nalgebra::Matrix3::new(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0).transpose();

    // TODO: I wrote out the entire filter so that I could guarantee that it
    // was doing what the Python implementation was doing. This section
    // can totally be rewritten for performance.
    let mut g_x: GradientBlock = GradientBlock::zeros();
    let mut g_y: GradientBlock = GradientBlock::zeros();
    for x in 1..PATCH_SIZE - 1 {
        for y in 1..PATCH_SIZE - 1 {
            let mut h_p = 0.0;
            let mut v_p = 0.0;
            for i in [-1 as i64, 0, 1].iter() {
                for j in [-1 as i64, 0, 1].iter() {
                    let patch_offset: (usize, usize) =
                        ((x as i64 + i) as usize, (y as i64 + j) as usize);
                    let filter_offset: (usize, usize) = ((i + 1) as usize, (j + 1) as usize);
                    h_p += input[patch_offset] * vertical_filter[filter_offset];
                    v_p += input[patch_offset] * horiz_filter[filter_offset];
                }
            }
            g_x[(x - 1, y - 1)] = h_p;
            g_y[(x - 1, y - 1)] = v_p;
        }
    }

    (g_x, g_y)
}

// TODO: Currently there's a lot of casting happening in this function!
// That's because we need to maintain compatibility with my changes
// to movehand's code which uses Python floats (i.e f64) along with
// numpy arrays that are f32.
fn eigendecomposition<T: 'static + From<f_t> + Float + Debug + Display>(
    m: &nalgebra::Matrix2<f_t>,
) -> (nalgebra::Vector2<T>, nalgebra::Matrix2<T>) {
    let (a, b, c, d) = (
        cast(m[(0, 0)]).unwrap(),
        cast(m[(0, 1)]).unwrap(),
        cast(m[(1, 0)]).unwrap(),
        cast(m[(1, 1)]).unwrap(),
    );
    if b * c <= cast(1e-20).unwrap() {
        (
            nalgebra::Vector2::new(a, d),
            nalgebra::Matrix2::new(
                cast(1.0).unwrap(),
                cast(0.0).unwrap(),
                cast(0.0).unwrap(),
                cast(1.0).unwrap(),
            ),
        )
    } else {
        let tr = a + d;
        let det = a * d - b * c;
        let s = ((tr / cast(2.0).unwrap()).powf(cast(2.0).unwrap()) - det).sqrt();
        let lamb = nalgebra::Vector2::new(tr / cast(2.0).unwrap() + s, tr / cast(2.0).unwrap() - s);

        let ss = (((a - d) / cast(2.0).unwrap()).powi(2) + b * c)
            .max(cast(0.0).unwrap())
            .sqrt();
        let mut ev = if a - d < cast(0.0).unwrap() {
            nalgebra::Matrix2::new(
                c,
                (a - d) / cast(2.0).unwrap() - ss,
                -(a - d) / cast(2.0).unwrap() + ss,
                b,
            )
        } else {
            nalgebra::Matrix2::new(
                (a - d) / cast(2.0).unwrap() + ss,
                c,
                b,
                -(a - d) / cast(2.0).unwrap() - ss,
            )
        };

        let n1 = (ev[(0, 0)].powi(2) + ev[(0, 1)].powi(2)).sqrt();
        ev[0] = ev[0] / n1;

        let n2 = (ev[(1, 0)].powi(2) + ev[(1, 1)].powi(2)).sqrt();
        ev[1] = ev[1] / n2;

        (lamb, ev)
    }
}

pub fn hashkey<T: From<u8> + Copy>(block: &ImagePatch) -> (T, T, T) {
    let (gy, gx) = sobel_filter(block);
    let gx: GradientVector = GradientVector::from_column_slice(gx.as_slice());
    let gy: GradientVector = GradientVector::from_column_slice(gy.as_slice());

    type GType = nalgebra::Matrix<
        f_t,
        GradientVectorSizeType,
        nalgebra::U2,
        nalgebra::MatrixArray<f_t, GradientVectorSizeType, nalgebra::U2>,
    >;

    let mut g = GType::zeros();
    g.set_column(0, &gx);
    g.set_column(1, &gy);

    let weights_diag = WeightsType::from_diagonal(&GradientVector::from_row_slice(&WEIGHTS));

    let gtwg = g.transpose() * weights_diag * g;
    let wv: (nalgebra::Vector2<f64>, nalgebra::Matrix2<f64>) = eigendecomposition(&gtwg);
    let (w, v) = wv;

    let theta = f64::atan2(v[(1, 0)], v[(0, 0)]);
    let theta = if theta < 0.0 { theta + PI } else { theta };
    let lambda = w[0];

    let sqrtlambda1 = w[0].sqrt();
    let sqrtlambda2 = w[1].sqrt();
    let u = if sqrtlambda1 + sqrtlambda2 == 0.0 {
        0.0
    } else if w[0] < 0.0 || w[1] < 0.0 {
        // TODO: Eventually eliminate this case once the training process
        // is in pure Rust. Currently only kept for compatibility with
        // movehand's implementation (numpy handles negative sqrt
        // differently than Rust).
        0.33
    } else {
        (sqrtlambda1 - sqrtlambda2) / (sqrtlambda1 + sqrtlambda2)
    };

    let angle = (theta / PI * Q_ANGLE as f64).floor();
    let strength: T = if lambda < 0.0001 {
        0
    } else if lambda > 0.001 {
        2
    } else {
        1
    }
    .into();
    let coherence: T = if u < 0.25 {
        0
    } else if u > 0.5 {
        2
    } else {
        1
    }
    .into();

    let angle: T = if angle > 23.0 {
        23
    } else if angle < 0.0 {
        0
    } else {
        angle as u8
    }
    .into();

    (angle, strength, coherence)
}

#[cfg(test)]
mod tests {
    use constants::*;
    use flate2::read::GzDecoder;
    use hashkey::*;
    use nalgebra;
    use std::fs;
    use std::io::prelude::*;
    use std::io::BufReader;

    fn get_test_patch() -> ImagePatch {
        let mut patch: [f_t; PATCH_SIZE * PATCH_SIZE] = [0.0; PATCH_SIZE * PATCH_SIZE];
        for x in 0..PATCH_SIZE {
            for y in 0..PATCH_SIZE {
                patch[x * PATCH_SIZE + y] = f_t::cos(x as f_t) * f_t::sin(y as f_t);
            }
        }
        ImagePatch::from_row_slice(&patch).transpose() / 5.0
    }

    #[test]
    fn test_sobel() {
        let patch = get_test_patch();
        println!("Patch: {}", patch);
        println!("Sobel: {}", sobel_filter(&patch).0);
    }

    #[test]
    fn test_pathological_case() {
        let patch_arr: [f_t; 121] = [
            0.2509804, 0.25490198, 0.25882354, 0.2627451, 0.26666668, 0.27058825, 0.27450982,
            0.2784314, 0.28235295, 0.28627455, 0.29019612, 0.25490198, 0.25882354, 0.2627451,
            0.26666668, 0.27058825, 0.27450982, 0.2784314, 0.28235298, 0.28627455, 0.29019612,
            0.2941177, 0.25882354, 0.2627451, 0.26666668, 0.27058825, 0.27450982, 0.2784314,
            0.28235295, 0.28627455, 0.29019612, 0.2941177, 0.29803923, 0.2627451, 0.26666668,
            0.27058825, 0.27450982, 0.2784314, 0.28235298, 0.28627455, 0.2911765, 0.29607844, 0.3,
            0.30392158, 0.26666668, 0.27058825, 0.27450982, 0.2784314, 0.28235295, 0.28627455,
            0.29019612, 0.29607844, 0.3019608, 0.3058824, 0.30980396, 0.27058825, 0.27450982,
            0.2784314, 0.28235298, 0.28627455, 0.2911765, 0.29607844, 0.30098042, 0.3058824,
            0.30980396, 0.31372553, 0.27450982, 0.2784314, 0.28235295, 0.28627455, 0.29019612,
            0.29607844, 0.3019608, 0.3058824, 0.30980396, 0.31372553, 0.31764707, 0.2784314,
            0.28235298, 0.28627455, 0.2911765, 0.29607844, 0.30098042, 0.3058824, 0.30980396,
            0.31372553, 0.31862748, 0.32352942, 0.28235295, 0.28627455, 0.29019612, 0.29607844,
            0.3019608, 0.3058824, 0.30980396, 0.31372553, 0.31764707, 0.32352942, 0.32941177,
            0.28627455, 0.29019612, 0.2941177, 0.30000004, 0.3058824, 0.30980396, 0.31372553,
            0.31862748, 0.32352942, 0.32843137, 0.33333334, 0.29019612, 0.2941177, 0.29803923,
            0.30392158, 0.30980396, 0.31372553, 0.31764707, 0.32352942, 0.32941177, 0.33333334,
            0.3372549,
        ];

        let patch: ImagePatch = ImagePatch::from_row_slice(&patch_arr);

        // Annoying floating-point imprecision issues (when compared to numpy)
        // cause issues here :/ . This case sits right at a bucket boundary.
        let result = hashkey::<u8>(&patch);
        println!("{}", result.0);
    }

    #[test]
    fn test_white_case() {
        let patch: ImagePatch = ImagePatch::repeat(1.0 - 1e-6);
        println!("patch: {}", patch);
        let result = hashkey::<u8>(&patch);
        println!("{:?}", result);
    }

    #[test]
    fn test_hashkey() {
        let reference = fs::File::open("reference/hash_reference.txt.gz").unwrap();
        let reference = BufReader::new(&reference);
        let mut decoder = GzDecoder::new(reference);
        let mut reference = String::new();
        decoder.read_to_string(&mut reference).unwrap();

        // Load the reference data
        for line in reference.lines() {
            let patch_str_index = line.find(']').unwrap() + 1;
            let (patch_str, line) = (
                line[..patch_str_index].to_owned(),
                line[patch_str_index + 1..].to_owned(),
            );
            let patch_str = patch_str.replace("[", "");
            let patch_str = patch_str.replace("]", "");
            let patch_str = patch_str.replace(" ", "");
            let mut patch: Vec<f_t> = patch_str
                .split(",")
                .map(|a| str::parse(a).unwrap())
                .collect();
            let patch = ImagePatch::from_row_slice(&patch);
            let reference_values: Vec<u8> =
                line.split(" ").map(|a| str::parse(a).unwrap()).collect();
            assert!(reference_values.len() == 3);
            let (a, s, c) = (
                reference_values[0],
                reference_values[1],
                reference_values[2],
            );
            let hash = hashkey::<u8>(&patch);

            println!("patch: {}", patch);

            println!("Hash: {:?}", hash);
            println!("Reference values: {:?}\n\n\n", reference_values);

            assert!(hash == (a, s, c));
        }
    }

    #[test]
    fn test_eigendecomposition() {
        // TODO: Add a more comprehensive set of tests.
        println!(
            "{:?}",
            eigendecomposition::<f64>(&nalgebra::Matrix2::new(1.0, 2.0, 2.0, 3.0))
        );
        println!(
            "{:?}",
            eigendecomposition::<f64>(&nalgebra::Matrix2::new(3.0, 5.0, 5.0, 2.0))
        );
        println!(
            "{:?}",
            eigendecomposition::<f64>(&nalgebra::Matrix2::new(5.0, 3.0, 3.0, 2.0))
        );
    }
}
