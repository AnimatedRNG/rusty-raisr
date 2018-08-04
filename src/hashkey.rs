use constants::*;
use nalgebra;
use std::cmp::max;
use std::f64::consts::PI;

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

fn eigendecomposition(
    m: &nalgebra::Matrix2<f_t>,
) -> (nalgebra::Vector2<f_t>, nalgebra::Matrix2<f_t>) {
    let (a, b, c, d) = (m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
    if b * c <= 1e-20 {
        (
            nalgebra::Vector2::new(a, d),
            nalgebra::Matrix2::new(1.0, 0.0, 0.0, 1.0),
        )
    } else {
        let tr = a + d;
        let det = a * d - b * c;
        let s = ((tr / 2.0).powf(2.0) - det).sqrt();
        let lamb = nalgebra::Vector2::new(tr / 2.0 + s, tr / 2.0 - s);

        let ss = (((a - d) / 2.0).powi(2) + b * c).max(0.0).sqrt();
        let mut ev: nalgebra::Matrix2<f_t> = if a - d < 0.0 {
            nalgebra::Matrix2::new(c, (a - d) / 2.0 - ss, -(a - d) / 2.0 + ss, b)
        } else {
            nalgebra::Matrix2::new((a - d) / 2.0 + ss, c, b, -(a - d) / 2.0 - ss)
        };

        let n1 = (ev[(0, 0)].powi(2) + ev[(0, 1)].powi(2)).sqrt();
        ev[0] /= n1;

        let n2 = (ev[(1, 0)].powi(2) + ev[(1, 1)].powi(2)).sqrt();
        ev[1] /= n2;

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

    type WType = nalgebra::Matrix<
        f_t,
        GradientVectorSizeType,
        GradientVectorSizeType,
        nalgebra::MatrixArray<f_t, GradientVectorSizeType, GradientVectorSizeType>,
    >;

    let weights_diag = WType::from_diagonal(&GradientVector::from_row_slice(&weights));

    let gtwg = g.transpose() * weights_diag * g;
    let (w, v) = eigendecomposition(&gtwg);

    let theta = f_t::atan2(v[(1, 0)], v[(0, 0)]);
    let theta = if theta < 0.0 {
        theta + PI as f_t
    } else {
        theta
    };
    let lambda = w[0];

    let sqrtlambda1 = w[0].sqrt();
    let sqrtlambda2 = w[1].sqrt();
    let u = if sqrtlambda1 + sqrtlambda2 == 0.0 {
        0.0
    } else {
        (sqrtlambda1 - sqrtlambda2) / (sqrtlambda1 + sqrtlambda2)
    };

    let angle = (theta / PI as f_t * Q_ANGLE as f_t).floor();
    let strength: T = if lambda < 0.0001 {
        0
    } else if lambda > 0.001 {
        2
    } else {
        1
    }.into();
    let coherence: T = if u < 0.25 {
        0
    } else if u > 0.5 {
        2
    } else {
        1
    }.into();

    let angle: T = if angle > 23.0 {
        23
    } else if angle < 0.0 {
        0
    } else {
        angle as u8
    }.into();

    (angle, strength, coherence)
}

#[cfg(test)]
mod tests {
    use constants::*;
    use hashkey::*;
    use nalgebra;

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
        let mut patch = get_test_patch();
        println!("Patch: {}", patch);
        println!("Sobel: {}", sobel_filter(&patch).0);
    }

    #[test]
    fn test_hashkey() {
        let mut patch = get_test_patch();
        println!("Hash: {:?}", hashkey::<u8>(&patch));
    }

    #[test]
    fn test_eigendecomposition() {
        println!(
            "{:?}",
            eigendecomposition(&nalgebra::Matrix2::new(1.0, 2.0, 2.0, 3.0))
        );
        println!(
            "{:?}",
            eigendecomposition(&nalgebra::Matrix2::new(3.0, 5.0, 5.0, 2.0))
        );
        println!(
            "{:?}",
            eigendecomposition(&nalgebra::Matrix2::new(5.0, 3.0, 3.0, 2.0))
        );
    }
}
