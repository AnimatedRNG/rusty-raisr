use constants::*;
use nalgebra;

pub fn cgls(A: &PatchSqMatrix, b: &PatchVector) -> PatchVector {
    //let (height, width) = A.shape();

    let mut x: PatchVector = nalgebra::zero();

    let mut A = *A;

    loop {
        let sum_a: f32 = A.iter().sum();

        if sum_a < 100.0 {
            break;
        }

        if A.determinant() < 1.0 {
            A = A + PatchSqMatrix::identity() * sum_a * 0.000000005;
        } else {
            x = A.try_inverse().unwrap() * b;
            break;
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgls_simple() {
        let A = PatchSqMatrix::identity();
        let mut b = PatchVector::zeros();
        b[0] = 1.0;

        let x = cgls(&A, &b);
        assert!((x - b).iter().sum::<f32>().abs() < 1e-6);
    }

    #[test]
    fn test_cgls_example() {
        let mut A = PatchSqMatrix::identity();
        let mut b = PatchVector::zeros();

        A[(0, 1)] = 1.0;
        A[(1, 0)] = 1.0;

        b[0] = 1.0;
        b[1] = 1.0;

        let mut xpected = PatchVector::zeros();
        xpected[0] = 0.5;
        xpected[1] = 0.5;

        let x = cgls(&A, &b);
        assert!((x - xpected).iter().sum::<f32>().abs() < 1e-1);
    }
}
