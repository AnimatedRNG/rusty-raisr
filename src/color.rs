use constants::FloatType;
use nalgebra;

pub fn to_ycbcr(
    red: &nalgebra::DMatrix<FloatType>,
    green: &nalgebra::DMatrix<FloatType>,
    blue: &nalgebra::DMatrix<FloatType>,
) -> (
    nalgebra::DMatrix<FloatType>,
    nalgebra::DMatrix<FloatType>,
    nalgebra::DMatrix<FloatType>,
) {
    let y = 0.299 * red + 0.587 * green + 0.114 * blue;
    let cb = -0.168736 * red - 0.331264 * green + 0.500 * blue;
    let cr = 0.5 * red - 0.418688 * green - 0.081312 * blue;

    (y, cb, cr)
}

pub fn from_ycbcr(
    y: &nalgebra::DMatrix<FloatType>,
    cb: &nalgebra::DMatrix<FloatType>,
    cr: &nalgebra::DMatrix<FloatType>,
) -> (
    nalgebra::DMatrix<FloatType>,
    nalgebra::DMatrix<FloatType>,
    nalgebra::DMatrix<FloatType>,
) {
    let red = y + 1.402 * cr;
    let green = y - 0.344136 * cb - 0.714136 * cr;
    let blue = y + 1.772 * cb;

    (red, green, blue)
}
