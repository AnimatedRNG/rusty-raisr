use image;
use image::GenericImage;
use nalgebra::DMatrix;

pub fn read_image(filename: &str) -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {
    let img = image::open(filename).unwrap();
    let dims = img.dimensions();
    let (mut red, mut green, mut blue) = (
        DMatrix::zeros(dims.0 as usize, dims.1 as usize),
        DMatrix::zeros(dims.0 as usize, dims.1 as usize),
        DMatrix::zeros(dims.0 as usize, dims.1 as usize),
    );

    for (x_i, y_i, pixel) in img.pixels() {
        let (x, y) = (x_i as usize, y_i as usize);
        red[(x, y)] = pixel[0] as f32 / 255.0;
        green[(x, y)] = pixel[1] as f32 / 255.0;
        blue[(x, y)] = pixel[2] as f32 / 255.0;
    }

    (red, green, blue)
}
