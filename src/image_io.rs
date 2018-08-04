use image;
use image::GenericImage;
use nalgebra::DMatrix;
use std::fs;

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

pub fn write_image(filename: &str, img: &(DMatrix<f32>, DMatrix<f32>, DMatrix<f32>)) {
    if fs::metadata(filename).is_ok() {
        fs::remove_file(filename).unwrap();
    }

    let to_u8 = |f| (f as f32 * 255.0) as u8;

    let ref mut buffer =
        image::ImageBuffer::from_fn(img.0.shape().0 as u32, img.0.shape().1 as u32, |x, y| {
            let (x, y) = (x as usize, y as usize);
            let pixel = [
                to_u8(img.0[(x, y)]),
                to_u8(img.1[(x, y)]),
                to_u8(img.2[(x, y)]),
            ];
            image::Rgb(pixel)
        });

    buffer.save(filename).unwrap();
}
