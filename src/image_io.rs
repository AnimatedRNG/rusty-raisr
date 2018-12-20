use constants::FloatType;
use image;
use image::GenericImage;
use nalgebra::DMatrix;
use std::fs;

pub type RGBFloatImage = (DMatrix<FloatType>, DMatrix<FloatType>, DMatrix<FloatType>);
pub type RGBUnsignedImage = (DMatrix<u8>, DMatrix<u8>, DMatrix<u8>);

pub trait ReadableImage<T> {
    fn read_image(filename: &str) -> T;
}

pub trait WriteableImage<T> {
    fn write_image(filename: &str, img: &T);
}

impl ReadableImage<RGBFloatImage> for RGBFloatImage {
    fn read_image(filename: &str) -> RGBFloatImage {
        let img = image::open(filename).expect(&format!("Unable to read image {}", filename));
        let dims = img.dimensions();
        let (mut red, mut green, mut blue): RGBFloatImage = (
            DMatrix::<FloatType>::zeros(dims.0 as usize, dims.1 as usize),
            DMatrix::<FloatType>::zeros(dims.0 as usize, dims.1 as usize),
            DMatrix::<FloatType>::zeros(dims.0 as usize, dims.1 as usize),
        );

        for (x_i, y_i, pixel) in img.pixels() {
            let (x, y) = (x_i as usize, y_i as usize);
            red[(x, y)] = pixel[0] as FloatType / 255.0;
            green[(x, y)] = pixel[1] as FloatType / 255.0;
            blue[(x, y)] = pixel[2] as FloatType / 255.0;
        }

        (red.transpose(), green.transpose(), blue.transpose())
    }
}

impl WriteableImage<RGBFloatImage> for RGBFloatImage {
    fn write_image(filename: &str, img: &RGBFloatImage) {
        if fs::metadata(filename).is_ok() {
            fs::remove_file(filename).unwrap();
        }

        let to_u8 = |f| (FloatType::min(FloatType::max(f, 0.0), 1.0) * 255.0) as u8;

        let ref mut buffer =
            image::ImageBuffer::from_fn(img.0.shape().1 as u32, img.0.shape().0 as u32, |x, y| {
                let (x, y) = (x as usize, y as usize);
                let pixel = [
                    to_u8(img.0[(y, x)]),
                    to_u8(img.1[(y, x)]),
                    to_u8(img.2[(y, x)]),
                ];
                image::Rgb(pixel)
            });

        buffer.save(filename).unwrap();
    }
}

impl WriteableImage<RGBUnsignedImage> for RGBUnsignedImage {
    fn write_image(filename: &str, img: &RGBUnsignedImage) {
        if fs::metadata(filename).is_ok() {
            fs::remove_file(filename).unwrap();
        }

        let ref mut buffer =
            image::ImageBuffer::from_fn(img.0.shape().1 as u32, img.0.shape().0 as u32, |x, y| {
                let (x, y) = (x as usize, y as usize);
                let pixel = [img.0[(y, x)], img.1[(y, x)], img.2[(y, x)]];
                image::Rgb(pixel)
            });

        buffer.save(filename).unwrap();
    }
}
