use constants::FloatType;
use glium::texture::RawImage2d;
use image;
use image::GenericImage;
use nalgebra::DMatrix;
use std::borrow::Cow;
use std::fs;

pub type RGBFloatImage = (DMatrix<FloatType>, DMatrix<FloatType>, DMatrix<FloatType>);
pub type RGBUnsignedImage = (DMatrix<u8>, DMatrix<u8>, DMatrix<u8>);
pub struct SizedRawImage2d<'a> {
    pub img: RawImage2d<'a, u8>,
    pub size: (u32, u32),
}

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

impl<'a> ReadableImage<SizedRawImage2d<'a>> for SizedRawImage2d<'a> {
    fn read_image(filename: &str) -> SizedRawImage2d<'a> {
        let image = image::open(filename)
            .expect(&format!("Unable to read image {}", filename))
            .to_rgba();

        let image_dimensions = image.dimensions();

        SizedRawImage2d {
            img: glium::texture::RawImage2d::from_raw_rgba(image.into_raw(), image_dimensions),
            size: image_dimensions,
        }
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

impl<'a> WriteableImage<SizedRawImage2d<'a>> for SizedRawImage2d<'a> {
    fn write_image(filename: &str, image: &SizedRawImage2d<'a>) {
        let tex_data = image.img.data.to_vec();
        //let tex_data: &[palette::LinSrgb<u8>] = Pixel::from_raw_slice(&tex_data);
        //let tex_data = Pixel::into_raw_slice(&tex_data).to_vec();

        let tex_img = image::ImageBuffer::from_raw(image.size.0, image.size.1, tex_data).unwrap();
        let tex_img = image::DynamicImage::ImageRgba8(tex_img);
        tex_img.save(format!("{}", filename)).unwrap();
    }
}

// Would use From, but I don't own either type
pub fn convert_to_glium<'a>(img: &RGBUnsignedImage) -> SizedRawImage2d<'a> {
    let mut packed: Vec<u8> = Vec::new();
    let dims = img.0.shape();

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            packed.push(img.0[(i, j)]);
            packed.push(img.1[(i, j)]);
            packed.push(img.2[(i, j)]);
        }
    }

    let dims = (dims.1 as u32, dims.0 as u32);

    SizedRawImage2d {
        img: RawImage2d::from_raw_rgb(packed, dims),
        size: dims,
    }
}
