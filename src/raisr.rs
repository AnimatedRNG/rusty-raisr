use std::collections::BTreeMap;
use std::fs::{create_dir, read_dir};
use std::path::Path;

use cgls::cgls;
use constants::*;
use filters::{apply_filter, write_filter, FilterBank};
use hashkey::hashkey;
use image_io::{RGBFloatImage, RGBUnsignedImage, ReadableImage, WriteableImage};
use itertools::Itertools;
use nalgebra;
use nalgebra::DMatrix;
use ndarray::prelude::*;
use rayon::prelude::*;

pub type FilterImage = (DMatrix<u8>, DMatrix<u8>, DMatrix<u8>);

fn get_pixel_clamped(img: &DMatrix<FloatType>, coord: (i64, i64)) -> FloatType {
    let coord = (
        (coord.0.max(0) as usize).min(img.shape().0 - 1),
        (coord.1.max(0) as usize).min(img.shape().1 - 1),
    );

    img[coord]
}

fn grab_patch(img: &DMatrix<FloatType>, center: (usize, usize)) -> ImagePatch {
    let mut patch: ImagePatch = ImagePatch::zeros();

    let center = (center.0 as i64, center.1 as i64);
    let margin = PATCH_MARGIN as i64;
    let patch_size = PATCH_SIZE as i64;

    for x in 0..patch_size {
        for y in 0..patch_size {
            patch[(x as usize, y as usize)] =
                get_pixel_clamped(img, (center.0 + x - margin, center.1 + y - margin));
        }
    }

    patch
}

fn lerp(s: FloatType, e: FloatType, t: FloatType) -> FloatType {
    s + (e - s) * t
}

fn blerp(
    block: &nalgebra::Matrix2<FloatType>,
    b_interp: &nalgebra::Vector2<FloatType>,
) -> FloatType {
    lerp(
        lerp(block[(0, 0)], block[(0, 1)], b_interp[0]),
        lerp(block[(1, 0)], block[(1, 1)], b_interp[0]),
        b_interp[1],
    )
}

pub fn bilinear_filter(img: &DMatrix<FloatType>, ideal_size: (usize, usize)) -> DMatrix<FloatType> {
    let dx = img.shape().0 as FloatType / ideal_size.0 as FloatType;
    let dy = img.shape().1 as FloatType / ideal_size.1 as FloatType;

    let mut output_image = DMatrix::zeros(ideal_size.0, ideal_size.1);

    for i in 0..ideal_size.0 {
        let x = i as FloatType * dx;
        for j in 0..ideal_size.1 {
            let y = j as FloatType * dy;

            let i_x = x as i64;
            let i_y = y as i64;

            let f_x = x - x.floor();
            let f_y = y - y.floor();

            output_image[(i, j)] = blerp(
                &nalgebra::Matrix2::new(
                    get_pixel_clamped(img, (i_x, i_y)),
                    get_pixel_clamped(img, (i_x + 1, i_y)),
                    get_pixel_clamped(img, (i_x, i_y + 1)),
                    get_pixel_clamped(img, (i_x + 1, i_y + 1)),
                ),
                &nalgebra::Vector2::new(f_x, f_y),
            );
        }
    }

    output_image
}

// TODO: Refactor using parallel_image_op
pub fn create_filter_image(hr_y: &DMatrix<FloatType>) -> FilterImage {
    let dims = hr_y.shape();

    let ideal_size = (dims.0, dims.1);

    let results: Vec<Vec<((usize, usize), (u8, u8, u8))>> = (0..ideal_size.0)
        .into_iter()
        .map(|x: usize| {
            /*println!(
                "{}% complete",
                x as f32 / ideal_size.0 as f32 * 100.0 as f32
            );*/
            (0..ideal_size.1)
                .map(|y: usize| {
                    let patch = grab_patch(&hr_y, (x, y));
                    let key = hashkey(&patch);
                    ((x, y), key)
                })
                .collect()
        })
        .collect();

    let mut final_angle: DMatrix<u8> = DMatrix::zeros(ideal_size.0, ideal_size.1);
    let mut final_strength: DMatrix<u8> = DMatrix::zeros(ideal_size.0, ideal_size.1);
    let mut final_coherence: DMatrix<u8> = DMatrix::zeros(ideal_size.0, ideal_size.1);

    results.iter().foreach(|row| {
        row.iter().foreach(|((x, y), value)| {
            final_angle[(*x, *y)] = value.0;
            final_strength[(*x, *y)] = value.1;
            final_coherence[(*x, *y)] = value.2;
        })
    });

    (final_angle, final_strength, final_coherence)
}

// TODO: DRY
pub fn image_op<T: Send + Sync>(
    dims: (usize, usize),
    op: &(Fn(usize, usize) -> T + Send + Sync),
) -> Vec<Vec<((usize, usize), T)>> {
    (0..dims.0)
        .into_iter()
        .map(|x: usize| {
            (0..dims.1)
                .map(|y: usize| ((x, y), op(x, y)))
                .collect::<Vec<((usize, usize), T)>>()
        })
        .collect()
}

pub fn parallel_image_op<T: Send + Sync>(
    dims: (usize, usize),
    op: &(Fn(usize, usize) -> T + Send + Sync),
) -> Vec<Vec<((usize, usize), T)>> {
    (0..dims.0)
        .into_par_iter()
        .map(|x: usize| {
            (0..dims.1)
                .map(|y: usize| ((x, y), op(x, y)))
                .collect::<Vec<((usize, usize), T)>>()
        })
        .collect()
}

pub struct TrainImage {
    hr_y: DMatrix<FloatType>,
    hash_img: FilterImage,
    y: DMatrix<FloatType>,
}

fn check_cache(name: &str) -> Option<FilterImage> {
    if Path::new("cache").exists() {
        let hashimg_filename = Path::new("cache").join(name);
        if hashimg_filename.exists() {
            let filter_img_raw = RGBFloatImage::read_image(hashimg_filename.to_str().unwrap());

            Some((
                (filter_img_raw.0 * Q_ANGLE as f32).map(|f| f as u8),
                (filter_img_raw.1 * Q_STRENGTH as f32).map(|f| f as u8),
                (filter_img_raw.2 * Q_COHERENCE as f32).map(|f| f as u8),
            ))
        } else {
            None
        }
    } else {
        None
    }
}

fn cache_img(name: &str, image_contents: &FilterImage) {
    let path = Path::new("cache");
    if !path.exists() {
        create_dir("cache").expect("Unable to create cache directory!");
    }
    let path = Path::new("cache").join(name);
    let path = path.to_str().unwrap();
    println!("Writing to {}", path);

    RGBUnsignedImage::write_image(
        path,
        &debug_filter_image(&image_contents.0, &image_contents.1, &image_contents.2),
    );
}

pub fn training_generator(hr_folder: &str, lr_folder: &str, output_file: &str) {
    let hr_names: BTreeMap<String, String> = read_dir(hr_folder)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            (
                Path::new(path.file_name().unwrap())
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned(),
                path.to_str().unwrap().to_owned(),
            )
        })
        .collect();
    let lr_names: BTreeMap<String, String> = read_dir(lr_folder)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            (
                Path::new(path.file_name().unwrap())
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned(),
                path.to_str().unwrap().to_owned(),
            )
        })
        .collect();

    let Q: ArrayD<FloatType> = ArrayD::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
        PATCH_VECTOR_SIZE,
    ]));

    let V: ArrayD<FloatType> = ArrayD::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
    ]));

    // Parallel image load, hash, and send for training
    let tmp: Vec<(ArrayD<FloatType>, ArrayD<FloatType>)> = hr_names
        .par_iter()
        .map(move |hr_entry| {
            if lr_names.contains_key(hr_entry.0) {
                println!(
                    "Found training pair: {} {}",
                    hr_entry.1, lr_names[hr_entry.0]
                );
                let hr_img = RGBFloatImage::read_image(&hr_entry.1);
                let lr_img = RGBFloatImage::read_image(&lr_names[hr_entry.0]);
                println!("Finished reading {}", hr_entry.0);

                let (label_hr_y, _, _) = hr_img;
                let (lr_y, _, _) = lr_img;
                let hr_dims = (label_hr_y.shape().0, label_hr_y.shape().1);
                assert!(hr_dims == (lr_y.shape().0 * R, lr_y.shape().1 * R));

                let cheap_hr_y =
                    bilinear_filter(&lr_y, (label_hr_y.shape().0, label_hr_y.shape().1));
                println!("Bilinear filtered {}", hr_entry.0);

                let hash_img_name = format!("{}_hashimg.png", hr_entry.0);
                let hash_img = match check_cache(&hash_img_name) {
                    None => {
                        println!("{} is not in cache; computing...", hash_img_name);
                        let hash_img = create_filter_image(&cheap_hr_y);
                        cache_img(&hash_img_name, &hash_img);
                        hash_img
                    }
                    Some(hash_img) => hash_img,
                };
                println!("Hashed {}", hr_entry.0);

                Some(train_batch(TrainImage {
                    hr_y: cheap_hr_y,
                    hash_img: hash_img,
                    y: label_hr_y,
                }))
            } else {
                None
            }
        })
        .filter(|a| a.is_some())
        .map(|a| a.unwrap())
        .collect();

    let (Q, V) = tmp.iter().fold((Q, V), |old, new| {
        let new = new.clone();
        (old.0 + new.0, old.1 + new.1)
    });

    let mut filter = ArrayD::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
    ]));

    for (angle, strength, coherence, pixel_type) in
        iproduct!((0..Q_ANGLE), (0..Q_STRENGTH), (0..Q_COHERENCE), (0..R_2))
    {
        let q = Q.slice(s![angle, strength, coherence, pixel_type, .., ..]);
        let v = V.slice(s![angle, strength, coherence, pixel_type, ..]);

        let q = PatchSqMatrix::from_row_slice(q.as_slice().unwrap());
        let v = PatchVector::from_column_slice(v.as_slice().unwrap());

        let h = cgls(&q, &v);

        for i in 0..PATCH_VECTOR_SIZE {
            filter[[angle, strength, coherence, pixel_type, i]] = h[i];
        }
    }

    write_filter(output_file, &filter);
}

pub fn train_batch(received: TrainImage) -> (ArrayD<FloatType>, ArrayD<FloatType>) {
    let mut Q: ArrayD<FloatType> = ArrayD::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
        PATCH_VECTOR_SIZE,
    ]));

    let mut V: ArrayD<FloatType> = ArrayD::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
    ]));

    let (hr_y, hash_img, y_img) = (received.hr_y, received.hash_img, received.y);

    let ideal_size = (hr_y.shape().0, hr_y.shape().1);
    let margin = PATCH_MARGIN;

    for x in 0..ideal_size.0 {
        for y in 0..ideal_size.1 {
            let angle = hash_img.0[(x, y)] as usize;
            let strength = hash_img.1[(x, y)] as usize;
            let coherence = hash_img.2[(x, y)] as usize;
            let pixel_type: usize = ((x - margin) % R) * R + (y - margin) % R;

            let patch: PatchVector =
                PatchVector::from_row_slice(grab_patch(&hr_y, (x, y)).transpose().as_slice());

            let pixel_hr = y_img[(x, y)];

            let mut a_t_a = patch * patch.transpose();

            //let a_t_b = patch.transpose() * pixel_hr;
            //let a_t_b: PatchVector = a_t_b.transpose();
            let mut a_t_b = patch * pixel_hr;

            let mut q_slice = Q.slice_mut(s![angle, strength, coherence, pixel_type, .., ..]);
            let mut v_slice = V.slice_mut(s![angle, strength, coherence, pixel_type, ..]);

            let a_t_a = ArrayViewMut::from_shape(
                (PATCH_VECTOR_SIZE, PATCH_VECTOR_SIZE),
                a_t_a.as_mut_slice(),
            )
            .unwrap();
            let a_b_a = ArrayViewMut::from_shape(PATCH_VECTOR_SIZE, a_t_b.as_mut_slice()).unwrap();

            q_slice += &a_t_a;
            v_slice += &a_b_a;
        }
        println!(
            "{}% done with image",
            (x as f32 / ideal_size.0 as f32) * 100.0
        );
    }
    println!("Trained on image");

    (Q, V)
}

pub fn inference(
    hr_y: &DMatrix<FloatType>,
    filter_image: &FilterImage,
    filter_bank: &FilterBank,
) -> DMatrix<FloatType> {
    let ideal_size = (hr_y.shape().0, hr_y.shape().1);
    let margin = PATCH_MARGIN;

    let mut upscaled: DMatrix<FloatType> = DMatrix::zeros(ideal_size.0, ideal_size.1);

    let results = parallel_image_op(ideal_size, &|x, y| {
        let angle = filter_image.0[(x, y)] as usize;
        let strength = filter_image.1[(x, y)] as usize;
        let coherence = filter_image.2[(x, y)] as usize;
        let pixel_type = ((x - margin) % R) * R + (y - margin) % R;
        /*let filter = filter_bank.slice(s![angle, strength, coherence, pixel_type, ..]);
        let filter: &[FloatType] = filter.as_slice().unwrap();
        let filter: PatchVector = PatchVector::from_column_slice(filter);
        let patch: PatchVector =
            PatchVector::from_row_slice(grab_patch(&hr_y, (x, y)).transpose().as_slice());

        FloatType::min(FloatType::max(patch.dot(&filter), 1e-6), 1.0 - 1e-6)*/
        let patch: PatchVector =
            PatchVector::from_row_slice(grab_patch(&hr_y, (x, y)).transpose().as_slice());
        apply_filter(
            filter_bank,
            (angle, strength, coherence, pixel_type),
            &patch,
        )
    });

    results.iter().foreach(|row| {
        row.iter().foreach(|((x, y), value)| {
            upscaled[(*x, *y)] = *value;
        })
    });

    upscaled
}

fn to_float(m: &DMatrix<u8>, normalize: bool) -> DMatrix<FloatType> {
    m.map(|a| a as FloatType / (if normalize { 255.0 } else { 1.0 }))
}

fn to_byte(m: &DMatrix<FloatType>) -> DMatrix<u8> {
    m.map(|a| (a * 255.0) as u8)
}

pub fn debug_filter_image(
    angle: &DMatrix<u8>,
    strength: &DMatrix<u8>,
    coherence: &DMatrix<u8>,
) -> (DMatrix<u8>, DMatrix<u8>, DMatrix<u8>) {
    (
        to_byte(&(to_float(angle, false) / Q_ANGLE as FloatType)),
        to_byte(&(to_float(strength, false) / Q_STRENGTH as FloatType)),
        to_byte(&(to_float(coherence, false) / Q_COHERENCE as FloatType)),
    )
}

#[cfg(test)]
mod tests {
    use color::{from_ycbcr, to_ycbcr};
    use constants::*;
    use filters::*;
    use image_io::{RGBFloatImage, RGBUnsignedImage, ReadableImage, WriteableImage};
    use raisr::*;
    use std::thread;

    #[test]
    fn test_raisr() {
        //test_create_filter_image();
        //test_bilinear_filter_image();
        //test_patch();
        //test_hash_image();
        //test_apply_filter();
        test_inference();
        //test_training();
    }

    fn test_create_filter_image() {
        let (r, g, b) = RGBFloatImage::read_image("test/veronica.jpg");
        let (y, cb, cr) = to_ycbcr(&r, &g, &b);
        let (r, g, b) = from_ycbcr(&y, &cb, &cr);
        RGBFloatImage::write_image("output/veronica_result.png", &(r, g, b));
    }

    fn test_bilinear_filter_image() {
        let (r, g, b) = RGBFloatImage::read_image("test/veronica.jpg");
        let (y, cb, cr) = to_ycbcr(&r, &g, &b);

        let lr_dims = (r.shape().0 * 2, r.shape().1 * 2);

        let y = bilinear_filter(&y, (lr_dims.0, lr_dims.1));
        let cb = bilinear_filter(&cb, (lr_dims.0, lr_dims.1));
        let cr = bilinear_filter(&cr, (lr_dims.0, lr_dims.1));
        let (r, g, b) = from_ycbcr(&y, &cb, &cr);
        RGBFloatImage::write_image("output/veronica_result.png", &(r, g, b));
    }

    fn test_patch() {
        let img = RGBFloatImage::read_image("test/Fallout.png");
        let (r, g, b) = img;
        let (y, _, _) = to_ycbcr(&r, &g, &b);
        let dims = y.shape();
        let ideal_size = (dims.0 * R, dims.1 * R);
        let hr_y = bilinear_filter(&y, ideal_size);
        let patch = grab_patch(&hr_y, (100, 240));

        println!("Patch: {}", patch);
    }

    fn test_hash_image() {
        let img = RGBFloatImage::read_image("test/Fallout.png");

        let (r, g, b) = img;
        let (y, _, _) = to_ycbcr(&r, &g, &b);

        let dims = y.shape();

        let ideal_size = (dims.0 * R, dims.1 * R);

        let hr_y = bilinear_filter(&y, ideal_size);

        let filter_image = create_filter_image(&hr_y);
        let debug = debug_filter_image(&filter_image.0, &filter_image.1, &filter_image.2);

        RGBUnsignedImage::write_image("output/Fallout_hashimg.png", &debug);
    }

    fn test_apply_filter() {
        let filter_img_raw: RGBFloatImage = RGBFloatImage::read_image("test/Fallout_filters.png");
        let img = RGBFloatImage::read_image("test/Fallout.png");

        let filter_img: FilterImage = (
            (filter_img_raw.0 * Q_ANGLE as f32).map(|f| f as u8),
            (filter_img_raw.1 * Q_STRENGTH as f32).map(|f| f as u8),
            (filter_img_raw.2 * Q_COHERENCE as f32).map(|f| f as u8),
        );

        let (r, g, b) = img;
        let (y, cb, cr) = to_ycbcr(&r, &g, &b);

        let dims = y.shape();

        let ideal_size = (dims.0 * R, dims.1 * R);

        let hr_y = bilinear_filter(&y, ideal_size);
        let hr_cb = bilinear_filter(&cb, ideal_size);
        let hr_cr = bilinear_filter(&cr, ideal_size);

        let debug = debug_filter_image(&filter_img.0, &filter_img.1, &filter_img.2);
        let inferred_y = inference(&hr_y, &filter_img, &read_filter("filters/filterbank"));
        let new_rgb = from_ycbcr(&inferred_y, &hr_cb, &hr_cr);
        RGBFloatImage::write_image("output/Fallout_inferred.png", &new_rgb);
        RGBUnsignedImage::write_image("output/Fallout_hashimg.png", &debug);
    }

    fn test_inference() {
        let img = RGBFloatImage::read_image("test/Fallout.png");

        let (r, g, b) = img;
        let (y, cb, cr) = to_ycbcr(&r, &g, &b);

        let dims = y.shape();

        let ideal_size = (dims.0 * R, dims.1 * R);

        let hr_y = bilinear_filter(&y, ideal_size);
        let hr_cb = bilinear_filter(&cb, ideal_size);
        let hr_cr = bilinear_filter(&cr, ideal_size);

        let filter_image = create_filter_image(&hr_y);
        let debug = debug_filter_image(&filter_image.0, &filter_image.1, &filter_image.2);
        let inferred_y = inference(&hr_y, &filter_image, &read_filter("filters/filterbank"));
        let new_rgb = from_ycbcr(&inferred_y, &hr_cb, &hr_cr);
        RGBFloatImage::write_image("output/Fallout_inferred.png", &new_rgb);
        RGBUnsignedImage::write_image("output/Fallout_hashimg.png", &debug);
    }

    fn test_training() {
        test_training_generator();
    }

    fn test_training_generator() {
        training_generator("train/hr", "train/lr", "filters/new_filterbank");
    }
}
