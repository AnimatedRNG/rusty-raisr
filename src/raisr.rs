use constants::*;
use filters::FilterBank;
use hashkey::hashkey;
use itertools::Itertools;
use nalgebra;
use nalgebra::DMatrix;
use ndarray::prelude::*;
use rayon::prelude::*;

pub type FilterImage = (DMatrix<u8>, DMatrix<u8>, DMatrix<u8>);

fn get_pixel_clamped(img: &DMatrix<f_t>, coord: (i64, i64)) -> f_t {
    let coord = (
        (coord.0.max(0) as usize).min(img.shape().0 - 1),
        (coord.1.max(0) as usize).min(img.shape().1 - 1),
    );

    img[coord]
}

fn grab_patch(img: &DMatrix<f_t>, center: (usize, usize)) -> ImagePatch {
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

fn lerp(s: f_t, e: f_t, t: f_t) -> f_t {
    s + (e - s) * t
}

fn blerp(block: &nalgebra::Matrix2<f_t>, b_interp: &nalgebra::Vector2<f_t>) -> f_t {
    lerp(
        lerp(block[(0, 0)], block[(0, 1)], b_interp[0]),
        lerp(block[(1, 0)], block[(1, 1)], b_interp[0]),
        b_interp[1],
    )
}

fn bilinear_filter(img: &DMatrix<f_t>, ideal_size: (usize, usize)) -> DMatrix<f_t> {
    let dx = img.shape().0 as f_t / ideal_size.0 as f_t;
    let dy = img.shape().1 as f_t / ideal_size.1 as f_t;

    let mut output_image = DMatrix::zeros(ideal_size.0, ideal_size.1);

    for i in 0..ideal_size.0 {
        let x = i as f_t * dx;
        for j in 0..ideal_size.1 {
            let y = j as f_t * dy;

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

fn create_filter_image(hr_y: &DMatrix<f_t>) -> FilterImage {
    let dims = hr_y.shape();

    let ideal_size = (dims.0, dims.1);

    let results: Vec<Vec<((usize, usize), (u8, u8, u8))>> = (0..ideal_size.0)
        .into_par_iter()
        .map(|x: usize| {
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

fn inference(
    hr_y: &DMatrix<f_t>,
    filter_image: &FilterImage,
    filter_bank: &FilterBank,
) -> DMatrix<f_t> {
    let ideal_size = (hr_y.shape().0, hr_y.shape().1);
    let margin = PATCH_MARGIN;

    let mut upscaled: DMatrix<f_t> = DMatrix::zeros(ideal_size.0, ideal_size.1);

    let results: Vec<Vec<((usize, usize), f_t)>> = (0..ideal_size.0)
        .into_par_iter()
        .map(|x: usize| {
            (0..ideal_size.1)
                .map(|y: usize| {
                    let angle = filter_image.0[(x, y)] as usize;
                    let strength = filter_image.1[(x, y)] as usize;
                    let coherence = filter_image.2[(x, y)] as usize;
                    let pixel_type = ((x - margin) % R) * R + (y - margin) % R;
                    let filter: ArrayView1<f_t> =
                        filter_bank.slice(s![angle, strength, coherence, pixel_type, ..]);
                    let patch = grab_patch(&hr_y, (x, y)).transpose();
                    let patch_slice: &[f_t] = patch.as_slice();
                    let patch_nd = ArrayView::from_shape((121,), patch_slice).unwrap();

                    (
                        (x, y),
                        f_t::min(f_t::max(patch_nd.dot(&filter), 1e-6), 1.0 - 1e-6),
                    )
                })
                .collect()
        })
        .collect();

    results.iter().foreach(|row| {
        row.iter().foreach(|((x, y), value)| {
            upscaled[(*x, *y)] = *value;
        })
    });

    upscaled
}

#[cfg(test)]
mod tests {
    use color::{from_ycbcr, to_ycbcr};
    use constants::*;
    use filters::*;
    use image_io::{read_image, write_image, write_image_u8};
    use raisr::*;

    fn to_float(m: &DMatrix<u8>, normalize: bool) -> DMatrix<f_t> {
        m.map(|a| a as f_t / (if normalize { 255.0 } else { 1.0 }))
    }

    fn to_byte(m: &DMatrix<f_t>) -> DMatrix<u8> {
        m.map(|a| (a * 255.0) as u8)
    }

    fn debug_filter_image(
        angle: &DMatrix<u8>,
        strength: &DMatrix<u8>,
        coherence: &DMatrix<u8>,
    ) -> (DMatrix<u8>, DMatrix<u8>, DMatrix<u8>) {
        (
            to_byte(&(to_float(angle, false) / Q_ANGLE as f_t)),
            to_byte(&(to_float(strength, false) / Q_STRENGTH as f_t)),
            to_byte(&(to_float(coherence, false) / Q_COHERENCE as f_t)),
        )
    }

    #[test]
    fn test_raisr() {
        test_create_filter_image();
        test_bilinear_filter_image();
        test_patch();
        test_hash_image();
        test_apply_filter();
        test_inference();
    }

    fn test_create_filter_image() {
        let (r, g, b) = read_image("test/veronica.jpg");
        let (y, cb, cr) = to_ycbcr(&r, &g, &b);
        let (r, g, b) = from_ycbcr(&y, &cb, &cr);
        write_image("output/veronica_result.png", &(r, g, b));
    }

    fn test_bilinear_filter_image() {
        let (r, g, b) = read_image("test/veronica.jpg");
        let (y, cb, cr) = to_ycbcr(&r, &g, &b);

        let lr_dims = (r.shape().0 * 2, r.shape().1 * 2);

        let y = bilinear_filter(&y, (lr_dims.0, lr_dims.1));
        let cb = bilinear_filter(&cb, (lr_dims.0, lr_dims.1));
        let cr = bilinear_filter(&cr, (lr_dims.0, lr_dims.1));
        let (r, g, b) = from_ycbcr(&y, &cb, &cr);
        write_image("output/veronica_result.png", &(r, g, b));
    }

    fn test_patch() {
        let img = read_image("test/Fallout.png");
        let (r, g, b) = img;
        let (y, _, _) = to_ycbcr(&r, &g, &b);
        let dims = y.shape();
        let ideal_size = (dims.0 * R, dims.1 * R);
        let hr_y = bilinear_filter(&y, ideal_size);
        let patch = grab_patch(&hr_y, (100, 240));

        println!("Patch: {}", patch);
    }

    fn test_hash_image() {
        let img = read_image("test/Fallout.png");

        let (r, g, b) = img;
        let (y, _, _) = to_ycbcr(&r, &g, &b);

        let dims = y.shape();

        let ideal_size = (dims.0 * R, dims.1 * R);

        let hr_y = bilinear_filter(&y, ideal_size);

        let filter_image = create_filter_image(&hr_y);
        let debug = debug_filter_image(&filter_image.0, &filter_image.1, &filter_image.2);

        write_image_u8("output/Fallout_hashimg.png", &debug);
    }

    fn test_apply_filter() {
        let filter_img_raw = read_image("test/Fallout_filters.png");
        let img = read_image("test/Fallout.png");

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
        write_image("output/Fallout_inferred.png", &new_rgb);
        write_image_u8("output/Fallout_hashimg.png", &debug);
    }

    fn test_inference() {
        let img = read_image("test/Fallout.png");

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
        write_image("output/Fallout_inferred.png", &new_rgb);
        write_image_u8("output/Fallout_hashimg.png", &debug);
    }
}
