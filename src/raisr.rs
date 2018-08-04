use color::{from_ycbcr, to_ycbcr};
use constants::*;
use hashkey::hashkey;
use image_io::read_image;
use nalgebra;
use nalgebra::DMatrix;

fn get_pixel_clamped(img: &DMatrix<f32>, coord: (i64, i64)) -> f32 {
    let coord = (
        (coord.0.max(0) as usize).min(img.shape().0 - 1),
        (coord.1.max(0) as usize).min(img.shape().1 - 1),
    );

    img[coord]
}

fn grab_patch(img: &DMatrix<f32>, center: (usize, usize)) -> ImagePatch {
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

fn lerp(s: f32, e: f32, t: f32) -> f32 {
    s + (e - s) * t
}

fn blerp(block: &nalgebra::Matrix2<f32>, b_interp: &nalgebra::Vector2<f32>) -> f32 {
    lerp(
        lerp(block[(0, 0)], block[(0, 1)], b_interp[0]),
        lerp(block[(1, 0)], block[(1, 1)], b_interp[0]),
        b_interp[1],
    )
}

fn bilinear_filter(img: &DMatrix<f32>, ideal_size: (usize, usize)) -> DMatrix<f32> {
    let dx = img.shape().0 as f32 / ideal_size.0 as f32;
    let dy = img.shape().1 as f32 / ideal_size.1 as f32;

    let mut output_image = DMatrix::zeros(ideal_size.0, ideal_size.1);

    for i in 0..ideal_size.0 {
        let x = i as f32 * dx;
        for j in 0..ideal_size.1 {
            let y = j as f32 * dy;

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

fn create_filter_image(img: &(DMatrix<f32>, DMatrix<f32>, DMatrix<f32>)) {}

#[cfg(test)]
mod tests {
    use color::{from_ycbcr, to_ycbcr};
    use constants::*;
    use image_io::{read_image, write_image};
    use nalgebra;
    use raisr::*;

    #[test]
    fn test_raisr() {
        //test_create_filter_image();
        //test_bilinear_filter_image();
        test_patch();
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
        let mut img: DMatrix<f32> = nalgebra::DMatrix::zeros(50, 50);
        for i in 0..50 {
            for j in 0..50 {
                img[(i, j)] = (f32::sin(i as f32) + f32::cos(i as f32)) as f32;
            }
        }

        println!("Patch: {}", grab_patch(&img, (2, 2)));
    }
}
