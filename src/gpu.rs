use glium::backend::Facade;
use glium::glutin;

use image;
use std::borrow::Cow;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::time::{Duration, Instant};

use constants::*;
use filters::{read_filter, FilterBank};

const BLOCK_DIM: u32 = 8;
const NUM_TRIALS: usize = 100;
const ALIGNED_PATCH_VEC_SIZE: usize = 128;
const ALIGNED_PATCH_ELEMENT_SIZE: usize = 4;

fn filterbank_to_texture(
    display: &glium::backend::Facade,
    filterbank: &FilterBank,
) -> (
    glium::texture::buffer_texture::BufferTexture<(u8, u8, u8, u8)>,
    glium::texture::Texture1d,
) {
    let mut filterbank_vec = Vec::new();
    let mut bounds_vec: Vec<FloatType> = Vec::new();

    for angle in 0..Q_ANGLE {
        for strength in 0..Q_STRENGTH {
            for coherence in 0..Q_COHERENCE {
                for pixel_type in 0..R_2 {
                    let end_index: FloatType =
                        PATCH_VECTOR_SIZE as FloatType / ALIGNED_PATCH_ELEMENT_SIZE as FloatType;
                    let end_index: usize = end_index.ceil() as usize;
                    let elem_size: usize = ALIGNED_PATCH_ELEMENT_SIZE;
                    for i in 0..end_index {
                        let get_ind = |a: usize| {
                            let ind = elem_size * i + a;
                            if ind < PATCH_VECTOR_SIZE {
                                filterbank.0[[angle, strength, coherence, pixel_type, ind]]
                            } else {
                                0
                            }
                        };
                        let entry = (get_ind(0), get_ind(1), get_ind(2), get_ind(3));
                        filterbank_vec.push(entry);
                    }

                    // Don't need to build this up explcitly, but this is here for readibility
                    bounds_vec.push(filterbank.1[[angle, strength, coherence, pixel_type, 0]]);
                    bounds_vec.push(filterbank.1[[angle, strength, coherence, pixel_type, 1]]);

                    for _ in end_index..ALIGNED_PATCH_VEC_SIZE / ALIGNED_PATCH_ELEMENT_SIZE {
                        filterbank_vec.push((0, 0, 0, 0));
                    }
                }
            }
        }
    }

    /*let raw_filterbank = glium::texture::RawImage1d {
        data: Cow::from(&filterbank_vec),
        width: (Q_ANGLE * Q_STRENGTH * Q_COHERENCE * R_2 * ALIGNED_PATCH_VEC_SIZE
            / ALIGNED_PATCH_ELEMENT_SIZE) as u32,
        format: glium::texture::ClientFormat::U8U8U8U8,
    };*/
    let raw_bounds = glium::texture::RawImage1d {
        data: Cow::from(&bounds_vec),
        width: (Q_ANGLE * Q_STRENGTH * Q_COHERENCE * R_2) as u32,
        format: glium::texture::ClientFormat::F32F32,
    };
    (
        glium::texture::buffer_texture::BufferTexture::immutable(
            display,
            &filterbank_vec,
            glium::texture::buffer_texture::BufferTextureType::Unsigned,
        )
        .unwrap(),
        glium::texture::Texture1d::with_format(
            display,
            raw_bounds,
            glium::texture::UncompressedFloatFormat::F32F32,
            glium::texture::MipmapsOption::NoMipmap,
        )
        .unwrap(),
    )
}

fn inference_gpu(filename: &str, filterbank: &str, output: &str) {
    let events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new().with_visibility(false);
    let context = glutin::ContextBuilder::new();
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let image = image::open(filename)
        .expect(&format!("Unable to read image {}", filename))
        .to_rgba();

    let image_dimensions = image.dimensions();
    let image =
        glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);

    //let input_texture = glium::texture::SrgbTexture2d::new(&display, image).unwrap();
    let input_texture = glium::texture::Texture2d::new(&display, image).unwrap();
    let output_texture = glium::texture::Texture2d::empty(
        &display,
        image_dimensions.0 * R as u32,
        image_dimensions.1 * R as u32,
    )
    .unwrap();

    let (filterbank_texture, bounds_texture) =
        filterbank_to_texture(&display, &read_filter(filterbank));

    let mut raisr_shader_file = File::open("shaders/raisr.glsl").expect("Can't find raisr.glsl!");
    let mut raisr_shader = String::new();
    raisr_shader_file.read_to_string(&mut raisr_shader).unwrap();

    let mut gradient_gather_file =
        File::open("shaders/gradient_gather.glsl").expect("Can't find gradient_gather.glsl!");
    let mut gradient_gather_shader = String::new();
    gradient_gather_file
        .read_to_string(&mut gradient_gather_shader)
        .unwrap();

    let raisr_shader = raisr_shader.replace(
        "#define BLOCK_DIM",
        &format!("#define BLOCK_DIM {}", BLOCK_DIM),
    );

    let raisr_shader = raisr_shader.replace("#define GRADIENT_GATHER", &gradient_gather_shader);

    let program = glium::program::ComputeShader::from_source(&display, &raisr_shader).unwrap();

    let now = Instant::now();

    for _ in 0..NUM_TRIALS {
        program.execute(
            uniform! {
                lr_image: input_texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest).minify_filter(glium::uniforms::MinifySamplerFilter::Nearest).wrap_function(glium::uniforms::SamplerWrapFunction::Clamp),
                filterbank: &filterbank_texture,
                bounds: bounds_texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest).minify_filter(glium::uniforms::MinifySamplerFilter::Nearest).wrap_function(glium::uniforms::SamplerWrapFunction::Clamp),
                hr_image: output_texture.image_unit().set_format(glium::uniforms::ImageUnitFormat::RGBA8).set_access(glium::uniforms::ImageUnitAccess::ReadWrite).set_level(0),
                R: R as u32,
        },
        (image_dimensions.0 * R as u32) / BLOCK_DIM,
        (image_dimensions.1 * R as u32) / BLOCK_DIM,
        1,
    );

        display.memory_barrier(
            glium::backend::MemoryBarrier::SHADER_IMAGE_ACCESS_BARRIER
                | glium::backend::MemoryBarrier::TEXTURE_UPDATE_BARRIER,
        );
    }

    // Really force everything to flush right here
    display.get_context().finish();

    let elapsed = now.elapsed();

    let elapsed: f32 = elapsed.as_secs() as f32 * 1000.0 + elapsed.subsec_millis() as f32;
    println!("Time elapsed: {}", elapsed / NUM_TRIALS as f32);

    let tex_img: glium::texture::RawImage2d<u8> = output_texture.read();
    let tex_data: Vec<u8> = tex_img.data.into_owned();
    //let tex_data: &[palette::LinSrgb<u8>] = Pixel::from_raw_slice(&tex_data);
    //let tex_data = Pixel::into_raw_slice(&tex_data).to_vec();
    let tex_img = image::ImageBuffer::from_raw(tex_img.width, tex_img.height, tex_data).unwrap();
    let tex_img = image::DynamicImage::ImageRgba8(tex_img).flipv();
    tex_img.save(format!("{}", output)).unwrap();
}

#[cfg(test)]
mod tests {
    use gpu::*;

    #[test]
    fn test_gpu_inference() {
        /*inference_gpu(
            "test/Fallout.png",
            "filters/filterbank",
            "output/Fallout_gpu_inferred.png",
        );*/
        inference_gpu(
            "test/veronica.png",
            "filters/filterbank",
            "output/veronica_gpu_inferred.png",
        );
        /*inference_gpu(
            "test/MirrorsEdge_1.jpg",
            "filters/filterbank",
            "output/MirrorsEdge_1_inferred.png",
        );*/
    }
}
