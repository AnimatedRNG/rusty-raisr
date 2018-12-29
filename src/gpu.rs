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
use image_io::{ReadableImage, SizedRawImage2d, WriteableImage};

const BLOCK_DIM: u32 = 8;
const NUM_TRIALS: usize = 100;
//const ALIGNED_PATCH_VEC_SIZE: usize = 132;
const ALIGNED_PATCH_ELEMENT_SIZE: usize = 4;

pub struct GLSLConfiguration {
    pub benchmark: bool,
    pub filter_unroll: bool,
    pub unroll_loops: bool,
    pub half_precision: bool,
}

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
                        PATCH_SIZE as FloatType / ALIGNED_PATCH_ELEMENT_SIZE as FloatType;
                    let end_index: usize = end_index.ceil() as usize;
                    let elem_size: usize = ALIGNED_PATCH_ELEMENT_SIZE;
                    for i in 0..PATCH_SIZE {
                        for j in 0..end_index {
                            let get_ind = |a: usize| {
                                let j = j * elem_size + a;
                                let ind = i * PATCH_SIZE + j;
                                if i < PATCH_SIZE && j < PATCH_SIZE {
                                    filterbank.0[[angle, strength, coherence, pixel_type, ind]]
                                } else {
                                    0
                                }
                            };
                            let entry = (get_ind(0), get_ind(1), get_ind(2), get_ind(3));
                            filterbank_vec.push(entry);
                        }
                    }

                    // Don't need to build this up explcitly, but this is here for readibility
                    bounds_vec.push(filterbank.1[[angle, strength, coherence, pixel_type, 0]]);
                    bounds_vec.push(filterbank.1[[angle, strength, coherence, pixel_type, 1]]);
                }
            }
        }
    }

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

pub fn inference_gpu<'a>(
    input_image: SizedRawImage2d,
    filterbank: &FilterBank,
    hash_image: Option<SizedRawImage2d>,
    configuration: &GLSLConfiguration,
) -> SizedRawImage2d<'a> {
    let events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new().with_visibility(false);
    let context = glutin::ContextBuilder::new();
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let (image, image_dimensions) = (input_image.img, input_image.size);

    // TODO: Retrain on SRGB
    //let input_texture = glium::texture::SrgbTexture2d::new(&display, image).unwrap();
    let input_texture = glium::texture::Texture2d::new(&display, image).unwrap();
    let output_texture = glium::texture::Texture2d::empty(
        &display,
        image_dimensions.0 * R as u32,
        image_dimensions.1 * R as u32,
    )
    .unwrap();
    let hash_texture = match hash_image {
        None => None,
        Some(hash_image) => {
            glium::texture::unsigned_texture2d::UnsignedTexture2d::new(&display, hash_image.img)
                .ok()
        }
    };

    let (filterbank_texture, bounds_texture) = filterbank_to_texture(&display, filterbank);

    let mut raisr_shader_file = File::open("shaders/raisr.glsl").expect("Can't find raisr.glsl!");
    let mut raisr_shader = String::new();
    raisr_shader_file.read_to_string(&mut raisr_shader).unwrap();

    let mut gradient_gather_file =
        File::open("shaders/gradient_gather.glsl").expect("Can't find gradient_gather.glsl!");
    let mut gradient_gather_shader = String::new();
    gradient_gather_file
        .read_to_string(&mut gradient_gather_shader)
        .unwrap();

    let mut accumulate_filter_file =
        File::open("shaders/filter.glsl").expect("Can't find filter.glsl!");
    let mut accumulate_filter_shader = String::new();
    accumulate_filter_file
        .read_to_string(&mut accumulate_filter_shader)
        .unwrap();

    let raisr_shader = raisr_shader.replace(
        "#define BLOCK_DIM",
        &format!("#define BLOCK_DIM {}", BLOCK_DIM),
    );

    let raisr_shader = raisr_shader.replace("#define GRADIENT_GATHER", &gradient_gather_shader);
    let mut raisr_shader = raisr_shader.replace("#define ACCUMULATE_FILTER", &accumulate_filter_shader);

    {
        let mut set_shader_toggle = |name: &str, value: bool| {
            let value_int = if value { 1 } else { 0 };
            raisr_shader = raisr_shader.replace(
                &format!("#define {} 0", name),
                &format!("#define {} {}", name, value_int),
            )
        };

        set_shader_toggle("HASH_IMAGE_ENABLED", hash_texture.is_some());
        set_shader_toggle(
            "FILTER_UNROLL_ENABLED",
            configuration.filter_unroll,
        );
        set_shader_toggle("UNROLL_LOOPS", configuration.unroll_loops);
        set_shader_toggle("FLOAT_16_ENABLED", configuration.half_precision)
    }
    //println!("Shader: {}", raisr_shader);

    let program = glium::program::ComputeShader::from_source(&display, &raisr_shader).unwrap();

    let now = Instant::now();

    struct GaussianWeights {
        weights: [f32],
    }

    implement_buffer_content!(GaussianWeights);
    implement_uniform_block!(GaussianWeights, weights);

    let mut weight_buffer =
        glium::uniforms::UniformBuffer::<GaussianWeights>::empty_unsized_immutable(
            &display,
            GRADIENT_SIZE * GRADIENT_SIZE * 4,
        )
        .unwrap();

    {
        let mut weight_buffer = weight_buffer.map();

        // Awful; refactor sometime
        for (index, val) in weight_buffer.weights.iter_mut().enumerate() {
            *val = WEIGHTS[index];
        }
    }

    let lr_sampler = input_texture
        .sampled()
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
        .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
        .wrap_function(glium::uniforms::SamplerWrapFunction::Clamp);
    let bounds_sampler = bounds_texture
        .sampled()
        .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
        .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
        .wrap_function(glium::uniforms::SamplerWrapFunction::Clamp);

    let num_trials = if configuration.benchmark {
        NUM_TRIALS
    } else {
        1
    };

    for _ in 0..num_trials {
        let hr_image_object = output_texture
            .image_unit()
            .set_format(glium::uniforms::ImageUnitFormat::RGBA8)
            .set_access(glium::uniforms::ImageUnitAccess::ReadWrite)
            .set_level(0);

        match &hash_texture {
            None => program.execute(
                uniform! {
                    lr_image: lr_sampler,
                    filterbank: &filterbank_texture,
                    bounds: bounds_sampler,
                    hr_image: hr_image_object,
                    hash_image_enabled: false,
                    R: R as u32,
                    GaussianWeights: &*weight_buffer,
                },
                ((image_dimensions.0 as FloatType * R as FloatType) / BLOCK_DIM as FloatType).ceil()
                    as u32,
                ((image_dimensions.1 as FloatType * R as FloatType) / BLOCK_DIM as FloatType).ceil()
                    as u32,
                1,
            ),
            Some(hash_texture) => program.execute(
                uniform! {
                    lr_image: lr_sampler,
                    filterbank: &filterbank_texture,
                    bounds: bounds_sampler,
                    hash_image: hash_texture,
                    hr_image: hr_image_object,
                    R: R as u32,
                    GaussianWeights: &weight_buffer,
                },
                ((image_dimensions.0 as FloatType * R as FloatType) / BLOCK_DIM as FloatType).ceil()
                    as u32,
                ((image_dimensions.1 as FloatType * R as FloatType) / BLOCK_DIM as FloatType).ceil()
                    as u32,
                1,
            ),
        }

        display.memory_barrier(
            glium::backend::MemoryBarrier::SHADER_IMAGE_ACCESS_BARRIER
                | glium::backend::MemoryBarrier::TEXTURE_UPDATE_BARRIER,
        );
    }

    // Really force everything to flush right here
    display.get_context().finish();

    let elapsed = now.elapsed();

    let elapsed: f32 = elapsed.as_secs() as f32 * 1000.0 + elapsed.subsec_millis() as f32;

    if configuration.benchmark {
        let elapsed = elapsed / NUM_TRIALS as f32;
        let pixels_per_second = (output_texture.width() * output_texture.height()) as f32 / (elapsed / 1000.0);
        println!("Time elapsed: {}", elapsed);
        println!(
            "MP/s (destination space): {}",
            pixels_per_second / 1000000.0
        )
    }

    let tex_img: glium::texture::RawImage2d<u8> = output_texture.read();

    SizedRawImage2d {
        img: tex_img,
        size: (output_texture.width(), output_texture.height()),
    }
}

#[cfg(test)]
mod tests {
    use filters::read_filter;
    use gpu::*;
    use image_io::{
        convert_to_glium, RGBFloatImage, RGBUnsignedImage, ReadableImage, SizedRawImage2d,
        WriteableImage,
    };

    fn perform_inference(
        input_image_name: &str,
        output_image_name: &str,
        hash_img_name: Option<String>,
    ) {
        let config = GLSLConfiguration {
            benchmark: true,
            filter_unroll: true,
            unroll_loops: false,
            half_precision: true,
        };
        let filterbank = read_filter("filters/filterbank");
        let input_image = SizedRawImage2d::read_image(input_image_name);
        let hash_image = match hash_img_name {
            None => None,
            Some(hash_img_name) => {
                let filter_img_raw: RGBFloatImage = RGBFloatImage::read_image(&hash_img_name);

                Some(convert_to_glium(&(
                    (filter_img_raw.0 * Q_ANGLE as f32).map(|f| f as u8),
                    (filter_img_raw.1 * Q_STRENGTH as f32).map(|f| f as u8),
                    (filter_img_raw.2 * Q_COHERENCE as f32).map(|f| f as u8),
                )))
            }
        };
        SizedRawImage2d::write_image(
            output_image_name,
            &inference_gpu(input_image, &filterbank, hash_image, &config),
        );
    }

    #[test]
    fn test_gpu_inference() {
        perform_inference("test/Fallout.png", "output/Fallout_gpu_inferred.png", None);
        perform_inference(
            "test/veronica.png",
            "output/veronica_gpu_inferred.png",
            None,
        );
        perform_inference("test/full_hd.png", "output/full_hd_inferred.png", None);
    }

    #[test]
    fn test_gpu_apply_filter() {
        perform_inference(
            "test/Fallout.png",
            "output/Fallout_gpu_inferred.png",
            Some("output/Fallout_hashimg.png".to_owned()),
        );
        perform_inference(
            "test/veronica.png",
            "output/veronica_gpu_inferred.png",
            Some("output/veronica_hashimg.png".to_owned()),
        );
        perform_inference("test/full_hd.png",
                          "output/full_hd_inferred.png",
                          Some("output/full_hd_hashimg.png".to_owned())
        );
    }
}
