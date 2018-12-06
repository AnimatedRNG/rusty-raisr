use glium::glutin;

use image;
/*use palette;
use palette::Pixel;*/
use std::fs::File;
use std::io::Read;
use std::mem;

use constants::R;

const LOCAL_SIZE: (usize, usize) = (8, 8);

fn inference_gpu(filename: &str, output: &str) {
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

    let mut raisr_shader_file = File::open("shaders/raisr.glsl").expect("Can't find raisr.glsl!");
    let mut raisr_shader = String::new();
    raisr_shader_file.read_to_string(&mut raisr_shader).unwrap();

    let raisr_shader = raisr_shader.replace(
        "#define LOCAL_SIZE_X",
        &format!("#define LOCAL_SIZE_X {}", LOCAL_SIZE.0),
    );
    let raisr_shader = raisr_shader.replace(
        "#define LOCAL_SIZE_Y",
        &format!("#define LOCAL_SIZE_Y {}", LOCAL_SIZE.1),
    );

    let program = glium::program::ComputeShader::from_source(&display, &raisr_shader).unwrap();

    program.execute(
        uniform! { lr_image: input_texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest).minify_filter(glium::uniforms::MinifySamplerFilter::Nearest).wrap_function(glium::uniforms::SamplerWrapFunction::Clamp),
                   hr_image: output_texture.image_unit().set_format(glium::uniforms::ImageUnitFormat::RGBA8).set_access(glium::uniforms::ImageUnitAccess::ReadWrite).set_level(0),
                   R: R as u32,
        },
        (image_dimensions.0 * R as u32) / LOCAL_SIZE.0 as u32,
        (image_dimensions.1 * R as u32) / LOCAL_SIZE.1 as u32,
        1,
    );

    display.memory_barrier(
        glium::backend::MemoryBarrier::SHADER_IMAGE_ACCESS_BARRIER
            | glium::backend::MemoryBarrier::TEXTURE_UPDATE_BARRIER,
    );

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
        inference_gpu("test/Fallout.png", "output/Fallout_gpu_inferred.png");
    }
}
