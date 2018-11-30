extern crate clap;
extern crate rusty_raisr;

use clap::{App, Arg};
use rusty_raisr::color::{from_ycbcr, to_ycbcr};
use rusty_raisr::constants::*;
use rusty_raisr::filters::read_filter;
use rusty_raisr::image_io::{read_image, write_image, write_image_u8};
use rusty_raisr::raisr::{bilinear_filter, create_filter_image, debug_filter_image, inference};

fn main() {
    let matches = App::new("rusty-raisr")
        .version("0.1")
        .author("Srinivas Kaza <kaza@mit.edu>")
        .about("rusty-raisr is a Rust implementation or RAISR")
        .arg(
            Arg::with_name("filterbank")
                .help("Filterbank to use")
                .value_name("FILE")
                .short("f")
                .default_value("filters/filterbank")
                .required(false)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("debug")
                .help("Name of filter image to write")
                .value_name("FILE")
                .short("d")
                .required(false)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("input")
                .help("Input image")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .help("Output image")
                .required(true)
                .takes_value(true),
        )
        .get_matches();

    let filterbank_name: String = matches.value_of("filterbank").unwrap().to_owned();

    let input_image_name: String = matches.value_of("input").unwrap().to_owned();
    let input_image = read_image(&input_image_name);

    let output_image_name: String = matches.value_of("output").unwrap().to_owned();

    let (r, g, b) = input_image;
    let (y, cb, cr) = to_ycbcr(&r, &g, &b);

    let dims = y.shape();

    let ideal_size = (dims.0 * R, dims.1 * R);

    let hr_y = bilinear_filter(&y, ideal_size);
    let hr_cb = bilinear_filter(&cb, ideal_size);
    let hr_cr = bilinear_filter(&cr, ideal_size);

    let filter_image = create_filter_image(&hr_y);
    let inferred_y = inference(&hr_y, &filter_image, &read_filter(&filterbank_name));
    let new_rgb = from_ycbcr(&inferred_y, &hr_cb, &hr_cr);
    write_image(&output_image_name, &new_rgb);

    match matches.value_of("debug") {
        Some(debug_file_name) => {
            let debug = debug_filter_image(&filter_image.0, &filter_image.1, &filter_image.2);
            write_image_u8(debug_file_name, &debug);
        }
        _ => {}
    };
}
