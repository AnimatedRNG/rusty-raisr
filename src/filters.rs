use byteorder::{ByteOrder, LittleEndian};
use constants::*;
use itertools;
use ndarray::prelude::*;
use std::fs;
use std::io::Read;

pub type FilterBank = ArrayD<f_t>;

pub fn read_filter(filename: &str) -> FilterBank {
    let reference = fs::File::open(filename).unwrap();
    let mut file_bytes = reference.bytes();

    let mut filter = FilterBank::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
    ]));

    let mut read_four_bytes = || {
        let bytes: Vec<u8> = file_bytes.by_ref().take(4).map(|a| a.unwrap()).collect();
        bytes
    };

    let q_angle: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let q_strength: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let q_coherence: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let r_2: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;

    assert!(q_angle == Q_ANGLE);
    assert!(q_strength == Q_STRENGTH);
    assert!(q_coherence == Q_COHERENCE);
    assert!(r_2 == R_2);

    for (angle, strength, coherence, pixel_type) in
        iproduct!((0..q_angle), (0..q_strength), (0..q_coherence), (0..r_2))
    {
        for i in 0..PATCH_VECTOR_SIZE {
            filter[[angle, strength, coherence, pixel_type, i]] =
                LittleEndian::read_f32(&read_four_bytes()) as f_t;
            println!("{}", filter[[angle, strength, coherence, pixel_type, i]]);
        }
    }

    for i in 0..PATCH_VECTOR_SIZE {
        println!("{}", filter[[3, 1, 1, 0, i]]);
    }

    filter
}

#[cfg(test)]
mod tests {
    use constants::*;
    use filters::*;
    use ndarray::prelude::*;

    #[test]
    fn test_filter_load() {
        read_filter("filters/filterbank");
    }
}
