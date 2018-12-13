use byteorder::{ByteOrder, LittleEndian};
use constants::*;
use ndarray::prelude::*;
use std::fs;
use std::io::{Read, Write};

pub type FilterBank = (ArrayD<u8>, ArrayD<f32>);

fn transform_u32_to_array_of_u8(x: u32) -> [u8; 4] {
    let b1: u8 = ((x >> 24) & 0xff) as u8;
    let b2: u8 = ((x >> 16) & 0xff) as u8;
    let b3: u8 = ((x >> 8) & 0xff) as u8;
    let b4: u8 = (x & 0xff) as u8;
    return [b1, b2, b3, b4];
}

pub fn write_filter(filename: &str, filterbank: &ArrayD<f32>) {
    let mut filter_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(filename)
        .expect(&format!("Unable to create {}", filename));

    {
        let mut write_u32 = |data: usize| {
            let mut buf = [0; 4];
            LittleEndian::write_u32(&mut buf, data as u32);
            filter_file.write(&buf).unwrap();
        };

        write_u32(Q_ANGLE);
        write_u32(Q_STRENGTH);
        write_u32(Q_COHERENCE);
        write_u32(R_2);
        write_u32(PATCH_VECTOR_SIZE);
    }

    for (angle, strength, coherence, pixel_type) in
        iproduct!((0..Q_ANGLE), (0..Q_STRENGTH), (0..Q_COHERENCE), (0..R_2))
    {
        for i in 0..PATCH_VECTOR_SIZE {
            let mut buf = [0; 4];
            LittleEndian::write_f32(
                &mut buf,
                filterbank[[angle, strength, coherence, pixel_type, i]],
            );
            filter_file.write(&buf).unwrap();
        }
    }
}

pub fn read_filter(filename: &str) -> FilterBank {
    let reference =
        fs::File::open(filename).expect(&format!("Unable to read filterbank {}", filename));
    let mut file_bytes = reference.bytes();

    let mut filter = ArrayD::zeros(IxDyn(&[
        Q_ANGLE,
        Q_STRENGTH,
        Q_COHERENCE,
        R_2,
        PATCH_VECTOR_SIZE,
    ]));

    let mut bounds = ArrayD::zeros(IxDyn(&[Q_ANGLE, Q_STRENGTH, Q_COHERENCE, R_2, 2]));

    let mut read_four_bytes = || {
        let bytes: Vec<u8> = file_bytes.by_ref().take(4).map(|a| a.unwrap()).collect();
        bytes
    };

    let q_angle: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let q_strength: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let q_coherence: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let r_2: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;
    let patch_vector_size: usize = LittleEndian::read_u32(&read_four_bytes()) as usize;

    assert!(q_angle == Q_ANGLE);
    assert!(q_strength == Q_STRENGTH);
    assert!(q_coherence == Q_COHERENCE);
    assert!(r_2 == R_2);
    assert!(patch_vector_size == PATCH_VECTOR_SIZE);

    for (angle, strength, coherence, pixel_type) in
        iproduct!((0..q_angle), (0..q_strength), (0..q_coherence), (0..r_2))
    {
        let mut float_data: Vec<FloatType> = Vec::new();
        for _ in 0..PATCH_VECTOR_SIZE {
            /*filter[[angle, strength, coherence, pixel_type, i]] =
            LittleEndian::read_f32(&read_four_bytes()) as FloatType;*/
            float_data.push(LittleEndian::read_f32(&read_four_bytes()) as FloatType);
            //println!("{}", filter[[angle, strength, coherence, pixel_type, i]]);
        }
        let min = float_data
            .iter()
            .fold(1e10 as FloatType, |a, b| FloatType::min(a, *b));
        let max = float_data
            .iter()
            .fold(-1e10 as FloatType, |a, b| FloatType::max(a, *b));
        let span = max - min;
        for i in 0..PATCH_VECTOR_SIZE {
            filter[[angle, strength, coherence, pixel_type, i]] =
                (((float_data[i] - min) / span) * 255.0) as u8;
            let p = filter[[angle, strength, coherence, pixel_type, i]];
        }

        bounds[[angle, strength, coherence, pixel_type, 0]] = min;
        bounds[[angle, strength, coherence, pixel_type, 1]] = max;
    }

    (filter, bounds)
}

pub fn apply_filter(
    filter_bank: &FilterBank,
    index: (usize, usize, usize, usize),
    patch: &PatchVector,
) -> FloatType {
    let (angle, strength, coherence, pixel_type) = index;
    let filter = filter_bank
        .0
        .slice(s![angle, strength, coherence, pixel_type, ..]);
    let bounds = filter_bank
        .1
        .slice(s![angle, strength, coherence, pixel_type, ..]);
    let min = bounds[0];
    let span = bounds[1] - bounds[0];
    let filter: Vec<FloatType> = filter
        .as_slice()
        .unwrap()
        .iter()
        .map(|a| ((*a as FloatType + 0.5) / 255.0) * span + min)
        .collect::<Vec<FloatType>>();
    let filter: PatchVector = PatchVector::from_column_slice(&filter);
    filter.dot(patch)
}

#[cfg(test)]
mod tests {
    use constants::*;
    use filters::*;
    use ndarray::prelude::*;

    #[test]
    fn test_filter_load() {
        let filter = read_filter("filters/filterbank");

        for i in 0..PATCH_VECTOR_SIZE {
            println!("{}", filter.0[[3, 2, 1, 2, i]]);
        }
    }
}
