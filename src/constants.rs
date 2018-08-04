use nalgebra;

pub const PATCH_SIZE: usize = 11;
pub const GRADIENT_SIZE: usize = 9;
pub const GRADIENT_VECTOR_SIZE: usize = GRADIENT_SIZE * GRADIENT_SIZE;
pub const Q_ANGLE: usize = 24;
pub const Q_COHERENCE: usize = 3;
pub const Q_STRENGTH: usize = 3;
pub const R: usize = 2;

pub const MAX_BLOCK_SIZE: usize = GRADIENT_SIZE;
pub const MARGIN: usize = MAX_BLOCK_SIZE / 2;
pub const PATCH_MARGIN: usize = PATCH_SIZE / 2;
pub const GRADIENT_MARGIN: usize = GRADIENT_SIZE / 2;

pub type PatchSizeType = nalgebra::U11;
pub type GradientSizeType = nalgebra::U9;
pub type GradientVectorSizeType = nalgebra::U81;

pub type ImagePatch = nalgebra::Matrix<
    f32,
    PatchSizeType,
    PatchSizeType,
    nalgebra::MatrixArray<f32, PatchSizeType, PatchSizeType>,
>;
pub type GradientBlock = nalgebra::Matrix<
    f32,
    GradientSizeType,
    GradientSizeType,
    nalgebra::MatrixArray<f32, GradientSizeType, GradientSizeType>,
>;
pub type GradientVector = nalgebra::Vector<
    f32,
    GradientVectorSizeType,
    nalgebra::MatrixArray<f32, GradientVectorSizeType, nalgebra::U1>,
>;
