use nalgebra;

pub type FloatType = f32;

pub const PATCH_SIZE: usize = 11;
pub const GRADIENT_SIZE: usize = 9;
pub const GRADIENT_VECTOR_SIZE: usize = GRADIENT_SIZE * GRADIENT_SIZE;
pub const PATCH_VECTOR_SIZE: usize = PATCH_SIZE * PATCH_SIZE;
pub const Q_ANGLE: usize = 24;
pub const Q_COHERENCE: usize = 3;
pub const Q_STRENGTH: usize = 3;
pub const R: usize = 2;
pub const R_2: usize = R * R;

pub const MAX_BLOCK_SIZE: usize = GRADIENT_SIZE;
pub const MARGIN: usize = MAX_BLOCK_SIZE / 2;
pub const PATCH_MARGIN: usize = PATCH_SIZE / 2;
pub const GRADIENT_MARGIN: usize = GRADIENT_SIZE / 2;

pub const TILE_SIZE: usize = 64;

pub type PatchSizeType = nalgebra::U11;
pub type GradientSizeType = nalgebra::U9;
pub type GradientVectorSizeType = nalgebra::U81;

pub type ImagePatch = nalgebra::Matrix<
    FloatType,
    PatchSizeType,
    PatchSizeType,
    nalgebra::MatrixArray<FloatType, PatchSizeType, PatchSizeType>,
>;
pub type GradientBlock = nalgebra::Matrix<
    FloatType,
    GradientSizeType,
    GradientSizeType,
    nalgebra::MatrixArray<FloatType, GradientSizeType, GradientSizeType>,
>;
pub type GradientVector = nalgebra::Vector<
    FloatType,
    GradientVectorSizeType,
    nalgebra::MatrixArray<FloatType, GradientVectorSizeType, nalgebra::U1>,
>;

pub const WEIGHTS: [FloatType; GRADIENT_SIZE * GRADIENT_SIZE] = [
    0.00076345, 0.00183141, 0.00342153, 0.0049783, 0.00564116, 0.0049783, 0.00342153, 0.00183141,
    0.00076345, 0.00183141, 0.00439334, 0.00820783, 0.01194233, 0.01353243, 0.01194233, 0.00820783,
    0.00439334, 0.00183141, 0.00342153, 0.00820783, 0.01533425, 0.0223112, 0.0252819, 0.0223112,
    0.01533425, 0.00820783, 0.00342153, 0.0049783, 0.01194233, 0.0223112, 0.03246261, 0.03678495,
    0.03246261, 0.0223112, 0.01194233, 0.0049783, 0.00564116, 0.01353243, 0.0252819, 0.03678495,
    0.04168281, 0.03678495, 0.0252819, 0.01353243, 0.00564116, 0.0049783, 0.01194233, 0.0223112,
    0.03246261, 0.03678495, 0.03246261, 0.0223112, 0.01194233, 0.0049783, 0.00342153, 0.00820783,
    0.01533425, 0.0223112, 0.0252819, 0.0223112, 0.01533425, 0.00820783, 0.00342153, 0.00183141,
    0.00439334, 0.00820783, 0.01194233, 0.01353243, 0.01194233, 0.00820783, 0.00439334, 0.00183141,
    0.00076345, 0.00183141, 0.00342153, 0.0049783, 0.00564116, 0.0049783, 0.00342153, 0.00183141,
    0.00076345,
];
