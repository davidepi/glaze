use super::format::{ImageFormat, ImageUsage};

/// Group together features and extensions required, based on the GPU purpose.
#[derive(Debug, Copy, Clone)]
pub enum FeatureSet {
    /// Rendering to a surface (no raytracing)
    Present,
    /// Converting assets
    Convert,
}

impl FeatureSet {
    /// Formats required by the FeatureSet. These are application specific.
    ///
    /// Returns a tuple (format, usage, optimal tiling).
    pub fn required_formats(&self) -> Vec<(ImageFormat, ImageUsage, bool)> {
        match self {
            FeatureSet::Present => vec![(
                ImageFormat::D32_SFLOAT_S8_UINT,
                ImageUsage::DepthStencil,
                true,
            )],
            FeatureSet::Convert => vec![],
        }
    }
}

pub trait Device {}
