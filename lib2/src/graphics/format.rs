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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ImageUsage {
    SampledImage,
    StorageImage,
    ColorAttachment,
    DepthStencil,
    BlitSrc,
    BlitDst,
    TransferSrc,
    TransferDst,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PresentMode {
    Immediate,
    Fifo,
    Mailbox,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ColorSpace {
    SRGBNonlinear,
    ExtendedSRGBNonlinear,
    ExtendedSRGBLinear,
    Bt709Linear,
    Bt2020Linear,
    DCIP3Nonlinear,
    DisplayP3Nonlinear,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ImageFormat {
    BC1_RGBA_SRGB_BLOCK,
    BC1_RGBA_UNORM_BLOCK,
    BC2_SRGB_BLOCK,
    BC2_UNORM_BLOCK,
    BC3_SRGB_BLOCK,
    BC3_UNORM_BLOCK,
    BC4_SNORM_BLOCK,
    BC4_UNORM_BLOCK,
    BC5_SNORM_BLOCK,
    BC5_UNORM_BLOCK,
    BC6H_SFLOAT_BLOCK,
    BC6H_UFLOAT_BLOCK,
    BC7_SRGB_BLOCK,
    BC7_UNORM_BLOCK,
    D16_UNORM,
    D24_UNORM_S8_UINT,
    D32_SFLOAT,
    D32_SFLOAT_S8_UINT,
    R16_SFLOAT,
    R16_SINT,
    R16_SNORM,
    R16_UINT,
    R16_UNORM,
    R16G16_SFLOAT,
    R16G16_SINT,
    R16G16_SNORM,
    R16G16_UINT,
    R16G16_UNORM,
    R16G16B16A16_SFLOAT,
    R16G16B16A16_SINT,
    R16G16B16A16_SNORM,
    R16G16B16A16_UINT,
    R16G16B16A16_UNORM,
    R32_SFLOAT,
    R32_SINT,
    R32_UINT,
    R32G32_SFLOAT,
    R32G32_SINT,
    R32G32_UINT,
    R32G32B32A32_SFLOAT,
    R32G32B32A32_SINT,
    R32G32B32A32_UINT,
    R8_SINT,
    R8_SNORM,
    R8_SRGB,
    R8_UINT,
    R8_UNORM,
    R8G8_SINT,
    R8G8_SNORM,
    R8G8_SRGB,
    R8G8_UINT,
    R8G8_UNORM,
    R8G8B8A8_SINT,
    R8G8B8A8_SNORM,
    R8G8B8A8_SRGB,
    R8G8B8A8_UINT,
    R8G8B8A8_UNORM,
    R10G10B10A2_UNORM,
    R11G11B10_UFLOAT,
}
