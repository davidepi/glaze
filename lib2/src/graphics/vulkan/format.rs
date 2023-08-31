use crate::graphics::format::{ImageFormat, ImageUsage, PresentMode};
use ash::vk;

impl ImageUsage {
    /// Returns the corresponding [vk::FormatFeatureFlags]
    pub fn to_vk_format(self) -> vk::FormatFeatureFlags {
        match self {
            ImageUsage::SampledImage => vk::FormatFeatureFlags::SAMPLED_IMAGE,
            ImageUsage::StorageImage => vk::FormatFeatureFlags::STORAGE_IMAGE,
            ImageUsage::ColorAttachment => vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            ImageUsage::DepthStencil => vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageUsage::BlitSrc => vk::FormatFeatureFlags::BLIT_SRC,
            ImageUsage::BlitDst => vk::FormatFeatureFlags::BLIT_DST,
            ImageUsage::TransferSrc => vk::FormatFeatureFlags::TRANSFER_SRC,
            ImageUsage::TransferDst => vk::FormatFeatureFlags::TRANSFER_DST,
        }
    }
}

impl PresentMode {
    /// Returns the corresponding [vk::PresentModeKHR]
    pub fn to_vk(self) -> vk::PresentModeKHR {
        match self {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        }
    }
}

impl ImageFormat {
    /// Returns the corresponding [vk::Format]
    pub fn to_vk(self) -> vk::Format {
        match self {
            ImageFormat::BC1_RGB_SRGB_BLOCK => vk::Format::BC1_RGB_SRGB_BLOCK,
            ImageFormat::BC1_RGB_UNORM_BLOCK => vk::Format::BC1_RGB_UNORM_BLOCK,
            ImageFormat::BC1_RGBA_SRGB_BLOCK => vk::Format::BC1_RGBA_SRGB_BLOCK,
            ImageFormat::BC1_RGBA_UNORM_BLOCK => vk::Format::BC1_RGBA_UNORM_BLOCK,
            ImageFormat::BC2_SRGB_BLOCK => vk::Format::BC2_SRGB_BLOCK,
            ImageFormat::BC2_UNORM_BLOCK => vk::Format::BC2_UNORM_BLOCK,
            ImageFormat::BC3_SRGB_BLOCK => vk::Format::BC3_SRGB_BLOCK,
            ImageFormat::BC3_UNORM_BLOCK => vk::Format::BC3_UNORM_BLOCK,
            ImageFormat::BC4_SNORM_BLOCK => vk::Format::BC4_SNORM_BLOCK,
            ImageFormat::BC4_UNORM_BLOCK => vk::Format::BC4_UNORM_BLOCK,
            ImageFormat::BC5_SNORM_BLOCK => vk::Format::BC5_SNORM_BLOCK,
            ImageFormat::BC5_UNORM_BLOCK => vk::Format::BC5_UNORM_BLOCK,
            ImageFormat::BC6H_SFLOAT_BLOCK => vk::Format::BC6H_SFLOAT_BLOCK,
            ImageFormat::BC6H_UFLOAT_BLOCK => vk::Format::BC6H_UFLOAT_BLOCK,
            ImageFormat::BC7_SRGB_BLOCK => vk::Format::BC7_SRGB_BLOCK,
            ImageFormat::BC7_UNORM_BLOCK => vk::Format::BC7_UNORM_BLOCK,
            ImageFormat::D16_UNORM => vk::Format::D16_UNORM,
            ImageFormat::D16_UNORM_S8_UINT => vk::Format::D16_UNORM_S8_UINT,
            ImageFormat::D24_UNORM_S8_UINT => vk::Format::D24_UNORM_S8_UINT,
            ImageFormat::D32_SFLOAT => vk::Format::D32_SFLOAT,
            ImageFormat::D32_SFLOAT_S8_UINT => vk::Format::D32_SFLOAT_S8_UINT,
            ImageFormat::R16_SFLOAT => vk::Format::R16_SFLOAT,
            ImageFormat::R16_SINT => vk::Format::R16_SINT,
            ImageFormat::R16_SNORM => vk::Format::R16_SNORM,
            ImageFormat::R16_UINT => vk::Format::R16_UINT,
            ImageFormat::R16_UNORM => vk::Format::R16_UNORM,
            ImageFormat::R16G16_SFLOAT => vk::Format::R16G16_SFLOAT,
            ImageFormat::R16G16_SINT => vk::Format::R16G16_SINT,
            ImageFormat::R16G16_SNORM => vk::Format::R16G16_SNORM,
            ImageFormat::R16G16_UINT => vk::Format::R16G16_UINT,
            ImageFormat::R16G16_UNORM => vk::Format::R16G16_UNORM,
            ImageFormat::R16G16B16_SFLOAT => vk::Format::R16G16B16_SFLOAT,
            ImageFormat::R16G16B16_SINT => vk::Format::R16G16B16_SINT,
            ImageFormat::R16G16B16_SNORM => vk::Format::R16G16B16_SNORM,
            ImageFormat::R16G16B16_UINT => vk::Format::R16G16B16_UINT,
            ImageFormat::R16G16B16_UNORM => vk::Format::R16G16B16_UNORM,
            ImageFormat::R16G16B16A16_SFLOAT => vk::Format::R16G16B16A16_SFLOAT,
            ImageFormat::R16G16B16A16_SINT => vk::Format::R16G16B16A16_SINT,
            ImageFormat::R16G16B16A16_SNORM => vk::Format::R16G16B16A16_SNORM,
            ImageFormat::R16G16B16A16_UINT => vk::Format::R16G16B16A16_UINT,
            ImageFormat::R16G16B16A16_UNORM => vk::Format::R16G16B16A16_UNORM,
            ImageFormat::R32_SFLOAT => vk::Format::R32_SFLOAT,
            ImageFormat::R32_SINT => vk::Format::R32_SINT,
            ImageFormat::R32_UINT => vk::Format::R32_UINT,
            ImageFormat::R32G32_SFLOAT => vk::Format::R32G32_SFLOAT,
            ImageFormat::R32G32_SINT => vk::Format::R32G32_SINT,
            ImageFormat::R32G32_UINT => vk::Format::R32G32_UINT,
            ImageFormat::R32G32B32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
            ImageFormat::R32G32B32_SINT => vk::Format::R32G32B32_SINT,
            ImageFormat::R32G32B32_UINT => vk::Format::R32G32B32_UINT,
            ImageFormat::R32G32B32A32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
            ImageFormat::R32G32B32A32_SINT => vk::Format::R32G32B32A32_SINT,
            ImageFormat::R32G32B32A32_UINT => vk::Format::R32G32B32A32_UINT,
            ImageFormat::R64_SFLOAT => vk::Format::R64_SFLOAT,
            ImageFormat::R64_SINT => vk::Format::R64_SINT,
            ImageFormat::R64_UINT => vk::Format::R64_UINT,
            ImageFormat::R64G64_SFLOAT => vk::Format::R64G64_SFLOAT,
            ImageFormat::R64G64_SINT => vk::Format::R64G64_SINT,
            ImageFormat::R64G64_UINT => vk::Format::R64G64_UINT,
            ImageFormat::R64G64B64_SFLOAT => vk::Format::R64G64B64_SFLOAT,
            ImageFormat::R64G64B64A64_SFLOAT => vk::Format::R64G64B64A64_SFLOAT,
            ImageFormat::R64G64B64A64_SINT => vk::Format::R64G64B64A64_SINT,
            ImageFormat::R64G64B64A64_UINT => vk::Format::R64G64B64A64_UINT,
            ImageFormat::R8_SINT => vk::Format::R8_SINT,
            ImageFormat::R8_SNORM => vk::Format::R8_SNORM,
            ImageFormat::R8_SRGB => vk::Format::R8_SRGB,
            ImageFormat::R8_UINT => vk::Format::R8_UINT,
            ImageFormat::R8_UNORM => vk::Format::R8_UNORM,
            ImageFormat::R8G8_SINT => vk::Format::R8G8_SINT,
            ImageFormat::R8G8_SNORM => vk::Format::R8G8_SNORM,
            ImageFormat::R8G8_SRGB => vk::Format::R8G8_SRGB,
            ImageFormat::R8G8_UINT => vk::Format::R8G8_UINT,
            ImageFormat::R8G8_UNORM => vk::Format::R8G8_UNORM,
            ImageFormat::R8G8B8_SINT => vk::Format::R8G8B8_SINT,
            ImageFormat::R8G8B8_SNORM => vk::Format::R8G8B8_SNORM,
            ImageFormat::R8G8B8_SRGB => vk::Format::R8G8B8_SRGB,
            ImageFormat::R8G8B8_UINT => vk::Format::R8G8B8_UINT,
            ImageFormat::R8G8B8_UNORM => vk::Format::R8G8B8_UNORM,
            ImageFormat::R8G8B8A8_SINT => vk::Format::R8G8B8A8_SINT,
            ImageFormat::R8G8B8A8_SNORM => vk::Format::R8G8B8A8_SNORM,
            ImageFormat::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_SRGB,
            ImageFormat::R8G8B8A8_UINT => vk::Format::R8G8B8A8_UINT,
            ImageFormat::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_UNORM,
        }
    }
}
