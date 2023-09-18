#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use self::metal::MetalDevice;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod vulkan;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use self::vulkan::DeviceVulkan;

pub mod device;
pub mod error;
pub mod format;
#[cfg(any(feature = "display", doc))]
pub mod swapchain;

#[cfg(test)]
mod testhelpers {
    use super::device::{Device, FeatureSet};
    use super::error::GraphicError;

    pub fn create_device() -> Result<impl Device, GraphicError> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            super::vulkan::DeviceVulkan::new(None, FeatureSet::Convert)
        }
        #[cfg(target_os = "macos")]
        {
            super::metal::DeviceMetal::new(None, FeatureSet::Convert)
        }
    }
}
