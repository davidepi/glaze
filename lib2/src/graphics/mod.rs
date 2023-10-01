#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use self::metal::DeviceMetal;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod vulkan;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use self::vulkan::DeviceVulkan;

pub mod config;
pub mod device;
pub mod error;
pub mod format;
pub mod memory;
#[cfg(any(feature = "display", doc))]
pub mod swapchain;

#[cfg(test)]
mod testhelpers {
    use super::device::Device;
    use super::error::GraphicError;

    pub fn create_device() -> Result<impl Device, GraphicError> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            super::vulkan::DeviceVulkan::new(None, super::device::FeatureSet::Convert)
        }
        #[cfg(target_os = "macos")]
        {
            super::metal::DeviceMetal::new(None)
        }
    }
}
