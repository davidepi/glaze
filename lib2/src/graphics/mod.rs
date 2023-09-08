#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use self::metal::MetalDevice;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod vulkan;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use self::vulkan::VulkanDevice;

mod device;
mod format;
