use super::util::is_wayland;
use crate::graphics::device::FeatureSet;
use ash::vk;
use std::ffi::{c_void, CStr};
use std::pin::Pin;
use std::ptr;

/// Contains the vulkan features required for this crate.
///
/// This crate is not a graphics library, extensions and features are hardcoded.
pub struct VulkanFeatures {
    vk10: vk::PhysicalDeviceFeatures2,
    vk12: [vk::PhysicalDeviceVulkan12Features; 1],
}

impl VulkanFeatures {
    pub fn new(set: FeatureSet) -> Pin<Box<Self>> {
        match set {
            FeatureSet::Present => Self::surface(),
            FeatureSet::Convert => Self::convert(),
        }
    }

    /// The set of features that are mandatory for this crate
    fn surface() -> Pin<Box<Self>> {
        let original = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            ..Default::default()
        };
        let vk10 = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: ptr::null_mut(),
            features: original,
        };
        let vk12 = [vk::PhysicalDeviceVulkan12Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            p_next: ptr::null_mut(),
            draw_indirect_count: vk::TRUE,
            timeline_semaphore: vk::TRUE,
            ..Default::default()
        }];
        let res = VulkanFeatures { vk10, vk12 };
        // pin it so it does not move when returning the function
        let mut unmovable = Box::pin(res);
        // set the pointer of the structs
        unmovable.vk10.p_next = unmovable.vk12.as_mut_ptr() as *mut c_void;
        unmovable
    }

    fn convert() -> Pin<Box<Self>> {
        let original = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            ..Default::default()
        };
        let vk10 = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: ptr::null_mut(),
            features: original,
        };
        let vk12 = [vk::PhysicalDeviceVulkan12Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            p_next: ptr::null_mut(),
            ..Default::default()
        }];
        let res = VulkanFeatures { vk10, vk12 };
        // pin it so it does not move when returning the function
        let mut unmovable = Box::pin(res);
        // set the pointer of the structs
        unmovable.vk10.p_next = unmovable.vk12.as_mut_ptr() as *mut c_void;
        unmovable
    }

    /// Returns the pointer to the list of enabled features.
    pub fn as_ffi(&self) -> *const vk::PhysicalDeviceFeatures2 {
        &self.vk10
    }
}

impl FeatureSet {
    pub fn required_instance_extensions(&self) -> Vec<&'static CStr> {
        match self {
            FeatureSet::Present => instance_surface(),
            FeatureSet::Convert => instance_convert(),
        }
    }
    pub fn required_device_extensions(&self) -> Vec<&'static CStr> {
        match self {
            FeatureSet::Present => device_surface(),
            FeatureSet::Convert => device_convert(),
        }
    }
}

fn instance_surface() -> Vec<&'static CStr> {
    let mut ret = Vec::with_capacity(2);
    ret.push(ash::extensions::khr::Surface::name());
    #[cfg(target_os = "windows")]
    ret.push(ash::extensions::khr::Win32Surface::name());
    #[cfg(target_os = "linux")]
    if is_wayland() {
        ret.push(ash::extensions::khr::WaylandSurface::name());
    } else {
        ret.push(ash::extensions::khr::XlibSurface::name());
    }
    ret
}

fn instance_convert() -> Vec<&'static CStr> {
    Vec::new()
}

fn device_surface() -> Vec<&'static CStr> {
    vec![
        ash::extensions::khr::Swapchain::name(),
        ash::extensions::khr::DrawIndirectCount::name(),
    ]
}

fn device_convert() -> Vec<&'static CStr> {
    Vec::new()
}
