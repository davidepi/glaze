use super::error::VulkanError;
use super::instance::VulkanInstance;
use super::util::cchars_to_string;
use crate::graphics::device::{Device, PresentDevice};
use crate::graphics::vulkan::physical::PhysicalDeviceVulkan;
use ash::vk;
use std::collections::HashSet;
use std::ffi::CStr;

pub struct DeviceVulkan {
    logical: ash::Device,
    instance: VulkanInstance,
}

impl DeviceVulkan {
    pub fn new_compute(device_id: Option<u64>) -> Result<Self, VulkanError> {
        let instance = VulkanInstance::new(false, false)?;
        let pdevice = if let Some(id) = device_id {
            match PhysicalDeviceVulkan::with_id(&instance, id) {
                Ok(d) => Ok(d),
                Err(e) => {
                    log::error!("{e}. Using default device.");
                    PhysicalDeviceVulkan::with_default(&instance, &[])
                }
            }
        } else {
            PhysicalDeviceVulkan::with_default(&instance, &[])
        }?;
        todo!()
    }
}

impl Device for DeviceVulkan {
    type GraphicError = VulkanError;
    fn supports_raytracing(&self) -> bool {
        todo!()
    }

    fn supports_image_format(
        &self,
        format: crate::graphics::format::ImageFormat,
        usage: crate::graphics::format::ImageUsage,
        optimal: bool,
    ) -> bool {
        todo!()
    }
}

impl PresentDevice for DeviceVulkan {
    fn supports_swapchain(&self, format: crate::graphics::format::ImageFormat) -> bool {
        todo!()
    }

    fn support_present_mode(&self, present_mode: crate::graphics::format::PresentMode) -> bool {
        todo!()
    }
}

fn create_device(instance: &VulkanInstance) -> Result<DeviceVulkan, VulkanError> {
    todo!()
}
