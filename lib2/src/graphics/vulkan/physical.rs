use super::instance::InstanceVulkan;
use super::util::cchars_to_string;
use crate::graphics::error::{ErrorCategory, GraphicError};
use crate::graphics::format::FeatureSet;
use ash::vk;
use std::collections::HashSet;
use std::ffi::CStr;

/// Wrapper for a physical device (`VkPhysicalDevice`), with its properties and features.
#[derive(Copy, Clone)]
pub struct PhysicalDeviceVulkan {
    pub device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
}

impl PhysicalDeviceVulkan {
    /// Checks if the input vulkan extensions are supported by the device
    fn supports_extensions(
        &self,
        instance: &InstanceVulkan,
        ext: &[&'static CStr],
    ) -> Result<bool, GraphicError> {
        let available = unsafe {
            instance
                .vk_instance()
                .enumerate_device_extension_properties(self.device)
        }?
        .into_iter()
        .map(|x| cchars_to_string(&x.extension_name))
        .collect::<HashSet<_>>();
        let check = !ext
            .iter()
            .map(|x| x.to_str().unwrap().to_string())
            .any(|x| !available.contains(&x));
        Ok(check)
    }

    /// Checks if the device supports the given format (and usage) in either linear or optimal
    /// tiling mode.
    fn supports_format(
        &self,
        instance: &InstanceVulkan,
        format: vk::Format,
        usage: vk::FormatFeatureFlags,
        optimal: bool,
    ) -> bool {
        let props = unsafe {
            instance
                .vk_instance()
                .get_physical_device_format_properties(self.device, format)
        };
        if optimal {
            props.optimal_tiling_features.contains(usage)
        } else {
            props.linear_tiling_features.contains(usage)
        }
    }

    /// Selects the best suited physical device.
    ///
    /// It is not possible to get a "default" GPU with vulkan, so this routine tries to select
    /// the best dedicated device possible (supporting all extensions).
    ///
    /// No idea how to decently tests for features, so it's not implemented.
    pub fn with_default(instance: &InstanceVulkan) -> Result<PhysicalDeviceVulkan, GraphicError> {
        let features = instance.features();
        let devices = Self::list_compatible(instance, features)?;
        if let Some(dedicated) = devices
            .iter()
            .find(|device| device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        {
            Ok(*dedicated)
        } else if let Some(integrated) = devices
            .iter()
            .find(|device| device.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU)
        {
            Ok(*integrated)
        } else {
            Err(GraphicError::new(
                ErrorCategory::InitFailed,
                "Failed to find a compatible device",
            ))
        }
    }

    /// Returns the device with the specified ID.
    ///
    /// If no device can be found, [PhysicalDeviceVulkan::with_default] will be used.
    ///
    /// Won't check for extensions support or format support, so logical device creation MAY fail.
    /// The idea is that the GPU with this ID has been previously collected by using [list_all].
    /// ID is composed by vendor_id in MSB and device id in LSB.
    pub fn with_id(
        instance: &InstanceVulkan,
        id: u64,
    ) -> Result<PhysicalDeviceVulkan, GraphicError> {
        let vendor_id = ((id & 0xFFFFFFFF00000000) >> 32) as u32;
        let device_id = (id & 0x00000000FFFFFFFF) as u32;
        if let Some(device) = PhysicalDeviceVulkan::list_all(instance)?
            .into_iter()
            .find(|device| {
                device.properties.vendor_id == vendor_id && device.properties.device_id == device_id
            })
        {
            Ok(device)
        } else {
            let error = format!(
                "Could not find device with id {}. Using default device.",
                id
            );
            Err(GraphicError::new(ErrorCategory::InitFailed, error))
        }
    }

    /// List the devices compatible with the given feature set
    pub fn list_compatible(
        instance: &InstanceVulkan,
        features: FeatureSet,
    ) -> Result<Vec<PhysicalDeviceVulkan>, GraphicError> {
        let exts = features.required_device_extensions();
        let formats = features.required_formats();
        let list = PhysicalDeviceVulkan::list_all(instance)?
            .into_iter()
            .filter(|device| device.supports_extensions(instance, &exts).unwrap_or(false))
            .filter(|device| {
                for (format, usage, optimal) in &formats {
                    if !device.supports_format(
                        instance,
                        format.to_vk(),
                        usage.to_vk_format(),
                        *optimal,
                    ) {
                        return false;
                    }
                }
                true
            })
            .collect();
        Ok(list)
    }

    /// Lists all the physical devices on the system.
    pub fn list_all(instance: &InstanceVulkan) -> Result<Vec<PhysicalDeviceVulkan>, GraphicError> {
        let vk_instance = instance.vk_instance();
        let physical_devices = unsafe { vk_instance.enumerate_physical_devices() }?;
        let mut retval = Vec::with_capacity(physical_devices.len());
        for device in physical_devices {
            let properties = unsafe { vk_instance.get_physical_device_properties(device) };
            let features = unsafe { vk_instance.get_physical_device_features(device) };
            retval.push(PhysicalDeviceVulkan {
                device,
                properties,
                features,
            });
        }
        Ok(retval)
    }
}

#[cfg(test)]
mod tests {
    use super::PhysicalDeviceVulkan;
    use crate::graphics::error::GraphicError;
    use crate::graphics::format::{FeatureSet, ImageFormat, ImageUsage};
    use crate::graphics::vulkan::instance::InstanceVulkan;

    #[test]
    fn gpu_exists() {
        let instance = InstanceVulkan::new(FeatureSet::Convert).unwrap();
        let gpus_no = PhysicalDeviceVulkan::list_all(&instance).unwrap().len();
        assert!(gpus_no > 0);
    }

    #[test]
    fn gpu_with_video_exists() {
        let instance = InstanceVulkan::new(FeatureSet::Present).unwrap();
        let gpus_no = PhysicalDeviceVulkan::list_all(&instance).unwrap().len();
        assert!(gpus_no > 0);
    }

    #[test]
    fn select_optimal() -> Result<(), GraphicError> {
        let instance = InstanceVulkan::new(FeatureSet::Convert)?;
        PhysicalDeviceVulkan::with_default(&instance)?;
        Ok(())
    }

    #[test]
    fn select_optimal_video() -> Result<(), GraphicError> {
        // regression test (lol so simple, but initially I used Surface instead of Swapchain inside
        // the extensions, so it was failing only with FeatureSet::Present
        let instance = InstanceVulkan::new(FeatureSet::Present).unwrap();
        PhysicalDeviceVulkan::with_default(&instance)?;
        Ok(())
    }

    #[test]
    fn select_optimal_supported_format() -> Result<(), GraphicError> {
        let instance = InstanceVulkan::new(FeatureSet::Convert)?;
        PhysicalDeviceVulkan::with_default(&instance)?;
        Ok(())
    }

    #[test]
    fn select_optimal_unsupported_format() {
        let instance = InstanceVulkan::new(FeatureSet::Convert).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance).unwrap();
        let format_support = gpu.supports_format(
            ImageFormat::R8_SRGB.to_vk(),
            ImageUsage::DepthStencil.to_vk_format(),
            true,
        );
        assert!(!format_support);
    }

    #[test]
    fn supported_extension() {
        let instance = InstanceVulkan::new(FeatureSet::Present).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance).unwrap();
        let ext_support = gpu
            .supports_extensions(&[ash::extensions::khr::Swapchain::name()])
            .unwrap();
        assert!(ext_support)
    }

    #[test]
    fn unsupported_extension() {
        let instance = InstanceVulkan::new(FeatureSet::Present).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance).unwrap();
        // vulkan will never run on ios for this crate, so this test will always fail
        let ext_support = gpu
            .supports_extensions(&[ash::extensions::mvk::IOSSurface::name()])
            .unwrap();
        assert!(!ext_support)
    }

    #[test]
    fn by_id() {
        let instance = InstanceVulkan::new(FeatureSet::Present).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance).unwrap();
        let id = (gpu.properties.vendor_id as u64) << 32 | gpu.properties.device_id as u64;
        let gpu_by_id = PhysicalDeviceVulkan::with_id(&instance, id).unwrap();
        assert_eq!(gpu.properties.device_name, gpu_by_id.properties.device_name);
    }
}
