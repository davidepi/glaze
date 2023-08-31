use super::error::VulkanError;
use super::instance::VulkanInstance;
use super::util::cchars_to_string;
use crate::graphics::format::{ImageFormat, ImageUsage};
use ash::vk;
use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use std::pin::Pin;
use std::ptr;

/// extensions mandatory for applications without surface.
const MANDATORY_EXTENSIONS: [&CStr; 0] = [];
/// extensions optional for applications without surface.
const OPTIONAL_EXTENSIONS: [&CStr; 0] = [];
/// extensions mandatory for applications with surface.
/// Applied in addition to [MANDATORY_EXTENSIONS]
const MANDATORY_EXTENSIONS_SURFACE: [&CStr; 1] = [ash::extensions::khr::Swapchain::name()];
/// extensions optional for applications with surface.
/// Applied in addition to [OPTIONAL_EXTENSIONS]
const OPTIONAL_EXTENSIONS_SURFACE: [&CStr; 0] = [];

/// Contains the vulkan features required for this crate.
///
/// This crate is not a graphics library, extensions and features are hardcoded.
pub struct VulkanFeatures {
    vk10: vk::PhysicalDeviceFeatures2,
    vk12: [vk::PhysicalDeviceVulkan12Features; 1],
}

impl VulkanFeatures {
    /// The set of features that are mandatory for this crate
    pub fn application_mandatory() -> Pin<Box<Self>> {
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

    /// The set of features that are optional for this crate.
    ///
    /// This is a superset of the mandatory set.
    pub fn application_optional() -> Pin<Box<Self>> {
        VulkanFeatures::application_mandatory()
    }

    /// Returns the features as expected by the Vulkan FFI.
    pub fn as_ffi(&self) -> vk::PhysicalDeviceFeatures2 {
        self.vk10
    }
}

/// Wrapper for a physical device (`VkPhysicalDevice`), with its properties and features.
#[derive(Copy, Clone)]
pub struct PhysicalDeviceVulkan<'instance> {
    instance: &'instance ash::Instance,
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    features: vk::PhysicalDeviceFeatures,
}

impl PhysicalDeviceVulkan<'_> {
    /// Checks if the input vulkan extensions are supported by the device
    fn supports_extensions(&self, ext: &[&'static CStr]) -> Result<bool, VulkanError> {
        let available = unsafe {
            self.instance
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
        format: vk::Format,
        usage: vk::FormatFeatureFlags,
        optimal: bool,
    ) -> bool {
        let props = unsafe {
            self.instance
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
    pub fn with_default<'a>(
        instance: &'a VulkanInstance,
        formats: &[(ImageFormat, ImageUsage, bool)],
    ) -> Result<PhysicalDeviceVulkan<'a>, VulkanError> {
        let mandatory = Self::list_compatible_mandatory(instance, formats)?;
        let optional = Self::list_compatible_optional(instance, formats)?;
        // first try with dedicated devices supporting optional extensions
        if let Some(dedicated_optional) = optional
            .iter()
            .find(|device| device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        {
            Ok(*dedicated_optional)
        } else if let Some(dedicated_mandatory) = mandatory
            .iter()
            .find(|device| device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        {
            Ok(*dedicated_mandatory)
        } else {
            // then try with integrated optional
            if let Some(integrated_optional) = optional.iter().find(|device| {
                device.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            }) {
                Ok(*integrated_optional)
            } else if let Some(integrated_mandatory) = mandatory.iter().find(|device| {
                device.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            }) {
                Ok(*integrated_mandatory)
            } else {
                Err(VulkanError::Custom(
                    "Failed to find a compatible device".to_string(),
                ))
            }
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
        instance: &VulkanInstance,
        id: u64,
    ) -> Result<PhysicalDeviceVulkan, VulkanError> {
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
            Err(VulkanError::Custom(error))
        }
    }

    /// Lists the physical devices compatible with the crate in term of mandatory extensions.
    ///
    /// Dunno how to decently check for features so they are not checked...
    pub fn list_compatible_mandatory<'a>(
        instance: &'a VulkanInstance,
        formats: &[(ImageFormat, ImageUsage, bool)],
    ) -> Result<Vec<PhysicalDeviceVulkan<'a>>, VulkanError> {
        let mandatory = PhysicalDeviceVulkan::list_all(instance)?
            .into_iter()
            .filter(|device| {
                device
                    .supports_extensions(&MANDATORY_EXTENSIONS)
                    .unwrap_or(false)
            })
            .filter(|device| {
                if instance.supports_presentation() {
                    device
                        .supports_extensions(&MANDATORY_EXTENSIONS_SURFACE)
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .filter(|device| {
                for (format, usage, optimal) in formats {
                    if !device.supports_format(format.to_vk(), usage.to_vk_format(), *optimal) {
                        return false;
                    }
                }
                true
            })
            .collect::<Vec<_>>();
        Ok(mandatory)
    }

    /// Lists the physical devices compatible with the crate in term of optional extensions.
    ///
    /// As with the [list_compatible_mandatory], features are not checked.
    pub fn list_compatible_optional<'a>(
        instance: &'a VulkanInstance,
        formats: &[(ImageFormat, ImageUsage, bool)],
    ) -> Result<Vec<PhysicalDeviceVulkan<'a>>, VulkanError> {
        let mandatory = Self::list_compatible_mandatory(instance, formats)?;
        let optional = mandatory
            .iter()
            .filter(|device| {
                device
                    .supports_extensions(&OPTIONAL_EXTENSIONS)
                    .unwrap_or(false)
            })
            .filter(|device| {
                if instance.supports_presentation() {
                    device
                        .supports_extensions(&OPTIONAL_EXTENSIONS_SURFACE)
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .copied()
            .collect::<Vec<_>>();
        Ok(optional)
    }

    /// Lists all the physical devices on the system.
    pub fn list_all(instance: &VulkanInstance) -> Result<Vec<PhysicalDeviceVulkan>, VulkanError> {
        let vk_instance = instance.vk_instance();
        let physical_devices = unsafe { vk_instance.enumerate_physical_devices() }?;
        let mut retval = Vec::with_capacity(physical_devices.len());
        for device in physical_devices {
            let properties = unsafe { vk_instance.get_physical_device_properties(device) };
            let features = unsafe { vk_instance.get_physical_device_features(device) };
            retval.push(PhysicalDeviceVulkan {
                instance: vk_instance,
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
    use crate::graphics::format::{ImageFormat, ImageUsage};
    use crate::graphics::vulkan::instance::VulkanInstance;

    #[test]
    fn gpu_exists() {
        let instance = VulkanInstance::new(false, false).unwrap();
        let gpus_no = PhysicalDeviceVulkan::list_all(&instance).unwrap().len();
        assert!(gpus_no > 0);
    }

    #[test]
    fn gpu_with_video_exists() {
        let instance = VulkanInstance::new(true, false).unwrap();
        let gpus_no = PhysicalDeviceVulkan::list_all(&instance).unwrap().len();
        assert!(gpus_no > 0);
    }

    #[test]
    fn select_optimal() {
        let instance = VulkanInstance::new(false, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance, &[]);
        assert!(gpu.is_ok());
    }

    #[test]
    fn select_optimal_video() {
        // regression test (lol so simple, but initially I used Surface instead of Swapchain inside
        // the MANDATORY_EXTENSIONS_SURFACE, so it was failing only select_optimal with
        // presentation support)
        let instance = VulkanInstance::new(true, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance, &[]);
        assert!(gpu.is_ok());
    }

    #[test]
    fn select_optimal_supported_format() {
        let instance = VulkanInstance::new(false, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(
            &instance,
            &[(ImageFormat::D32_SFLOAT, ImageUsage::DepthStencil, true)],
        );
        assert!(gpu.is_ok());
    }

    #[test]
    fn select_optimal_unsupported_format() {
        let instance = VulkanInstance::new(false, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(
            &instance,
            &[(ImageFormat::R8_SRGB, ImageUsage::DepthStencil, true)],
        );
        assert!(gpu.is_err());
    }

    #[test]
    fn supported_extension() {
        let instance = VulkanInstance::new(true, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance, &[]).unwrap();
        let ext_support = gpu
            .supports_extensions(&[ash::extensions::khr::Swapchain::name()])
            .unwrap();
        assert!(ext_support)
    }

    #[test]
    fn unsupported_extension() {
        let instance = VulkanInstance::new(true, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance, &[]).unwrap();
        // vulkan will never run on ios for this crate, so this test will always fail
        let ext_support = gpu
            .supports_extensions(&[ash::extensions::mvk::IOSSurface::name()])
            .unwrap();
        assert!(!ext_support)
    }

    #[test]
    fn by_id() {
        let instance = VulkanInstance::new(false, false).unwrap();
        let gpu = PhysicalDeviceVulkan::with_default(&instance, &[]).unwrap();
        let id = (gpu.properties.vendor_id as u64) << 32 | gpu.properties.device_id as u64;
        let gpu_by_id = PhysicalDeviceVulkan::with_id(&instance, id).unwrap();
        assert_eq!(gpu.properties.device_name, gpu_by_id.properties.device_name);
    }
}
