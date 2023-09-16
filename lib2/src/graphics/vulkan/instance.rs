use super::debug::ValidationLayers;
use super::error::VulkanError;
use crate::graphics::format::FeatureSet;
use ash::vk;
use std::ffi::{CStr, CString};
use std::ptr;

/// Constains the entry point for the vulkan library.
///
/// Vulkan library does not have a global state and application state is stored in this struct.
/// Alongside the application state, using [ash::Instance] the library loader [ash::Entry] is
/// store.
pub struct InstanceVulkan {
    features: FeatureSet,
    instance: ash::Instance,
    entry: ash::Entry,
}

impl InstanceVulkan {
    /// Returns the entry point for the vulkan library
    pub fn vk_loader(&self) -> &ash::Entry {
        &self.entry
    }

    /// Returns the vulkan instance handle provided by the ash crate.
    pub fn vk_instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Returns the features requested when creating the instance.
    pub fn features(&self) -> FeatureSet {
        self.features
    }

    /// Returns true if presentation support is enable at instance level.
    pub fn supports_presentation(&self) -> bool {
        matches!(self.features, FeatureSet::Present)
    }

    /// Creates a new vulkan instance.
    ///
    /// The required extensions are defined by the struct [ExtensionsVulkanInstance].
    /// This struct, additionally, injects the `VK_EXT_debug_utils` extension for debugging
    /// purposes if the build is unoptimized.
    pub fn new(features: FeatureSet) -> Result<Self, VulkanError> {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(err) => return Err(VulkanError::new(format!("Failed to create entry: {}", err))),
        };
        let validations = ValidationLayers::application_default();
        let mut extensions = features.required_instance_extensions();
        if cfg!(debug_assertions) {
            if validations.check_support(&entry) {
                extensions.push(ash::extensions::ext::DebugUtils::name());
            } else {
                log::warn!("Debug build, but validation layers are not available. Is the vulkan SDK installed?");
            }
        }
        let instance = create_instance(&entry, &extensions, validations.names())?;
        Ok(InstanceVulkan {
            features,
            entry,
            instance,
        })
    }
}

impl Drop for InstanceVulkan {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

/// Creates a new vulkan instance.
fn create_instance(
    entry: &ash::Entry,
    extensions: &[&'static CStr],
    validations: &[CString],
) -> Result<ash::Instance, VulkanError> {
    let app_name_string = format!("{}-app", env!("CARGO_PKG_NAME"));
    let engine_name_string = env!("CARGO_PKG_NAME");
    let ver_major = env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap();
    let ver_minor = env!("CARGO_PKG_VERSION_MINOR").parse::<u32>().unwrap();
    let ver_patch = env!("CARGO_PKG_VERSION_PATCH").parse::<u32>().unwrap();
    let application_name = CString::new(app_name_string).unwrap();
    let engine_name = CString::new(engine_name_string).unwrap();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: application_name.as_ptr(),
        application_version: vk::make_api_version(0, ver_major, ver_minor, ver_patch),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0, ver_major, ver_minor, ver_patch),
        api_version: vk::API_VERSION_1_2,
    };
    let extensions_array = extensions.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
    let validations_array = validations.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
    let creation_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: validations_array.len() as u32,
        pp_enabled_layer_names: validations_array.as_ptr(),
        enabled_extension_count: extensions_array.len() as u32,
        pp_enabled_extension_names: extensions_array.as_ptr(),
    };
    let instance = unsafe { entry.create_instance(&creation_info, None) }?;
    Ok(instance)
}

#[cfg(test)]
mod tests {
    use super::InstanceVulkan;
    use crate::graphics::format::FeatureSet;
    use crate::graphics::vulkan::error::VulkanError;

    #[test]
    fn create_present() -> Result<(), VulkanError> {
        InstanceVulkan::new(FeatureSet::Present)?;
        Ok(())
    }

    #[test]
    fn create_no_present() -> Result<(), VulkanError> {
        InstanceVulkan::new(FeatureSet::Convert)?;
        Ok(())
    }
}
