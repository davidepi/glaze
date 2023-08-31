use ash::vk;
use std::ffi::{CStr, CString};
use std::ptr;
use super::debug::ValidationLayers;
use super::error::VulkanError;

#[cfg(debug_assertions)]

/// Constains the entry point for the vulkan library.
///
/// Vulkan library does not have a global state and application state is stored in this struct.
/// Alongside the application state, using [ash::Instance] the library loader [ash::Entry] is
/// store.
pub struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl VulkanInstance {
    /// Creates a new vulkan instance.
    ///
    /// if `present` is true, surface presentation support is added to the instance.
    ///
    /// The following extensions support are required:
    /// - `VK_KHR_swapchain` if `present` is true.
    /// - `VK_KHR_xlib_surface` if `present` is true, the platform is linux and `wayland` is false.
    /// - `VK_KHR_wayland_surface` if `present` and `wayland` are true and the platform is linux.
    /// - `VK_KHR_win32_surface` if `present` is true and the platform is windows.
    /// - `VK_EXT_debug_utils` in case of unoptimized builds.
    pub fn new(present: bool, wayland: bool) -> Result<Self, VulkanError> {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(err) => panic!("Failed to create entry: {}", err),
        };
        let validations = ValidationLayers::application_default();
        let mut extensions = Vec::new();
        if cfg!(debug_assertions) {
            if validations.check_support(&entry) {
                extensions.push(ash::extensions::ext::DebugUtils::name());
            } else {
                log::warn!("Debug build, but validation layers are not available. Is the vulkan SDK installed?");
            }
        }
        if present {
            extensions.push(ash::extensions::khr::Surface::name());
            #[cfg(target_os = "linux")]
            if wayland {
                extensions.push(ash::extensions::khr::WaylandSurface::name());
            } else {
                extensions.push(ash::extensions::khr::XlibSurface::name());
            }
            #[cfg(target_os = "windows")]
            extensions.push(ash::extensions::khr::Win32Surface::name());
        }
        let instance = create_instance(&entry, &extensions, validations.names())?;
        Ok(VulkanInstance {
            entry,
            instance,
        })
    }
}

impl Drop for VulkanInstance {
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
    use super::VulkanInstance;

    #[test]
    fn create_present() {
        let instance = VulkanInstance::new(true, false);
        assert!(instance.is_ok())
    }

    #[test]
    fn create_no_present() {
        let instance = VulkanInstance::new(false, false);
        assert!(instance.is_ok())
    }
}
