#[cfg(debug_assertions)]
use crate::vulkan::debug::logger::VkDebugLogger;
use crate::vulkan::debug::ValidationLayers;
use crate::vulkan::device::{Device, PresentDevice, SurfaceSupport};
use crate::vulkan::surface::Surface;
use ash::vk;
use std::ffi::{CStr, CString};
use std::ptr;
use winit::window::Window;

/// Trait used by Vulkan instance wrappers.
///
/// Common trait used by the wrappers of this crate. Allows the retrieval of the instance itself,
/// and the underlying device.
pub trait Instance {
    /// The type of device used by the instance.
    type DeviceItem;

    /// Returns a reference to the raw vulkan instance.
    fn instance(&self) -> &ash::Instance;

    /// Returns a reference to the underlying device wrapper.
    fn device(&self) -> &Self::DeviceItem;
}

/// Vulkan instance for a device supporting a presentation surface.
///
/// Wraps together a Vulkan instance, a device and a presentation surface.
/// When compiled in debug mode, validations are automatically enabled.
pub struct PresentInstance {
    #[cfg(debug_assertions)]
    _logger: VkDebugLogger,
    surface: Surface,
    device: PresentDevice,
    //the following one must be destroyed for last
    instance: BasicInstance,
}

impl PresentInstance {
    /// Creates a new instance using the given window as presentation surface.
    ///
    /// # Extensions
    /// The following Vulkan extensions are required:
    /// - VK_KHR_surface
    /// - VK_KHR_swapchain
    /// - VK_EXT_debug_utils (only when compiled in debug mode)
    /// - VK_KHR_xlib_surface (only when compiled for GNU/Linux)
    /// - VK_KHR_win32_surface (only when compiled for Windows)
    /// - VK_MVK_macos_surface (only when compiled for macOs)
    ///
    /// # Features
    /// The following features are required to be supported on the physical device:
    /// - sampler anisotropy
    ///
    /// # Examples
    /// Basic usage:
    /// ``` no_run
    /// let window = winit::Window::new();
    /// let instance = PresentInstance::new(&window);
    /// ```
    pub fn new(window: &Window) -> Self {
        let instance_extensions = required_extensions();
        let device_extensions = vec![ash::extensions::khr::Swapchain::name()];
        let device_features = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            ..Default::default()
        };
        let instance = BasicInstance::new(&instance_extensions);
        let surface = Surface::new(&instance.entry, &instance.instance, window);
        let device = PresentDevice::new(
            &instance.instance,
            &device_extensions,
            device_features,
            &surface,
        );
        PresentInstance {
            #[cfg(debug_assertions)]
            _logger: VkDebugLogger::new(&instance.entry, &instance.instance),
            instance,
            surface,
            device,
        }
    }

    /// Returns the device used for presentation.
    pub fn present_device(&self) -> &PresentDevice {
        &self.device
    }

    /// Returns the surface used for presentation.
    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    /// Returns the capabilities of the current surface.
    pub fn surface_capabilities(&self) -> SurfaceSupport {
        self.device.physical().surface_capabilities(&self.surface)
    }
}

impl Instance for PresentInstance {
    type DeviceItem = PresentDevice;

    fn instance(&self) -> &ash::Instance {
        &self.instance.instance
    }

    fn device(&self) -> &PresentDevice {
        &self.device
    }
}

/// Returns the required vulkan extension names to present on a surface.
/// Note that each OS returns a different set of extensions.
fn required_extensions() -> Vec<&'static CStr> {
    let retval;
    #[cfg(target_os = "macos")]
    {
        retval = vec![
            ash::extensions::khr::Surface::name(),
            ash::extensions::mvk::MacOSSurface::name(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ]
    }
    #[cfg(target_os = "windows")]
    {
        retval = vec![
            ash::extensions::khr::Surface::name(),
            ash::extensions::khr::Win32Surface::name(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ]
    }
    #[cfg(target_os = "linux")]
    {
        retval = vec![
            ash::extensions::khr::Surface::name(),
            ash::extensions::khr::XlibSurface::name(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ]
    }
    retval
}

/// Basic vulkan instance. Wrapper for ash::Entry and ash::Instance (to avoid Entry being dropped)
struct BasicInstance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl BasicInstance {
    /// creates a new instance with the given extensions
    fn new(extensions: &[&'static CStr]) -> Self {
        let entry = match unsafe { ash::Entry::new() } {
            Ok(entry) => entry,
            Err(err) => panic!("Failed to create entry: {}", err),
        };
        let instance = create_instance(&entry, extensions);
        BasicInstance { entry, instance }
    }
}

impl Drop for BasicInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

/// Creates a new vulkan instance.
fn create_instance(entry: &ash::Entry, extensions: &[&'static CStr]) -> ash::Instance {
    let validations = ValidationLayers::application_default();
    if !validations.check_support(entry) {
        panic!("Some validation layers requested are not available");
    }
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
    let validations_arr = validations.as_ptr();
    let creation_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: validations_arr.len() as u32,
        pp_enabled_layer_names: validations_arr.as_ptr(),
        enabled_extension_count: extensions_array.len() as u32,
        pp_enabled_extension_names: extensions_array.as_ptr(),
    };
    unsafe {
        entry
            .create_instance(&creation_info, None)
            .expect("Failed to create instance")
    }
}
