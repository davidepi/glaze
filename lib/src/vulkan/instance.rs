#[cfg(debug_assertions)]
use crate::vulkan::debug::logger::VkDebugLogger;
use crate::vulkan::debug::ValidationLayers;
use crate::vulkan::device::{Device, SurfaceSupport};
use crate::vulkan::surface::Surface;
use crate::DeviceInfo;
use ash::vk;
use std::ffi::{CStr, CString};
use std::ptr;
use winit::window::Window;

/// Trait used by Vulkan instance wrappers.
///
/// Common trait used by the wrappers of this crate. Allows the retrieval of the instance itself,
/// and the underlying device.
pub trait Instance {
    /// Returns a reference to the raw vulkan instance.
    fn instance(&self) -> &ash::Instance;

    /// Returns a reference to the underlying device wrapper.
    fn device(&self) -> &Device;

    /// Returns the list of extensions enabled for this instance.
    fn extensions(&self) -> &[String];
}

/// Vulkan instance for a device supporting a presentation surface.
///
/// Wraps together a Vulkan instance, a device and a presentation surface.
/// When compiled in debug mode, validations are automatically enabled.
///
/// Supports raytracing if there is at least one GPU in the system supporting presentation and
/// raytracing combined.
pub struct PresentInstance {
    #[cfg(debug_assertions)]
    _logger: VkDebugLogger,
    surface: Surface,
    device: Device,
    raytrace: bool,
    enabled_extensions: Vec<String>,
    //the following one must be destroyed for last
    instance: BasicInstance,
}

impl PresentInstance {
    /// Creates a new instance using the given window as presentation surface.
    ///
    /// The method attempts at creating an instance that can support raytracing. If this fails, a
    /// normal instance is instead created. This failure is silent and, to know which instance was
    /// created, the method [PresentInstance::supports_raytrace] is provided.
    ///
    /// Returns None if no matching device can be found.
    ///
    /// # Extensions
    /// The following Vulkan extensions are required for presentation only:
    /// - VK_KHR_surface
    /// - VK_KHR_swapchain
    /// - VK_EXT_debug_utils (only when compiled in debug mode)
    /// - VK_KHR_xlib_surface (only when compiled for GNU/Linux)
    /// - VK_KHR_win32_surface (only when compiled for Windows)
    /// - VK_MVK_macos_surface (only when compiled for macOs)
    ///
    /// Additionally, to support raytrace, the following extensions are required:
    /// - VK_KHR_deferred_host_operations
    /// - VK_KHR_acceleration_structure
    /// - VK_KHR_ray_tracing_pipeline
    ///
    /// # Features
    /// The following features are required to be supported on the physical device:
    /// - sampler anisotropy
    ///
    /// # Examples
    /// Basic usage:
    /// ``` no_run
    /// let mut event_loop = winit::event_loop::EventLoop::new();
    /// let window = winit::window::Window::new(&event_loop).unwrap();
    /// let instance = glaze::PresentInstance::new(&window);
    /// ```
    pub fn new(window: &Window) -> Option<Self> {
        let instance_extensions = required_extensions();
        let present_only_ext = vec![ash::extensions::khr::Swapchain::name()];
        let present_and_raytrace_ext = vec![
            ash::extensions::khr::Swapchain::name(),
            ash::extensions::khr::DeferredHostOperations::name(),
            ash::extensions::khr::AccelerationStructure::name(),
            ash::extensions::khr::RayTracingPipeline::name(),
        ];
        let device_features = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            ..Default::default()
        };
        let instance = BasicInstance::new(&instance_extensions);
        let surface = Surface::new(&instance.entry, &instance.instance, window);
        let maybe_raytrace_device = Device::new_present(
            &instance.instance,
            &present_and_raytrace_ext,
            device_features,
            &surface,
        );
        if let Some(device) = maybe_raytrace_device {
            let enabled_extensions = present_and_raytrace_ext
                .into_iter()
                .map(CStr::to_bytes)
                .flat_map(std::str::from_utf8)
                .map(String::from)
                .collect::<Vec<_>>();
            Some(PresentInstance {
                #[cfg(debug_assertions)]
                _logger: VkDebugLogger::new(&instance.entry, &instance.instance),
                surface,
                device,
                raytrace: true,
                enabled_extensions,
                instance,
            })
        } else {
            let maybe_device = Device::new_present(
                &instance.instance,
                &present_only_ext,
                device_features,
                &surface,
            );
            if let Some(device) = maybe_device {
                let enabled_extensions = present_only_ext
                    .into_iter()
                    .map(CStr::to_bytes)
                    .flat_map(std::str::from_utf8)
                    .map(String::from)
                    .collect::<Vec<_>>();
                Some(PresentInstance {
                    #[cfg(debug_assertions)]
                    _logger: VkDebugLogger::new(&instance.entry, &instance.instance),
                    surface,
                    device,
                    raytrace: false,
                    enabled_extensions,
                    instance,
                })
            } else {
                None
            }
        }
    }

    /// Returns true if the instance supports raytracing.
    pub fn supports_raytrace(&self) -> bool {
        self.raytrace
    }

    /// Returns the surface used for presentation.
    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    /// Returns the capabilities of the current surface.
    pub fn surface_capabilities(&self) -> SurfaceSupport {
        self.device.physical().surface_capabilities(&self.surface)
    }

    /// Returns the properties of the underlying physical device.
    pub fn device_properties(&self) -> DeviceInfo {
        self.device.physical().info()
    }

    /// Returns a list of loaded extensions.
    pub fn loaded_extensions(&self) -> &[String] {
        // this trait is not exposed (otherwise I have to expose the entirety of Device)
        // hence the reason of this method
        self.extensions()
    }
}

impl Instance for PresentInstance {
    fn instance(&self) -> &ash::Instance {
        &self.instance.instance
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn extensions(&self) -> &[String] {
        &self.enabled_extensions
    }
}

/// Vulkan instance for a device supporting raytracing.
///
/// Wraps together a Vulkan instance and a device.
/// When compiled in debug mode, validations are automatically enabled.
///
/// No presentation provided, this is meant mostly for command line applications.
/// Use [PresentInstance] to have a device supporting both presentation and raytracing.
pub struct RayTraceInstance {
    #[cfg(debug_assertions)]
    _logger: VkDebugLogger,
    device: Device,
    enabled_extensions: Vec<String>,
    //the following one must be destroyed for last
    instance: BasicInstance,
}

impl RayTraceInstance {
    /// Creates a new instance that can be used for raytracing, without presentation support.
    ///
    /// Returns None if no supported device can be found.
    ///
    /// # Extensions
    /// The following Vulkan extensions are required:
    /// - VK_EXT_debug_utils (only when compiled in debug mode)
    /// - VK_KHR_deferred_host_operations
    /// - VK_KHR_acceleration_structure
    /// - VK_KHR_ray_tracing_pipeline
    ///
    /// # Features
    /// No features are required to be supported on the physical device:
    ///
    /// # Examples
    /// Basic usage:
    /// ``` no_run
    /// let instance = glaze::RayTracingInstance::new();
    /// ```
    pub fn new() -> Option<Self> {
        let instance_extensions = vec![
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ];
        let device_extensions = vec![
            ash::extensions::khr::DeferredHostOperations::name(),
            ash::extensions::khr::AccelerationStructure::name(),
            ash::extensions::khr::RayTracingPipeline::name(),
        ];
        let device_features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };
        let instance = BasicInstance::new(&instance_extensions);
        let maybe_device =
            Device::new_compute(&instance.instance, &device_extensions, device_features);
        if let Some(device) = maybe_device {
            let enabled_extensions = device_extensions
                .into_iter()
                .map(CStr::to_bytes)
                .flat_map(std::str::from_utf8)
                .map(String::from)
                .collect::<Vec<_>>();
            Some(RayTraceInstance {
                #[cfg(debug_assertions)]
                _logger: VkDebugLogger::new(&instance.entry, &instance.instance),
                device,
                enabled_extensions,
                instance,
            })
        } else {
            None
        }
    }
}

impl Instance for RayTraceInstance {
    fn instance(&self) -> &ash::Instance {
        &self.instance.instance
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn extensions(&self) -> &[String] {
        &self.enabled_extensions
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
        let entry = match unsafe { ash::Entry::load() } {
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
