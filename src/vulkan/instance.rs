use super::device::SurfaceSupport;
use super::platform;
use super::surface::Surface;
use crate::vulkan::debug::ValidationLayers;
use crate::vulkan::device::create_logical_device;
use crate::vulkan::device::PhysicalDevice;
use ash::vk;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use winit::window::Window;

#[cfg(debug_assertions)]
use crate::vulkan::debug::logger::VkDebugLogger;

pub trait Instance {
    fn entry(&self) -> &ash::Entry;
    fn instance(&self) -> &ash::Instance;
    fn device(&self) -> &ash::Device;
    fn physical_device(&self) -> &PhysicalDevice;
}

pub struct PresentedInstance {
    #[cfg(debug_assertions)]
    logger: VkDebugLogger,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: ash::Device,
    //the following one must be destroyed for last
    instance: BasicInstance,
}

impl PresentedInstance {
    pub fn new(window: &Window) -> Self {
        let instance_extensions = platform::required_extensions();
        let device_extensions = vec![ash::extensions::khr::Swapchain::name()];
        let instance = BasicInstance::new(&instance_extensions);
        let surface = Surface::new(&instance.entry, &instance.instance, &window);
        let physical_device =
            PhysicalDevice::list_compatible(&instance.instance, &device_extensions, &surface)
                .into_iter()
                .filter(|x| {
                    x.surface_capabilities(&surface)
                        .has_formats_and_present_modes()
                })
                .last()
                .expect("No compatible devices found");
        let device =
            create_logical_device(&instance.instance, &device_extensions, &physical_device);
        PresentedInstance {
            #[cfg(debug_assertions)]
            logger: VkDebugLogger::new(&instance.entry, &instance.instance),
            instance,
            surface,
            physical_device,
            device,
        }
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn surface_capabilities(&self) -> SurfaceSupport {
        self.physical_device.surface_capabilities(&self.surface)
    }
}

impl Drop for PresentedInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

impl Instance for PresentedInstance {
    fn entry(&self) -> &ash::Entry {
        &self.instance.entry
    }

    fn instance(&self) -> &ash::Instance {
        &self.instance.instance
    }

    fn device(&self) -> &ash::Device {
        &self.device
    }

    fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
}

struct BasicInstance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl BasicInstance {
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
    let application_name = CString::new(&app_name_string[..]).unwrap();
    let engine_name = CString::new(&engine_name_string[..]).unwrap();
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
    let creation_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: validations.len() as u32,
        pp_enabled_layer_names: validations.as_ptr(),
        enabled_extension_count: extensions_array.len() as u32,
        pp_enabled_extension_names: extensions_array.as_ptr(),
    };
    unsafe {
        entry
            .create_instance(&creation_info, None)
            .expect("Failed to create instance")
    }
}
