use super::surface::Surface;
use crate::vulkan::debug::ValidationLayers;
use crate::vulkan::device::{create_logical_device, PhysicalDevice};
use crate::vulkan::platform::required_extension_names;
use ash::vk;
use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    ptr,
};
use winit::window::Window;

#[cfg(debug_assertions)]
use crate::vulkan::debug::logger::VkDebugLogger;

pub struct VkPresentedInstance {
    #[cfg(debug_assertions)]
    logger: VkDebugLogger,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: ash::Device,
    instance: VkInstance,
}

impl VkPresentedInstance {
    pub fn new(instance: VkInstance, window: &Window) -> Self {
        let surface = Surface::new(&instance, &window);
        let physical_device = PhysicalDevice::list_compatible(&instance, &surface)
            .into_iter()
            .filter(|x| {
                x.surface_capabilities(&surface)
                    .has_formats_and_present_modes()
            })
            .last()
            .expect("No compatible devices found");
        let device = create_logical_device(&instance, &physical_device);
        VkPresentedInstance {
            #[cfg(debug_assertions)]
            logger: VkDebugLogger::new(&instance),
            instance,
            surface,
            physical_device,
            device,
        }
    }
}

pub struct VkInstance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub required_extensions: Vec<&'static CStr>,
}

impl VkInstance {
    pub fn new(required_extensions: &[&'static CStr]) -> Self {
        let entry = match unsafe { ash::Entry::new() } {
            Ok(entry) => entry,
            Err(err) => panic!("Failed to create entry: {}", err),
        };
        let instance = create_instance(&entry, required_extensions);
        let required_extensions = required_extensions.into_iter().copied().collect::<Vec<_>>();
        VkInstance {
            entry,
            instance,
            required_extensions,
        }
    }

    pub fn required_extensions(&self) -> &[&'static CStr] {
        &self.required_extensions
    }
}

impl Drop for VkInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn create_instance(entry: &ash::Entry, required_extensions: &[&'static CStr]) -> ash::Instance {
    let extensions = required_extensions;
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
    let extensions = extensions
        .iter()
        .cloned()
        .chain(required_extension_names().into_iter())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
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
