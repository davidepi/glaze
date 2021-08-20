use crate::vulkan::device::{create_logical_device, PhysicalDevice};
use crate::vulkan::platform::{self, required_extension_names};
use crate::vulkan::validations::ValidationsRequested;
use ash::vk;
use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    ptr,
};
use winit::window::Window;

pub struct VkInstance {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: ash::Device,
}

impl VkInstance {
    pub fn new(required_extensions: &[&'static CStr], window: &Window) -> Self {
        let entry = match unsafe { ash::Entry::new() } {
            Ok(entry) => entry,
            Err(err) => panic!("Failed to create entry: {}", err),
        };
        let validations = ValidationsRequested::new(&["VK_LAYER_KHRONOS_validation"]);
        let instance = create_instance(&entry, required_extensions, &validations);
        let surface = Surface::new(&entry, &instance, window);
        let physical_device =
            PhysicalDevice::list_compatible(&instance, &surface, required_extensions)
                .into_iter()
                .filter(|x| {
                    x.surface_capabilities(&surface)
                        .has_formats_and_present_modes()
                })
                .last()
                .expect("No compatible devices found");
        let device = create_logical_device(
            &instance,
            &physical_device,
            required_extensions,
            &validations,
        );
        VkInstance {
            _entry: entry,
            instance,
            surface,
            physical_device,
            device,
        }
    }
}

impl Drop for VkInstance {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface
                .loader
                .destroy_surface(self.surface.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub loader: ash::extensions::khr::Surface,
}

impl Surface {
    fn new(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> Self {
        let surface = unsafe { platform::create_surface(entry, instance, window) }
            .expect("Failed to create surface");
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);
        Surface {
            surface,
            loader: surface_loader,
        }
    }
}

fn create_instance(
    entry: &ash::Entry,
    required_extensions: &[&'static CStr],
    validations: &ValidationsRequested,
) -> ash::Instance {
    #[cfg(debug_assertions)]
    {
        if !validations.check_support(entry) {
            panic!("Some validation layers requested are not available");
        }
    }
    let app_name_string = format!("{}-app", env!("CARGO_PKG_NAME"));
    let engine_name_string = format!("{}", env!("CARGO_PKG_NAME"));
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
    let extensions = required_extensions
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
        #[cfg(debug_assertions)]
        enabled_layer_count: validations.layers_ptr().len() as u32,
        #[cfg(debug_assertions)]
        pp_enabled_layer_names: validations.layers_ptr().as_ptr(),
        #[cfg(not(debug_assertions))]
        enabled_layer_count: 0,
        #[cfg(not(debug_assertions))]
        pp_enabled_layer_names: ptr::null(),
        enabled_extension_count: extensions_array.len() as u32,
        pp_enabled_extension_names: extensions_array.as_ptr(),
    };
    unsafe {
        entry
            .create_instance(&creation_info, None)
            .expect("Failed to create instance")
    }
}
