use super::debug::{cchars_to_string, ValidationLayers};
use crate::vulkan::instance::Surface;
use ash::vk;
use std::{
    collections::{BTreeSet, HashSet},
    ffi::CStr,
    ptr,
};

pub struct SurfaceSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupport {
    pub fn has_formats_and_present_modes(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
}

struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl QueueFamilyIndices {
    fn as_array(&self) -> [u32; 2] {
        [self.graphics_family, self.present_family]
    }
}

pub struct PhysicalDevice {
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    features: vk::PhysicalDeviceFeatures,
    queue_indices: QueueFamilyIndices,
}

impl PhysicalDevice {
    pub fn list_compatible(
        instance: &ash::Instance,
        surface: &Surface,
        extensions: &[&'static CStr],
    ) -> Vec<Self> {
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.expect("No physical devices found");
        let mut wscores = physical_devices
            .into_iter()
            .map(|device| rate_physical_device_suitability(instance, device, surface, extensions))
            .flatten()
            .filter(|(score, _)| *score > 0)
            .collect::<Vec<_>>();
        wscores.sort_by_key(|(score, _)| *score);
        wscores.into_iter().map(|(_, device)| device).collect()
    }

    pub fn surface_capabilities(&self, surface: &Surface) -> SurfaceSupport {
        let capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(self.device, surface.surface)
        }
        .expect("could not get surface capabilities");
        let formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(self.device, surface.surface)
        }
        .expect("Could not get surface formats");
        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(self.device, surface.surface)
        }
        .expect("Failed to get present modes");
        SurfaceSupport {
            capabilities,
            formats,
            present_modes,
        }
    }
}

fn rate_physical_device_suitability(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
    extensions: &[&'static CStr],
) -> Option<(u32, PhysicalDevice)> {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_features = unsafe { instance.get_physical_device_features(physical_device) };
    let score = match device_properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
        vk::PhysicalDeviceType::CPU => 1,
        vk::PhysicalDeviceType::OTHER | _ => 0,
    };
    if device_supports_requested_extensions(instance, physical_device, extensions) {
        let queue_family = find_queue_families(instance, physical_device, surface);
        if let Some(queue_family) = queue_family {
            Some((
                score,
                PhysicalDevice {
                    device: physical_device,
                    properties: device_properties,
                    features: device_features,
                    queue_indices: queue_family,
                },
            ))
        } else {
            None
        }
    } else {
        None
    }
}

fn device_supports_requested_extensions(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    requested_extensions: &[&'static CStr],
) -> bool {
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(device) }
        .expect("Failed to get device extensions")
        .into_iter()
        .map(|x| cchars_to_string(&x.extension_name))
        .collect::<HashSet<_>>();
    !requested_extensions
        .iter()
        .map(|x| x.to_str().unwrap().to_string())
        .any(|x| !available_extensions.contains(&x))
}

fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> Option<QueueFamilyIndices> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let graphics_family = queue_families
        .iter()
        .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    let present_family = (0..queue_families.len()).into_iter().find(|x| unsafe {
        surface
            .loader
            .get_physical_device_surface_support(physical_device, *x as u32, surface.surface)
            .unwrap()
    });
    if let (Some(graphics_family), Some(present_family)) = (graphics_family, present_family) {
        Some(QueueFamilyIndices {
            graphics_family: graphics_family as u32,
            present_family: present_family as u32,
        })
    } else {
        None
    }
}

pub fn create_logical_device(
    instance: &ash::Instance,
    device: &PhysicalDevice,
    extensions: &[&'static CStr],
) -> ash::Device {
    let validations = ValidationLayers::application_default();
    let physical_device = device.device;
    let queue_families = &device.queue_indices;
    let queue_families_set = queue_families
        .as_array()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut queue_create_infos = Vec::with_capacity(queue_families_set.len());
    let queue_priorities = [1.0];
    for queue_index in queue_families_set {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_index,
            queue_count: queue_priorities.len() as u32,
            p_queue_priorities: queue_priorities.as_ptr(),
        };
        queue_create_infos.push(queue_create_info);
    }
    //TODO: find a way to set this array (when I have a clearer vision of the overall structure)
    let mut physical_features = vk::PhysicalDeviceFeatures {
        ..Default::default()
    };
    let required_device_extensions = extensions
        .into_iter()
        .map(|x| x.as_ptr())
        .collect::<Vec<_>>();
    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: validations.len() as u32,
        pp_enabled_layer_names: validations.as_ptr(),
        enabled_extension_count: required_device_extensions.len() as u32,
        pp_enabled_extension_names: required_device_extensions.as_ptr(),
        p_enabled_features: &physical_features,
    };
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .expect("Failed to create logical device");
    device
}
