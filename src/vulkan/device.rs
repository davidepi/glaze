use super::validations::cchars_to_string;
use crate::vulkan::instance::Surface;
use ash::vk;
use std::{collections::HashSet, ffi::CStr};
struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl QueueFamilyIndices {
    fn as_array(&self) -> [u32; 2] {
        [self.graphics_family, self.present_family]
    }
}

struct PhysicalDevice {
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    features: vk::PhysicalDeviceFeatures,
    queue_indices: QueueFamilyIndices,
}

impl PhysicalDevice {
    pub fn new(instance: &ash::Instance, surface: &Surface, extensions: &[&'static CStr]) -> Self {
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.expect("No physical devices found");
        physical_devices
            .into_iter()
            .map(|device| rate_physical_device_suitability(instance, device, surface, extensions))
            .flatten()
            .max_by_key(|(score, _)| *score)
            .expect("No compatible physical devices found")
            .1
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
        vk::PhysicalDeviceType::OTHER => 10,
        _ => 10,
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
