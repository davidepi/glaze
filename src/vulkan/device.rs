use super::debug::cchars_to_string;
use super::debug::ValidationLayers;
use super::surface::Surface;
use ash::vk;
use std::collections::HashSet;
use std::ffi::CStr;
use std::ptr;

pub trait Device {
    fn logical(&self) -> &ash::Device;
    fn physical(&self) -> &PhysicalDevice;
}

pub struct PresentDevice {
    logical: ash::Device,
    physical: PhysicalDevice,
    graphic_index: u32,
    graphic_pool: vk::CommandPool,
    graphic_queue: vk::Queue,
}

impl PresentDevice {
    pub fn new(instance: &ash::Instance, ext: &[&'static CStr], surface: &Surface) -> Self {
        let physical = PhysicalDevice::list_all(instance, surface)
            .into_iter()
            .filter(|x| {
                x.surface_capabilities(surface)
                    .has_formats_and_present_modes()
            })
            .filter(|device| device_supports_requested_extensions(instance, ext, device.device))
            .filter(|device| graphics_present_index(instance, device.device, surface).is_some())
            .last()
            .expect("No compatible devices found");
        let graphic_index = graphics_present_index(instance, physical.device, surface).unwrap();
        let all_queues = vec![graphic_index];
        let logical = create_logical_device(instance, ext, &physical, &all_queues);
        let graphic_pool = create_command_pool(&logical, graphic_index);
        let graphic_queue = unsafe { logical.get_device_queue(graphic_index, 0) };
        PresentDevice {
            logical,
            physical,
            graphic_index,
            graphic_pool,
            graphic_queue,
        }
    }

    pub fn create_command_buffers(&self, count: u32) -> Vec<vk::CommandBuffer> {
        create_command_buffers(&self.logical, self.graphic_pool, count)
    }

    pub fn destroy_command_buffers(&self, buffers: &[vk::CommandBuffer]) {
        unsafe {
            self.logical
                .free_command_buffers(self.graphic_pool, &buffers);
        }
    }

    pub fn graphic_queue(&self) -> vk::Queue {
        self.graphic_queue
    }
}

impl Drop for PresentDevice {
    fn drop(&mut self) {
        unsafe {
            self.logical.destroy_command_pool(self.graphic_pool, None);
            self.logical.destroy_device(None);
        }
    }
}

impl Device for PresentDevice {
    fn logical(&self) -> &ash::Device {
        &self.logical
    }

    fn physical(&self) -> &PhysicalDevice {
        &self.physical
    }
}

pub struct SurfaceSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupport {
    pub fn has_formats_and_present_modes(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
}

pub struct PhysicalDevice {
    pub device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
}

impl PhysicalDevice {
    pub fn list_all(instance: &ash::Instance, surface: &Surface) -> Vec<Self> {
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.expect("No physical devices found");
        let mut wscores = physical_devices
            .into_iter()
            .map(|device| rate_physical_device_suitability(instance, device, surface))
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
) -> (u32, PhysicalDevice) {
    let vkinstance = &instance;
    let device_properties = unsafe { vkinstance.get_physical_device_properties(physical_device) };
    let device_features = unsafe { vkinstance.get_physical_device_features(physical_device) };
    let score = match device_properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
        vk::PhysicalDeviceType::CPU => 1,
        vk::PhysicalDeviceType::OTHER | _ => 0,
    };
    (
        score,
        PhysicalDevice {
            device: physical_device,
            properties: device_properties,
            features: device_features,
        },
    )
}

fn device_supports_requested_extensions(
    instance: &ash::Instance,
    ext: &[&'static CStr],
    device: vk::PhysicalDevice,
) -> bool {
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(device) }
        .expect("Failed to get device extensions")
        .into_iter()
        .map(|x| cchars_to_string(&x.extension_name))
        .collect::<HashSet<_>>();
    !ext.iter()
        .map(|x| x.to_str().unwrap().to_string())
        .any(|x| !available_extensions.contains(&x))
}

fn graphics_present_index(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> Option<u32> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    queue_families
        .into_iter()
        .enumerate()
        .filter(|(index, prop)| prop.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .filter(|(index, prop)| {
            unsafe {
                surface.loader.get_physical_device_surface_support(
                    physical_device,
                    *index as u32,
                    surface.surface,
                )
            }
            .is_ok()
        })
        .map(|(index, _)| index as u32)
        .next()
}

pub fn create_logical_device(
    instance: &ash::Instance,
    ext: &[&'static CStr],
    device: &PhysicalDevice,
    queue_indices: &[u32],
) -> ash::Device {
    let validations = ValidationLayers::application_default();
    let physical_device = device.device;
    let mut queue_create_infos = Vec::with_capacity(queue_indices.len());
    let queue_priorities = [1.0];
    for queue_index in queue_indices {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: *queue_index,
            queue_count: queue_priorities.len() as u32,
            p_queue_priorities: queue_priorities.as_ptr(),
        };
        queue_create_infos.push(queue_create_info);
    }
    //TODO: find a way to set this array (when I have a clearer vision of the overall structure)
    let mut physical_features = vk::PhysicalDeviceFeatures {
        ..Default::default()
    };
    let required_device_extensions = ext.into_iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
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

fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> vk::CommandPool {
    let pool_ci = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        queue_family_index,
    };
    unsafe { device.create_command_pool(&pool_ci, None) }.expect("Failed to create command pool")
}

fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    let alloc_ci = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: count,
    };
    unsafe { device.allocate_command_buffers(&alloc_ci) }
        .expect("Failed to allocate command buffers")
}
