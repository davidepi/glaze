use super::debug::{cchars_to_string, ValidationLayers};
use super::memory::AllocatedBuffer;
use super::surface::Surface;
use super::AllocatedImage;
use crate::Pipeline;
use ash::vk;
use fnv::FnvHashMap;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashSet};
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::{Arc, Mutex};

#[derive(Copy, Clone)]
/// Represents a vulkan queue family
pub struct Queue {
    pub idx: u32,
    pub queue: vk::Queue,
}

/// A wrapper for a Vulkan device.
///
/// This struct wraps together a logical device, with some features and extensions, and a queue
/// family.
/// The device is enclosed within an [Arc] to be able to be shared between threads without
/// additional clones.
#[derive(Clone)]
pub struct Device {
    logical: Arc<ash::Device>,
    physical: Arc<PhysicalDevice>,
    graphic_queue: Queue,
    compute_queue: Queue,
    transfer_queue: Queue,
    immediate_fences: Arc<Mutex<Vec<vk::Fence>>>,
}

impl Device {
    /// Creates a new device which supports rendering on a presentation surface.
    ///
    /// Requires the raw Vulkan instance, a list of device extensions, a list of physical device
    /// features and the Surface to use.
    ///
    /// In addition to the requested extensions and features, this method will select a device that:
    /// - supports a depth buffer with format `D32_SFLOAT`,
    /// - supports a graphics, compute and transfer queues,
    /// - the graphics queue suports presentation to the input [Surface]
    ///
    /// Returns None if no devices are found.
    #[cfg(feature = "vulkan-interactive")]
    pub fn new_present(
        instance: &ash::Instance,
        ext: &[&'static CStr],
        features: vk::PhysicalDeviceFeatures,
        ext_features: Option<*const c_void>,
        surface: &Surface,
    ) -> Option<Self> {
        create_device(instance, ext, features, ext_features, Some(surface))
    }

    /// Creates a new device for computational purposes.
    ///
    /// Requires the raw Vulkan instance, a list of device extensions, and a list of physical device
    /// features
    ///
    /// Unlike the [new_present] method, this one will not check for a depth buffer support.
    /// However, the graphics queue and transfer queue support (and of course the compute queue) is
    /// still required, albeit without presentation support.
    ///
    /// Returns None if no devices are found.
    pub fn new_compute(
        instance: &ash::Instance,
        ext: &[&'static CStr],
        features: vk::PhysicalDeviceFeatures,
        ext_features: Option<*const c_void>,
    ) -> Option<Self> {
        create_device(instance, ext, features, ext_features, None)
    }

    /// Returns an atomic reference counted clone of the Vulkan logical device.
    pub fn logical_clone(&self) -> Arc<ash::Device> {
        self.logical.clone()
    }

    /// Returns the Vulkan logical device.
    pub fn logical(&self) -> &ash::Device {
        &self.logical
    }

    /// Returns the Vulkan physical device.
    pub fn physical(&self) -> &PhysicalDevice {
        &self.physical
    }

    /// Returns a queue family with graphics capabilities.
    ///
    /// If the device is created with the [Self::present] method, this queue is required to support
    /// also presentation to a surface.
    pub fn graphic_queue(&self) -> Queue {
        self.graphic_queue
    }

    /// Returns a queue family with compute capabilities.
    pub fn compute_queue(&self) -> Queue {
        self.compute_queue
    }

    /// Returns a queue family with transfer capabilities.
    pub fn transfer_queue(&self) -> Queue {
        self.transfer_queue
    }

    /// Executes a closure with the Vulkan logical device.
    ///
    /// The closure is executed asynchronously in the command buffer `cmd` and returns a fence as a
    /// result.
    /// In order to ensure completion, one has to call [Device::wait_completion] passing the
    /// fence received from the current function.
    #[must_use]
    pub fn immediate_execute<F>(
        &self,
        cmd: vk::CommandBuffer,
        queue: Queue,
        command: F,
    ) -> vk::Fence
    where
        F: Fn(&ash::Device, vk::CommandBuffer),
    {
        let fence = self
            .immediate_fences
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| {
                let ci = vk::FenceCreateInfo {
                    s_type: vk::StructureType::FENCE_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: vk::FenceCreateFlags::empty(),
                };
                unsafe { self.logical.create_fence(&ci, None) }.expect("Failed to create fence")
            });
        let cmd_begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),
        };
        let cmds = [cmd];
        let submit_ci = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: cmds.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        unsafe {
            self.logical
                .begin_command_buffer(cmd, &cmd_begin)
                .expect("Failed to begin command");
            command(&self.logical, cmd);
            self.logical
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");
            self.logical
                .queue_submit(queue.queue, &[submit_ci], fence)
                .expect("Failed to submit to queue");
        }
        fence
    }

    /// Waits for the completion of a [Device::immediate_execute] function.
    pub fn wait_completion(&self, tokens: &[vk::Fence]) {
        unsafe {
            self.logical
                .wait_for_fences(tokens, true, u64::MAX)
                .expect("Failed to wait on fences");
            self.logical
                .reset_fences(tokens)
                .expect("Failed to reset fence");
            self.immediate_fences.lock().unwrap().extend(tokens);
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            if Arc::strong_count(&self.immediate_fences) == 1 {
                self.immediate_fences
                    .lock()
                    .unwrap()
                    .drain(..)
                    .for_each(|fence| {
                        self.logical.destroy_fence(fence, None);
                    });
            }
            if Arc::strong_count(&self.logical) == 1 {
                self.logical.destroy_device(None);
            }
        }
    }
}

pub fn create_device(
    instance: &ash::Instance,
    ext: &[&'static CStr],
    features: vk::PhysicalDeviceFeatures,
    ext_features: Option<*const c_void>,
    surface: Option<&Surface>,
) -> Option<Device> {
    // ext features are not checked, just the extensions
    let maybe_physical = PhysicalDevice::list_all(instance)
        .into_iter()
        .filter(|x| {
            if let Some(surface) = surface {
                x.surface_capabilities(surface)
                    .has_formats_and_present_modes()
            } else {
                true
            }
        })
        .filter(|device| device_supports_requested_extensions(instance, ext, device.device))
        .filter(|device| device_support_queues(instance, device.device, surface))
        .filter(|device| device_supports_features(device, features))
        .filter(|device| {
            if surface.is_some() {
                device_supports_depth_buffer(instance, vk::Format::D32_SFLOAT, device.device)
            } else {
                true
            }
        })
        .last();
    if let Some(physical) = maybe_physical {
        let queue_families = get_queues(instance, physical.device, surface);
        let all_queues = assign_queue_index(queue_families);
        let logical =
            create_logical_device(instance, ext, &physical, features, ext_features, all_queues);
        let gq = unsafe { logical.get_device_queue(all_queues[0].0, all_queues[0].1) };
        let cq = unsafe { logical.get_device_queue(all_queues[1].0, all_queues[1].1) };
        let tq = unsafe { logical.get_device_queue(all_queues[2].0, all_queues[2].1) };
        let graphic_queue = Queue {
            idx: all_queues[0].0,
            queue: gq,
        };
        let compute_queue = Queue {
            idx: all_queues[1].0,
            queue: cq,
        };
        let transfer_queue = Queue {
            idx: all_queues[2].0,
            queue: tq,
        };
        let ci = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::empty(),
        };
        let immediate_fences = Arc::new(Mutex::new(
            (0..10)
                .map(|_| {
                    unsafe { logical.create_fence(&ci, None) }.expect("Failed to create fence")
                })
                .collect::<Vec<_>>(),
        ));
        Some(Device {
            logical: Arc::new(logical),
            physical: Arc::new(physical),
            graphic_queue,
            compute_queue,
            transfer_queue,
            immediate_fences,
        })
    } else {
        None
    }
}

/// Temporary stores buffers while the GPU is still executing.
///
/// Sometimes, especially when copying from CPU to GPU buffers, a buffer can not be deallocated
/// until the GPU finishes executing. This may be a problem if the buffer goes out of scope as it
/// prevents assignment of new tasks to the GPU. This struct can be used to temporarily store these
/// buffers, assign multiple tasks to the GPU and waits for them all at once. The buffers are
/// dropped when this struct goes out of scope.
pub struct UnfinishedExecutions<'device> {
    /// Fences to be waited on.
    fences: Vec<vk::Fence>,
    /// Buffers that are to be freed after waiting on the fences.
    buffers_to_free: Vec<AllocatedBuffer>,
    /// Images that are to be freed after waiting on the fences.
    images_to_free: Vec<AllocatedImage>,
    /// Pipelines that will be freed after waiting on the fences.
    pipelines: Vec<Pipeline>,
    /// device to be waited on.
    device: &'device Device,
}

impl<'device> UnfinishedExecutions<'device> {
    /// Creates a new instance of this struct.
    pub fn new(device: &'device Device) -> Self {
        UnfinishedExecutions {
            fences: Vec::new(),
            buffers_to_free: Vec::new(),
            images_to_free: Vec::new(),
            pipelines: Vec::new(),
            device,
        }
    }

    /// Adds a fence only, without any buffer.
    pub fn add_fence(&mut self, fence: vk::Fence) {
        self.fences.push(fence);
    }

    /// Adds a buffer only, without any fence.
    pub fn add_buffer(&mut self, buffer: AllocatedBuffer) {
        self.buffers_to_free.push(buffer);
    }

    /// Adds an image only, without any fence.
    pub fn add_image(&mut self, image: AllocatedImage) {
        self.images_to_free.push(image);
    }

    /// Add the fence and corresponding pipeline executing.
    pub fn add_pipeline_execution(&mut self, fence: vk::Fence, pipeline: Pipeline) {
        self.fences.push(fence);
        self.pipelines.push(pipeline);
    }

    /// Add a fence and the corresponding buffer.
    pub fn add(&mut self, fence: vk::Fence, buffer: AllocatedBuffer) {
        self.fences.push(fence);
        self.buffers_to_free.push(buffer);
    }

    /// Waits for the GPU to finish all the jobs (assigned to this struct) and drops the buffers.
    pub fn wait_completion(self) {
        // force going out of scope
    }
}

impl Drop for UnfinishedExecutions<'_> {
    fn drop(&mut self) {
        if !self.fences.is_empty() {
            self.device.wait_completion(&self.fences);
        }
    }
}

/// Features of the current surface
pub struct SurfaceSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupport {
    /// Returns true if the surface supports at least one format and one present mode.
    pub fn has_formats_and_present_modes(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
}

/// Information about a device in String form.
pub struct DeviceInfo {
    /// Device name.
    pub name: String,
    /// Vulkan API version in form `Major.Minor.Patch`.
    pub vulkan_api_ver: String,
    /// Vendor name.
    pub vendor: &'static str,
    /// Driver version.
    pub driver_ver: String,
}

/// Decodes the driver version into a String.
fn driver_conversion(version: u32, vendor: u32) -> String {
    // nvidia
    if vendor == 0x10DE {
        let major = (version >> 22) & 0x3FF;
        let minor = (version >> 14) & 0xFF;
        let secondary = (version >> 6) & 0xFF;
        let tertiary = version & 0x3F;
        format!("{}.{}.{}.{}", major, minor, secondary, tertiary)
    } else if cfg!(target_os = "windows") && vendor == 0x8086 {
        // intel on windows
        let major = version >> 14;
        let minor = version & 0x3FFF;
        format!("{}.{}", major, minor)
    } else {
        // follow the vulkan convention
        let major = version >> 22;
        let minor = version >> 12 & 0x3FF;
        let patch = version & 0xFFF;
        format!("{}.{}.{}", major, minor, patch)
    }
}

/// Decodes the vendor id into a string.
fn vendor_id_conversion(id: u32) -> &'static str {
    match id {
        0x1002 => "AMD",
        0x1010 => "ImgTec",
        0x10DE => "Nvidia",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x8086 => "Intel",
        _ => "",
    }
}

/// Wrapper for a physical device, with its properties and features.
#[derive(Debug, Clone)]
pub struct PhysicalDevice {
    pub device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
}

impl PhysicalDevice {
    /// Lists all the physical devices on the system.
    pub fn list_all(instance: &ash::Instance) -> Vec<Self> {
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.expect("No physical devices found");
        let mut wscores = physical_devices
            .into_iter()
            .map(|device| rate_physical_device_suitability(instance, device))
            .filter(|(score, _)| *score > 0)
            .collect::<Vec<_>>();
        wscores.sort_by_key(|(score, _)| *score);
        wscores.into_iter().map(|(_, device)| device).collect()
    }

    /// Retrieves the surface capabilities for a given surface and the current physical device.
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

    /// Returns information about a physical device.
    pub fn info(&self) -> DeviceInfo {
        let props = self.properties;
        let vulkan_api_ver = driver_conversion(props.api_version, 0);
        let driver_ver = driver_conversion(props.driver_version, props.vendor_id);
        let name = cchars_to_string(&props.device_name);
        let vendor = vendor_id_conversion(self.properties.vendor_id);
        DeviceInfo {
            name,
            vulkan_api_ver,
            vendor,
            driver_ver,
        }
    }
}

/// Assigns a score to the physical device in the order discrete, integrated, and cpu
fn rate_physical_device_suitability(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> (u32, PhysicalDevice) {
    let vkinstance = &instance;
    let device_properties = unsafe { vkinstance.get_physical_device_properties(physical_device) };
    let device_features = unsafe { vkinstance.get_physical_device_features(physical_device) };
    let score = match device_properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
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

/// Returns true if the device supports the given extensions
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

/// Returns true if the device supports the given features
fn device_supports_features(device: &PhysicalDevice, features: vk::PhysicalDeviceFeatures) -> bool {
    // so, the features class has every feature as a field so it's a bit tricky to check in a loop.
    // instead, I will bitwise AND the device features and the requested features, and expect the
    // result to be equal to the requested features.
    let requested_features = unsafe {
        std::slice::from_raw_parts(
            (&features as *const vk::PhysicalDeviceFeatures) as *const u8,
            std::mem::size_of::<vk::PhysicalDeviceFeatures>(),
        )
    };
    let device_features = unsafe {
        std::slice::from_raw_parts(
            (&device.features as *const vk::PhysicalDeviceFeatures) as *const u8,
            std::mem::size_of::<vk::PhysicalDeviceFeatures>(),
        )
    };
    let result = requested_features
        .iter()
        .zip(device_features.iter())
        .map(|(x, y)| x & y)
        .collect::<Vec<_>>();
    result == requested_features
}

/// Returns true if the physical device supports the depth buffer.
fn device_supports_depth_buffer(
    instance: &ash::Instance,
    format: vk::Format,
    device: vk::PhysicalDevice,
) -> bool {
    let props = unsafe { instance.get_physical_device_format_properties(device, format) };
    props
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
}

/// Requires the device to support graphics, compute and transfer queues.
fn device_support_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: Option<&Surface>,
) -> bool {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let graphics_present_support = queue_families
        .iter()
        .enumerate()
        .filter(|(_, prop)| prop.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .any(|(index, _)| {
            if let Some(surface) = surface {
                unsafe {
                    surface.loader.get_physical_device_surface_support(
                        physical_device,
                        index as u32,
                        surface.surface,
                    )
                }
                .is_ok()
            } else {
                true
            }
        });
    let compute_support = queue_families
        .iter()
        .any(|prop| prop.queue_flags.contains(vk::QueueFlags::COMPUTE));
    let transfer_support = queue_families
        .iter()
        .any(|prop| prop.queue_flags.contains(vk::QueueFlags::TRANSFER));
    graphics_present_support && compute_support && transfer_support
}

/// Returns the queue family index for graphics, compute and tranfer queues.
/// Handle cases where each family index supports a single queue.
///
/// If the surface is Some, the graphics queue is required to support presentation.
fn get_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: Option<&Surface>,
) -> [u32; 3] {
    // filter by resources, takes the index that has the highest availability
    // intel graphics card can have 1 queue for queue family index, and all supports everything
    // nvidia has the graphics only on the some indices, but supports more queue
    let mut qf = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    // graphics queue + optional present to surface
    let graphics_qfi = qf
        .iter()
        .enumerate()
        .filter(|(_, &prop)| prop.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .filter(|(idx, _)| {
            if let Some(surface) = surface {
                unsafe {
                    surface.loader.get_physical_device_surface_support(
                        physical_device,
                        *idx as u32,
                        surface.surface,
                    )
                }
                .is_ok()
            } else {
                true
            }
        })
        .max_by_key(|(_, prop)| prop.queue_count)
        .expect("Not enough graphics queues in the device")
        .0;
    qf[graphics_qfi].queue_count -= 1;
    let compute_qfi = qf
        .iter()
        .enumerate()
        .filter(|(_, &prop)| prop.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .max_by_key(|(_, prop)| prop.queue_count)
        .expect("Not enough compute queues in the device")
        .0;
    qf[compute_qfi].queue_count -= 1;
    let transfer_qfi = qf
        .iter()
        .enumerate()
        .filter(|(_, &prop)| prop.queue_flags.contains(vk::QueueFlags::TRANSFER))
        .max_by_key(|(_, prop)| prop.queue_count)
        .expect("Not enough compute queues in the device")
        .0;
    qf[transfer_qfi].queue_count -= 1;
    [graphics_qfi as u32, compute_qfi as u32, transfer_qfi as u32]
}

/// Splits a list of queue families into a list of (family index, queue index) tuples
fn assign_queue_index<const FAMILIES: usize>(
    queue_families: [u32; FAMILIES],
) -> [(u32, u32); FAMILIES] {
    let mut used_queues = FnvHashMap::with_capacity_and_hasher(FAMILIES, Default::default());
    let mut retval = [(0, 0); FAMILIES];
    for (i, family_index) in queue_families.into_iter().enumerate() {
        let next = match used_queues.entry(family_index) {
            Entry::Occupied(mut entry) => {
                let next = (family_index, *entry.get());
                *entry.get_mut() += 1;
                next
            }
            Entry::Vacant(entry) => {
                let next = (family_index, 0);
                entry.insert(0);
                next
            }
        };
        retval[i] = next;
    }
    retval
}

/// Creates the logical device, using a given physical device, extensions, features and queue indices
fn create_logical_device(
    instance: &ash::Instance,
    ext: &[&'static CStr],
    device: &PhysicalDevice,
    features_requested: vk::PhysicalDeviceFeatures,
    ext_features: Option<*const c_void>,
    queues: [(u32, u32); 3],
) -> ash::Device {
    let validations = ValidationLayers::application_default();
    let physical_device = device.device;
    let unique_families = queues
        .iter()
        .map(|(f, _)| f)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut queue_create_infos = Vec::with_capacity(unique_families.len());
    let mut queue_priorities = vec![Vec::new(); unique_families.len()];
    for (unique_index, queue_family_index) in unique_families.into_iter().enumerate() {
        let count = queues
            .iter()
            .filter(|(f, _)| *f == queue_family_index)
            .count();
        // this is to avoid dropping the pointed address before creating the device
        queue_priorities[unique_index] = vec![1.0; count];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index,
            queue_count: count as u32,
            p_queue_priorities: queue_priorities[unique_index].as_ptr(),
        };
        queue_create_infos.push(queue_create_info);
    }
    let p_next = if let Some(ext) = ext_features {
        ext
    } else {
        ptr::null()
    };
    let required_device_extensions = ext.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
    let validations_arr = validations.as_ptr();
    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next,
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: validations_arr.len() as u32,
        pp_enabled_layer_names: validations_arr.as_ptr(),
        enabled_extension_count: required_device_extensions.len() as u32,
        pp_enabled_extension_names: required_device_extensions.as_ptr(),
        p_enabled_features: &features_requested,
    };
    unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .expect("Failed to create logical device")
}
