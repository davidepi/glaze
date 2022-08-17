use super::debug::{cchars_to_string, ValidationLayers};
use super::memory::AllocatedBuffer;
use super::surface::Surface;
use super::CommandManager;
use crate::Pipeline;
use ash::vk;
use fnv::FnvHashMap;
use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
/// Represents a vulkan queue family
pub struct Queue {
    /// Family index.
    idx: u32,
    /// Vulkan queue.
    /// Private in order to enforce the use of Device::execute and Device::immediate_execute and
    /// pass through the main_thread.
    queue: vk::Queue,
    /// The type of queue.
    capabilities: vk::QueueFlags,
    /// Used to track the queue usage
    exclusive: Arc<()>,
}

impl Queue {
    /// Returns the family index for the current queue.
    pub fn family(&self) -> u32 {
        self.idx
    }
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
    queues: Arc<Mutex<Vec<Queue>>>,
    // assign each queue only once
    submit_lock: Option<Arc<Mutex<()>>>,
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
    /// Unlike the `new_present` method, this one will not check for a depth buffer support.
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
    /// If the device is created with the `new_present` method, this queue is required to support
    /// also presentation to a surface.
    pub fn graphic_queue(&self) -> Queue {
        self.queue(vk::QueueFlags::GRAPHICS)
    }

    /// Returns a queue family with compute capabilities.
    pub fn compute_queue(&self) -> Queue {
        self.queue(vk::QueueFlags::COMPUTE)
    }

    /// Returns a queue family with transfer capabilities.
    pub fn transfer_queue(&self) -> Queue {
        self.queue(vk::QueueFlags::TRANSFER)
    }

    /// Returns a queue of the given type.
    /// Handles returning an exclusive queue or a shared queue based on the status of the
    /// submit_lock field in the device.
    fn queue(&self, flag: vk::QueueFlags) -> Queue {
        if self.submit_lock.is_none() {
            self.queues
                .lock()
                .unwrap()
                .iter()
                .filter(|q| q.capabilities.contains(flag))
                .find(|q| Arc::strong_count(&q.exclusive) == 1)
                .expect("Using more queues than requested")
                .clone()
        } else {
            self.queues
                .lock()
                .unwrap()
                .iter()
                .find(|q| q.capabilities.contains(flag))
                .unwrap()
                .clone()
        }
    }

    /// Submits a command to the be executed by the device immediately.
    ///
    /// Creates the SubmitInfo and Fence on the spot and returns the latter.
    ///
    /// In order to ensure completion, one has to call [Device::wait_completion] passing the
    /// fence received from the current function.
    #[must_use]
    pub fn submit_immediate<F>(&self, cmdm: &mut CommandManager, command: F) -> vk::Fence
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
        let cmd = cmdm.get_cmd_buffer();
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
        }
        self.submit(cmdm, &[submit_ci], fence);
        fence
    }

    /// Submits a task with the given submit infos and fence to use.
    pub fn submit(&self, cmdm: &CommandManager, si: &[vk::SubmitInfo], fence: vk::Fence) {
        let queue = cmdm.queue();
        let _lock;
        if let Some(submit_lock) = &self.submit_lock {
            _lock = submit_lock.lock().unwrap();
        }
        unsafe { self.logical.queue_submit(queue.queue, si, fence) }
            .expect("Failed to submit to queue");
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

/// Creates a Device that wraps a logical device and a physical device.
/// If prenseting to a surface is required, the surface must be passed to the function.
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
        let mut submit_lock = None;
        // try to reserve 3 queues for each type
        let mut queue_fams = get_queue_families(instance, physical.device, [3; 3], surface);
        if queue_fams.is_empty() {
            // if it fails, just use a single queue per type and use mutexes
            queue_fams = get_queue_families(instance, physical.device, [1; 3], surface);
            submit_lock = Some(Arc::new(Mutex::new(())));
        }
        let logical = create_logical_device(
            instance,
            ext,
            &physical,
            features,
            ext_features,
            &queue_fams,
        );
        let queues = Arc::new(Mutex::new(create_queues(&logical, &queue_fams)));
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
            immediate_fences,
            submit_lock,
            queues,
        })
    } else {
        None
    }
}

/// Temporary stores buffers while the GPU is still executing.
///
/// Sometimes, especially when copying from CPU to GPU buffers, a buffer can not be deallocated
/// until the GPU finishes executing. This may be a problem if the buffer goes out of scope as it
/// will result in errors. This struct can be used to temporarily store these buffers,
/// assign multiple tasks to the GPU and waits for them all at once. The buffers are
/// dropped when this struct goes out of scope.
pub struct UnfinishedExecutions<'device> {
    /// Fences to be waited on.
    fences: Vec<vk::Fence>,
    /// Buffers that are to be freed after waiting on the fences.
    buffers_to_free: Vec<AllocatedBuffer>,
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

/// Attempts to reserve multiple queues and returns tuples (family index, amount of queues,
/// family type).
///
/// If not enough queues are available, returns empty vector.
///
/// `requested` contains the amount of requested queues (if available) in the order [GRAPHICS,
/// COMPUTE, TRANSFER].
///
/// If the surface is Some, the graphics queue is required to support presentation.
fn get_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    requested: [u32; 3],
    surface: Option<&Surface>,
) -> Vec<(u32, u32, vk::QueueFlags)> {
    // intel graphics card can have 1 queue for queue family index, and all supports everything
    // nvidia has the graphics only on the some indices, but supports more queues
    let mut props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut retval = FnvHashMap::default();
    // filter by queue flag, takes the family index that has the highest availability
    let transfer_index = props
        .iter()
        .enumerate()
        .filter(|(_, prop)| prop.queue_flags.contains(vk::QueueFlags::TRANSFER))
        .max_by_key(|(_, prop)| prop.queue_count)
        .map(|prop| prop.0);
    if let Some(family_index) = transfer_index {
        // try to reserve the requested amount of queues
        if props[family_index].queue_count >= requested[2] {
            // reduce the amount of available queues and insert the reserved into the map
            // the procedure for compute and graphics queue is the same
            retval
                .entry(family_index)
                .and_modify(|(count, _)| *count += requested[2])
                .or_insert((requested[2], props[family_index].queue_flags));
            props[family_index].queue_count -= requested[2];
        } else {
            return Vec::new();
        }
    } else {
        return Vec::new();
    }
    // repeat the same for compute flags
    let compute_index = props
        .iter()
        .enumerate()
        .filter(|(_, &prop)| prop.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .max_by_key(|(_, prop)| prop.queue_count)
        .map(|prop| prop.0);
    if let Some(family_index) = compute_index {
        // try to reserve the requested amount of queues
        if props[family_index].queue_count >= requested[1] {
            // reduce the amount of available queues and insert the reserved into the map
            // the procedure for compute and graphics queue is the same
            props[family_index].queue_count -= requested[1];
            retval
                .entry(family_index)
                .and_modify(|(count, _)| *count += requested[1])
                .or_insert((requested[1], props[family_index].queue_flags));
        } else {
            return Vec::new();
        }
    } else {
        return Vec::new();
    }
    // same for graphics queue but ensures present to surface capabilities if surface is Some
    let graphic_index = props
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
        .map(|prop| prop.0);
    if let Some(family_index) = graphic_index {
        if props[family_index].queue_count >= requested[0] {
            props[family_index].queue_count -= requested[0];
            retval
                .entry(family_index)
                .and_modify(|(count, _)| *count += requested[0])
                .or_insert((requested[0], props[family_index].queue_flags));
        } else {
            return Vec::new();
        }
    } else {
        return Vec::new();
    }
    retval
        .into_iter()
        .map(|(family_index, (count, flags))| (family_index as u32, count, flags))
        .collect()
}

fn create_queues(device: &ash::Device, queues: &[(u32, u32, vk::QueueFlags)]) -> Vec<Queue> {
    let mut retval = Vec::new();
    for (queue_family_index, amount, flags) in queues.iter().copied() {
        for queue_index in 0..amount {
            let vkqueue = unsafe { device.get_device_queue(queue_family_index, queue_index) };
            let queue = Queue {
                idx: queue_family_index,
                queue: vkqueue,
                capabilities: flags,
                exclusive: Arc::new(()),
            };
            retval.push(queue);
        }
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
    queues: &[(u32, u32, vk::QueueFlags)],
) -> ash::Device {
    let validations = ValidationLayers::application_default();
    let physical_device = device.device;
    let mut queue_create_infos = Vec::with_capacity(queues.len());
    let mut queue_priorities = vec![Vec::new(); queues.len()];
    for (i, (queue_family_index, queue_count, _)) in queues.iter().cloned().enumerate() {
        // this is to avoid dropping the pointed address before creating the device
        queue_priorities[i] = vec![1.0; queue_count as usize];
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index,
            queue_count,
            p_queue_priorities: queue_priorities[i].as_ptr(),
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

#[cfg(feature = "vulkan-interactive")]
// swapchain needs to access some private fields of Queue (vk::Queue).
// vk::Queue can not be public, otherwise the Device::submit may be bypassed and the submit
// performed in a non-main thread.
pub mod swapchain {
    use crate::vulkan::instance::Instance;
    use crate::vulkan::renderpass::FinalRenderPass;
    use crate::vulkan::sync::PresentFrameSync;
    use crate::vulkan::CommandManager;
    use crate::PresentInstance;
    use ash::vk;
    use std::ptr;
    use std::sync::Arc;

    /// The swapchain image used as target for the current frame
    pub struct AcquiredImage<'a> {
        /// Frame index (in the swapchain images array)
        pub index: u32,
        /// Reference to the Final
        pub renderpass: &'a FinalRenderPass,
    }

    /// Wrapper for the swapchain, its image views, and the render pass writing to the images.
    pub struct Swapchain {
        swapchain: vk::SwapchainKHR,
        loader: ash::extensions::khr::Swapchain,
        extent: vk::Extent2D,
        render_passes: Vec<FinalRenderPass>,
        instance: Arc<PresentInstance>,
    }

    impl Swapchain {
        /// Creates a new swapchain with the given size.
        pub fn create(instance: Arc<PresentInstance>, width: u32, height: u32) -> Self {
            swapchain_init(instance, width, height, None)
        }

        /// Recreate the current swapchain with a new size.
        pub fn recreate(&mut self, width: u32, height: u32) {
            destroy(self, true);
            let new = swapchain_init(self.instance.clone(), width, height, Some(self.swapchain));
            *self = new;
        }

        /// Returns the swapchain image extent.
        pub fn extent(&self) -> vk::Extent2D {
            self.extent
        }

        /// Returns the swapchain raw handle.
        pub fn raw_handle(&self) -> vk::SwapchainKHR {
            self.swapchain
        }

        /// Returns the render pass writing to the swapchain image.
        pub fn renderpass(&self) -> vk::RenderPass {
            // all render passes are essentially the same
            self.render_passes[0].renderpass
        }

        /// Presents an image to the surface.
        pub fn queue_present(&self, cmdm: &CommandManager, present_info: &vk::PresentInfoKHR) {
            let queue = cmdm.queue();
            unsafe {
                match self.loader.queue_present(queue.queue, present_info) {
                    Ok(_) => (),
                    Err(_val @ vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        log::debug!("Refusing to present out of date swapchain");
                    }
                    _ => panic!("Failed to acquire next image"),
                }
            }
        }

        /// Acquires the next swapchain frame to be renderered to.
        pub fn acquire_next_image<'a>(
            &'a self,
            sync: &PresentFrameSync,
        ) -> Option<AcquiredImage<'a>> {
            let acquired = unsafe {
                self.loader.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    sync.image_available,
                    vk::Fence::null(),
                )
            };
            match acquired {
                Ok((index, _)) => Some(AcquiredImage {
                    index,
                    renderpass: &self.render_passes[index as usize],
                }),
                Err(_val @ vk::Result::ERROR_OUT_OF_DATE_KHR) => None,
                _ => panic!("Failed to acquire next image"),
            }
        }
    }

    /// Destroys a swapchain.
    /// The partial parameter is used to indicate that the swapchain will be resized (keeping the raw
    /// handle as parameter for the ash::vk::SwapchainCreateInfoKHR).
    fn destroy(sc: &mut Swapchain, partial: bool) {
        unsafe {
            if !partial {
                sc.loader.destroy_swapchain(sc.swapchain, None);
            }
        }
    }

    impl Drop for Swapchain {
        fn drop(&mut self) {
            destroy(self, false);
        }
    }

    /// Creates a new swapchain with the given size. If an old swapchain is present, it will be recycled
    /// for a faster initialization.
    fn swapchain_init(
        instance: Arc<PresentInstance>,
        width: u32,
        height: u32,
        old: Option<vk::SwapchainKHR>,
    ) -> Swapchain {
        let surface_cap = instance.surface_capabilities();
        let device = instance.device();
        let capabilities = surface_cap.capabilities;
        let format = surface_cap
            .formats
            .iter()
            .find(|sf| {
                sf.format == vk::Format::B8G8R8A8_SRGB
                    && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| {
                let default = surface_cap.formats.first().unwrap();
                log::warn!("Failed to find suitable surface format, using the first one available");
                default
            });
        let present_mode = *surface_cap
            .present_modes
            .iter()
            .find(|&x| *x == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO);
        let extent = if surface_cap.capabilities.current_extent.width != u32::MAX {
            surface_cap.capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };
        let image_count = if surface_cap.capabilities.max_image_count == 0 {
            surface_cap.capabilities.min_image_count + 1
        } else {
            surface_cap
                .capabilities
                .max_image_count
                .max(surface_cap.capabilities.min_image_count + 1)
        };
        let ci = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: instance.surface().surface,
            min_image_count: image_count,
            image_format: format.format,
            image_color_space: format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: old.unwrap_or_else(vk::SwapchainKHR::null),
        };
        let loader = ash::extensions::khr::Swapchain::new(instance.instance(), device.logical());
        let swapchain =
            unsafe { loader.create_swapchain(&ci, None) }.expect("Failed to create swapchain");
        let images =
            unsafe { loader.get_swapchain_images(swapchain) }.expect("Failed to get images");
        let image_views = images
            .iter()
            .map(|i| create_image_views(device.logical(), *i, format.format))
            .collect::<Vec<_>>();
        let render_passes = images
            .into_iter()
            .zip(image_views)
            .map(|(i, iw)| {
                FinalRenderPass::new(device.logical_clone(), format.format, i, iw, extent)
            })
            .collect();
        Swapchain {
            swapchain,
            loader,
            extent: ci.image_extent,
            render_passes,
            instance,
        }
    }

    fn create_image_views(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> vk::ImageView {
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let iw_ci = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping::default(),
            subresource_range,
        };
        unsafe { device.create_image_view(&iw_ci, None) }.expect("Failed to create Image View")
    }
}
