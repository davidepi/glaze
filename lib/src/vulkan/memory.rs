use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use std::sync::{Arc, Mutex};
use std::{fmt, ptr};

/// An allocated buffer-memory pair.
pub struct AllocatedBuffer {
    /// Handle for the raw buffer
    pub buffer: vk::Buffer,
    /// Size of the buffer
    pub size: u64, //not necessarily the same as allocation size
    /// Allocated area on the memory
    allocation: Option<Allocation>,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<Allocator>>,
}

impl AllocatedBuffer {
    /// Returns a reference to the allocated memory.
    pub fn allocation(&self) -> &Allocation {
        unsafe {
            // SAFETY: the allocation Option is None only in the Drop method
            self.allocation.as_ref().unwrap_unchecked()
        }
    }
}

impl std::fmt::Debug for AllocatedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllocatedBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("allocation", &self.allocation)
            .finish()
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_buffer(self.buffer, None) };
        if let Ok(mut allocator) = self.allocator.lock() {
            // SAFETY: the allocation Option is None only after this block
            let allocation = unsafe { self.allocation.take().unwrap_unchecked() };
            if allocator.free(allocation).is_err() {
                log::warn!("Failed to free memory");
            }
        }
    }
}

/// An allocated buffer-image-imageview tuple.
pub struct AllocatedImage {
    /// Handle for the raw image
    pub image: vk::Image,
    /// Handle for the raw image view
    pub image_view: vk::ImageView,
    /// Allocated area on the memory
    allocation: Option<Allocation>,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<Allocator>>,
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image_view(self.image_view, None) };
        unsafe { self.device.destroy_image(self.image, None) };
        if let Ok(mut allocator) = self.allocator.lock() {
            // SAFETY: the allocation Option is None only after this block
            let allocation = unsafe { self.allocation.take().unwrap_unchecked() };
            if allocator.free(allocation).is_err() {
                log::warn!("Failed to free memory");
            }
        }
    }
}

impl std::fmt::Debug for AllocatedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllocatedImage")
            .field("image", &self.image)
            .field("image_view", &self.image_view)
            .field("allocation", &self.allocation)
            .finish()
    }
}

/// Manages allocations performed on the GPU.
#[derive(Clone)]
pub struct MemoryManager {
    allocator: Arc<Mutex<Allocator>>,
    device: Arc<ash::Device>,
}

impl MemoryManager {
    /// Creates a new memory manager for the given physical device.
    /// The `frames_in_flight` parameter is used to determine when it is appropriate to free
    /// deferred buffers.
    pub fn new(
        instance: &ash::Instance,
        device: Arc<ash::Device>,
        physical_device: vk::PhysicalDevice,
        buffer_device_address: bool,
    ) -> Self {
        let debug_settings = if cfg!(debug_assertions) {
            AllocatorDebugSettings {
                log_memory_information: true,
                log_leaks_on_shutdown: true,
                store_stack_traces: false,
                log_allocations: true,
                log_frees: true,
                log_stack_traces: false,
            }
        } else {
            AllocatorDebugSettings {
                log_memory_information: false,
                log_leaks_on_shutdown: false,
                store_stack_traces: false,
                log_allocations: false,
                log_frees: false,
                log_stack_traces: false,
            }
        };
        let acd = AllocatorCreateDesc {
            instance: instance.clone(),
            device: (*device).clone(),
            physical_device,
            debug_settings,
            buffer_device_address,
        };
        let allocator = Arc::new(Mutex::new(
            Allocator::new(&acd).expect("Failed to create memory allocator"),
        ));
        MemoryManager { allocator, device }
    }

    /// Creates a new AllocatedBuffer with the given name, size, usage and location.
    pub fn create_buffer(
        &mut self,
        name: &'static str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> AllocatedBuffer {
        let buf_ci = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            size: size as u64,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let buffer =
            unsafe { self.device.create_buffer(&buf_ci, None) }.expect("Failed to create buffer");
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alloc_desc = AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
        };
        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&alloc_desc)
            .expect("Allocation failed. OOM?");
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .expect("Failed to bind buffer memory");
        AllocatedBuffer {
            buffer,
            size,
            allocation: Some(allocation),
            device: self.device.clone(),
            allocator: self.allocator.clone(),
        }
    }

    /// Creates a new AllocatedImage on the GPU memory.
    pub fn create_image_gpu(
        &mut self,
        name: &'static str,
        format: vk::Format,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> AllocatedImage {
        // this methods default to ImageType 2D so it's weird to ask for a Extent3D
        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let img_ci = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: extent3d,
            mip_levels,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
        };
        let image =
            unsafe { self.device.create_image(&img_ci, None) }.expect("Failed to create image");
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alloc_desc = AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
        };
        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&alloc_desc)
            .expect("Allocation failed. OOM?");
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .expect("Failed to bind image memory");
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        };
        let iw_ci = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping::default(),
            subresource_range,
        };
        let iw = unsafe { self.device.create_image_view(&iw_ci, None) }
            .expect("Failed to create image view");
        AllocatedImage {
            image,
            image_view: iw,
            allocation: Some(allocation),
            device: self.device.clone(),
            allocator: self.allocator.clone(),
        }
    }
}
