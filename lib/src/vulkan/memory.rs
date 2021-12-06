use std::ptr;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};

#[derive(Debug)]
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub size: u64, //not necessarily the same as allocation size
    pub allocation: Allocation,
}

#[derive(Debug)]
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
}

pub struct MemoryManager {
    device: Arc<ash::Device>,
    frames_in_flight: u8,
    deferred_buffers: Vec<(AllocatedBuffer, u8)>,
    allocator: Allocator,
}

impl MemoryManager {
    pub fn new(
        instance: &ash::Instance,
        device: Arc<ash::Device>,
        physical_device: vk::PhysicalDevice,
        frames_in_flight: u8,
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
            buffer_device_address: false,
        };
        let allocator = Allocator::new(&acd).expect("Failed to create memory allocator");
        MemoryManager {
            device,
            frames_in_flight,
            deferred_buffers: Vec::new(),
            allocator,
        }
    }

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
            allocation,
        }
    }

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
            allocation,
        }
    }

    pub fn free_image(&mut self, image: AllocatedImage) {
        unsafe { self.device.destroy_image_view(image.image_view, None) };
        unsafe { self.device.destroy_image(image.image, None) };
        if self.allocator.free(image.allocation).is_err() {
            log::warn!("Failed to free memory");
        }
    }

    pub fn free_buffer(&mut self, buf: AllocatedBuffer) {
        unsafe { self.device.destroy_buffer(buf.buffer, None) };
        if self.allocator.free(buf.allocation).is_err() {
            log::warn!("Failed to free memory");
        }
    }

    pub fn deferred_free_buffer(&mut self, buf: AllocatedBuffer) {
        self.deferred_buffers.push((buf, 0));
    }

    pub fn frame_end_clean(&mut self) {
        if !self.deferred_buffers.is_empty() {
            let mut retain = Vec::with_capacity(self.deferred_buffers.len());
            let mut drop = Vec::with_capacity(self.deferred_buffers.len());
            while let Some((buf, mut frame_count)) = self.deferred_buffers.pop() {
                frame_count += 1;
                if frame_count > self.frames_in_flight {
                    drop.push(buf);
                } else {
                    retain.push((buf, frame_count));
                }
            }
            drop.into_iter().for_each(|buf| self.free_buffer(buf));
            self.deferred_buffers = retain;
        }
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        while let Some((buf, _)) = self.deferred_buffers.pop() {
            self.free_buffer(buf);
        }
    }
}
