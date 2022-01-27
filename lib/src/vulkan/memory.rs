use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use std::sync::{Arc, Mutex};
use std::{fmt, ptr};

use super::cmd::CommandManager;
use super::device::Device;
use super::Instance;

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

impl AllocatedImage {
    /// exports a GPU image with SHADER_READ_ONLY_OPTIMAL layout to a linear array of pixel
    pub fn export(
        &self,
        device: &Device,
        tcmdm: &mut CommandManager,
        width: u16,
        height: u16,
    ) -> image::RgbaImage {
        // create the dst image with LINEAR tiling
        let extent3d = vk::Extent3D {
            width: width as u32,
            height: height as u32,
            depth: 1,
        };
        let img_ci = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: extent3d,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::LINEAR,
            usage: vk::ImageUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
        };
        let image =
            unsafe { self.device.create_image(&img_ci, None) }.expect("Failed to create image");
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alloc_desc = AllocationCreateDesc {
            name: "Export Image GPU",
            requirements,
            location: MemoryLocation::GpuToCpu,
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
        // prepare the blit subregion
        let blit_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let blit_regions = [vk::ImageBlit {
            src_subresource: blit_subresource,
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            ],
            dst_subresource: blit_subresource,
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            ],
        }];
        // prepare the barriers
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier_src_sampled_to_transfer = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::MEMORY_READ,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.image,
            subresource_range,
        };
        let barrier_src_transfer_to_sampled = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::TRANSFER_READ,
            dst_access_mask: vk::AccessFlags::MEMORY_READ,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.image,
            subresource_range,
        };
        let barrier_dst_undefined_to_transfer = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range,
        };
        let barrier_dst_transfer_to_general = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::MEMORY_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range,
        };
        // run the commands
        let cmd = tcmdm.get_cmd_buffer();
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[
                        barrier_src_sampled_to_transfer,
                        barrier_dst_undefined_to_transfer,
                    ],
                );
                device.cmd_blit_image(
                    cmd,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blit_regions,
                    vk::Filter::LINEAR,
                );
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[
                        barrier_src_transfer_to_sampled,
                        barrier_dst_transfer_to_general,
                    ],
                );
            }
        };
        let fence = device.immediate_execute(cmd, device.transfer_queue(), command);
        device.wait_completion(&[fence]);
        // now extract the bytes from the figure
        let size = (width * height * 4) as usize;
        let mut raw = vec![0_u8; size];
        let mapped = allocation
            .mapped_ptr()
            .expect("Failed to map image")
            .cast()
            .as_ptr();
        unsafe { std::ptr::copy_nonoverlapping(mapped, &mut raw, size) };
        image::RgbaImage::from_raw(width as u32, height as u32, raw)
            .expect("Image conversion failed")
    }
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
        &self,
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
        &self,
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
            linear: false,
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

/// Exports an image with R8G8B8A8 format only!
pub fn export<T: Instance + ?Sized>(
    instance: &T,
    src_image: &AllocatedImage,
    tcmdm: &mut CommandManager,
    width: u16,
    height: u16,
) -> image::RgbaImage {
    // create the dst image with LINEAR tiling
    let extent3d = vk::Extent3D {
        width: width as u32,
        height: height as u32,
        depth: 1,
    };
    let device_properties = unsafe {
        instance.instance().get_physical_device_format_properties(
            instance.device().physical().device,
            vk::Format::R8G8B8A8_SRGB,
        )
    };
    let supports_linear_blit = (device_properties.linear_tiling_features
        & vk::FormatFeatureFlags::BLIT_DST)
        == vk::FormatFeatureFlags::BLIT_DST;
    let img_ci = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ImageCreateFlags::empty(),
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_SRGB,
        extent: extent3d,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::LINEAR,
        usage: vk::ImageUsageFlags::TRANSFER_DST,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        initial_layout: vk::ImageLayout::UNDEFINED,
    };
    let device = instance.device();
    let vkdevice = device.logical();
    let dst_image =
        unsafe { vkdevice.create_image(&img_ci, None) }.expect("Failed to create image");
    let requirements = unsafe { vkdevice.get_image_memory_requirements(dst_image) };
    let alloc_desc = AllocationCreateDesc {
        name: "Export Image GPU",
        requirements,
        location: MemoryLocation::GpuToCpu,
        linear: true,
    };
    let allocation = instance
        .allocator()
        .allocator
        .lock()
        .unwrap()
        .allocate(&alloc_desc)
        .expect("Allocation failed. OOM?");
    unsafe { vkdevice.bind_image_memory(dst_image, allocation.memory(), allocation.offset()) }
        .expect("Failed to bind image memory");
    // prepare the blit subregion
    let blit_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let blit_region = vk::ImageBlit {
        src_subresource: blit_subresource,
        src_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: width as i32,
                y: height as i32,
                z: 1,
            },
        ],
        dst_subresource: blit_subresource,
        dst_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: width as i32,
                y: height as i32,
                z: 1,
            },
        ],
    };
    // prepare the copy subregions, if the blit is not supported
    let image_copy_region = vk::ImageCopy {
        src_subresource: blit_subresource,
        src_offset: blit_region.src_offsets[0],
        dst_subresource: blit_subresource,
        dst_offset: blit_region.dst_offsets[0],
        extent: extent3d,
    };
    // prepare the barriers
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let barrier_src_sampled_to_transfer = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::MEMORY_READ,
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: src_image.image,
        subresource_range,
    };
    let barrier_src_transfer_to_sampled = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_READ,
        dst_access_mask: vk::AccessFlags::MEMORY_READ,
        old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: src_image.image,
        subresource_range,
    };
    let barrier_dst_undefined_to_transfer = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: dst_image,
        subresource_range,
    };
    let barrier_dst_transfer_to_general = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::MEMORY_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::GENERAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: dst_image,
        subresource_range,
    };
    // run the commands
    let cmd = tcmdm.get_cmd_buffer();
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier_src_sampled_to_transfer,
                    barrier_dst_undefined_to_transfer,
                ],
            );
            if supports_linear_blit {
                device.cmd_blit_image(
                    cmd,
                    src_image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit_region],
                    vk::Filter::LINEAR,
                );
            } else {
                device.cmd_copy_image(
                    cmd,
                    src_image.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[image_copy_region],
                )
            }
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier_src_transfer_to_sampled,
                    barrier_dst_transfer_to_general,
                ],
            );
        }
    };
    let fence = device.immediate_execute(cmd, device.transfer_queue(), command);
    device.wait_completion(&[fence]);
    // now extract the bytes from the figure
    let size = width as usize * height as usize * 4;
    let mut raw = vec![0_u8; size];
    let mapped = allocation
        .mapped_ptr()
        .expect("Failed to map image")
        .as_ptr() as *const u8;
    unsafe { std::ptr::copy_nonoverlapping(mapped, raw.as_mut_ptr(), size) };
    image::RgbaImage::from_raw(width as u32, height as u32, raw).expect("Image conversion failed")
}
