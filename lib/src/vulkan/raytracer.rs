use super::cmd::CommandManager;
use super::descriptor::DescriptorSetManager;
use super::instance::Instance;
use super::memory::AllocatedBuffer;
use super::scene::{padding, RayTraceScene};
use super::{AllocatedImage, Descriptor, UnfinishedExecutions};
use crate::vulkan::pipeline::build_raytracing_pipeline;
use crate::{
    ParsedScene, Pipeline, PresentInstance, RayTraceInstance, TextureFormat, TextureInfo,
    TextureLoaded, VulkanScene,
};
use ash::extensions::khr::{
    AccelerationStructure as AccelerationLoader, RayTracingPipeline as RTPipelineLoader,
};
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use gpu_allocator::MemoryLocation;
use std::ptr;
use std::sync::mpsc::Sender;
use std::sync::Arc;

#[repr(C)]
struct FrameDataRT {
    camera2world: Matrix4<f32>,
    screen2camera: Matrix4<f32>,
}

impl FrameDataRT {
    fn new(scene: &RayTraceScene, extent: vk::Extent2D) -> Self {
        let ar = extent.width as f32 / extent.height as f32;
        FrameDataRT {
            camera2world: scene.camera.look_at_rh().invert().unwrap(),
            screen2camera: scene.camera.projection(ar).invert().unwrap(),
        }
    }
}

struct ShaderBindingTable {
    rgen_addr: vk::StridedDeviceAddressRegionKHR,
    miss_addr: vk::StridedDeviceAddressRegionKHR,
    hit_addr: vk::StridedDeviceAddressRegionKHR,
    call_addr: vk::StridedDeviceAddressRegionKHR,
    buffer: AllocatedBuffer,
}

pub struct RayTraceRenderer<T: Instance + Send + Sync> {
    scene: RayTraceScene,
    extent: vk::Extent2D,
    out_img: AllocatedImage,
    descriptor: Descriptor,
    sbt: ShaderBindingTable,
    pipeline: Pipeline,
    _frame_data: AllocatedBuffer,
    dm: DescriptorSetManager,
    gcmdm: CommandManager,
    tcmdm: CommandManager,
    ccmdm: CommandManager,
    asloader: Arc<AccelerationLoader>,
    rploader: RTPipelineLoader,
    instance: Arc<T>,
}

impl<T: Instance + Send + Sync + 'static> RayTraceRenderer<T> {
    pub fn new(
        instance: Arc<RayTraceInstance>,
        scene: Box<dyn ParsedScene>,
        width: u32,
        height: u32,
    ) -> Result<RayTraceRenderer<RayTraceInstance>, std::io::Error> {
        let device = instance.device();
        let compute = device.compute_queue();
        let mut ccmdm = CommandManager::new(device.logical_clone(), compute.idx, 15);
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let scene = RayTraceScene::new(instance.clone(), loader.clone(), scene, &mut ccmdm)?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(instance, loader, ccmdm, scene, extent))
    }

    pub(crate) fn from_realtime(
        instance: Arc<PresentInstance>,
        scene: &mut VulkanScene,
        width: u32,
        height: u32,
    ) -> Result<RayTraceRenderer<PresentInstance>, std::io::Error> {
        let device = instance.device();
        let compute = device.compute_queue();
        let mut ccmdm = CommandManager::new(device.logical_clone(), compute.idx, 15);
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let scene = RayTraceScene::from(loader.clone(), scene, &mut ccmdm)?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(instance, loader, ccmdm, scene, extent))
    }

    pub fn draw(mut self, channel: Sender<String>) -> TextureLoaded {
        // if the other end disconnected, this thread can die anyway, so unwrap()
        channel.send("Tracing rays".to_string()).unwrap();
        let cmd = self.ccmdm.get_cmd_buffer();
        let device = self.instance.device();
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.pipeline.layout,
                    0,
                    &[self.descriptor.set],
                    &[],
                );
                self.rploader.cmd_trace_rays(
                    cmd,
                    &self.sbt.rgen_addr,
                    &self.sbt.miss_addr,
                    &self.sbt.hit_addr,
                    &self.sbt.call_addr,
                    self.extent.width,
                    self.extent.height,
                    1,
                );
            }
        };
        let fence = device.immediate_execute(cmd, device.compute_queue(), command);
        device.wait_completion(&[fence]);
        channel.send("Rendering finished".to_string()).unwrap();
        channel.send("Copying result".to_string()).unwrap();

        let out_image = copy_storage_to_output(
            self.instance.as_ref(),
            &mut self.gcmdm,
            &self.out_img,
            self.extent,
        );
        let out_info = TextureInfo {
            name: "RT out image".to_string(),
            format: TextureFormat::Rgba,
            width: self.extent.width as u16,
            height: self.extent.height as u16,
        };
        TextureLoaded {
            info: out_info,
            image: out_image,
            instance: self.instance,
        }
    }
}

fn copy_storage_to_output<T: Instance>(
    instance: &T,
    gcmdm: &mut CommandManager,
    img: &AllocatedImage,
    extent: vk::Extent2D,
) -> AllocatedImage {
    // blit the final image to one with a better layout to be displayed
    // probably later I want to just COPY the image, perform manual exposure compensation,
    // and then blit to sRGB or 8bit format.
    //TODO: probably I want this to be R32G32B32A32_SFLOAT
    let retval = instance.allocator().create_image_gpu(
        "RT output",
        vk::Format::R8G8B8A8_SRGB,
        extent,
        vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let blit_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let blit_regions = [vk::ImageBlit {
        src_subresource: blit_subresource,
        src_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: extent.width as i32,
                y: extent.height as i32,
                z: 1,
            },
        ],
        dst_subresource: blit_subresource,
        dst_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: extent.width as i32,
                y: extent.height as i32,
                z: 1,
            },
        ],
    }];
    let barrier_src_general_to_transfer = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::SHADER_WRITE,
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::GENERAL,
        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: img.image,
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
        image: retval.image,
        subresource_range,
    };
    let barrier_src_transfer_to_general = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_WRITE,
        old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        new_layout: vk::ImageLayout::GENERAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: img.image,
        subresource_range,
    };
    let barrier_dst_transfer_to_shader = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: retval.image,
        subresource_range,
    };
    let device = instance.device();
    let command = |vkdevice: &ash::Device, cmd: vk::CommandBuffer| unsafe {
        vkdevice.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                barrier_src_general_to_transfer,
                barrier_dst_undefined_to_transfer,
            ],
        );
        // blit works only on a graphic queue
        vkdevice.cmd_blit_image(
            cmd,
            img.image,
            vk::ImageLayout::GENERAL,
            retval.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &blit_regions,
            vk::Filter::LINEAR,
        );
        vkdevice.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                barrier_src_transfer_to_general,
                barrier_dst_transfer_to_shader,
            ],
        );
    };
    let cmd = gcmdm.get_cmd_buffer();
    let graphic_queue = device.graphic_queue();
    let fence = device.immediate_execute(cmd, graphic_queue, command);
    device.wait_completion(&[fence]);
    retval
}

fn setup_frame_data<T: Instance>(
    instance: &T,
    tcmdm: &mut CommandManager,
    extent: vk::Extent2D,
    scene: &RayTraceScene,
    unf: &mut UnfinishedExecutions,
) -> AllocatedBuffer {
    let size = std::mem::size_of::<FrameDataRT>() as u64;
    let framedata = FrameDataRT::new(scene, extent);
    let mm = instance.allocator();
    let cpu_buf = mm.create_buffer(
        "Raytrace frame data CPU staging",
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let mapped = cpu_buf
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    unsafe { std::ptr::copy_nonoverlapping(&framedata, mapped, 1) };
    let gpu_buf = mm.create_buffer(
        "Raytrace frame data GPU",
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
    );
    let buffer_copy = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: cpu_buf.size,
    };
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_copy_buffer(cmd, cpu_buf.buffer, gpu_buf.buffer, &[buffer_copy]);
        }
    };
    let cmd = tcmdm.get_cmd_buffer();
    let device = instance.device();
    let fence = device.immediate_execute(cmd, device.transfer_queue(), command);
    unf.add(fence, cpu_buf);
    gpu_buf
}

fn init_rt<T: Instance + Send + Sync>(
    instance: Arc<T>,
    loader: Arc<AccelerationLoader>,
    ccmdm: CommandManager,
    scene: RayTraceScene,
    extent: vk::Extent2D,
) -> RayTraceRenderer<T> {
    const AVG_DESC: [(vk::DescriptorType, f32); 3] = [
        (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
        (vk::DescriptorType::STORAGE_BUFFER, 1.0),
        (vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1.0),
    ];
    let device = instance.device();
    let mut unf = UnfinishedExecutions::new(device);
    let rploader = RTPipelineLoader::new(instance.instance(), device.logical());
    let graphic_queue = device.graphic_queue();
    let transfer_queue = device.transfer_queue();
    let gcmdm = CommandManager::new(device.logical_clone(), graphic_queue.idx, 1);
    let mut tcmdm = CommandManager::new(device.logical_clone(), transfer_queue.idx, 1);
    let mut dm = DescriptorSetManager::new(
        device.logical_clone(),
        &AVG_DESC,
        instance.desc_layout_cache(),
    );
    let vertex_buffer_info = vk::DescriptorBufferInfo {
        buffer: scene.vertex_buffer.buffer,
        offset: 0,
        range: scene.vertex_buffer.size,
    };
    let index_buffer_info = vk::DescriptorBufferInfo {
        buffer: scene.index_buffer.buffer,
        offset: 0,
        range: scene.index_buffer.size,
    };
    let instance_buffer_info = vk::DescriptorBufferInfo {
        buffer: scene.instance_buffer.buffer,
        offset: 0,
        range: scene.instance_buffer.size,
    };
    let out_img = create_storage_image(instance.as_ref(), &mut tcmdm, extent, &mut unf);
    let outimg_descinfo = vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: out_img.image_view,
        image_layout: vk::ImageLayout::GENERAL,
    };
    let frame_data = setup_frame_data(instance.as_ref(), &mut tcmdm, extent, &scene, &mut unf);
    let fdinfo = vk::DescriptorBufferInfo {
        buffer: frame_data.buffer,
        offset: 0,
        range: frame_data.size,
    };
    let descriptor = dm
        .new_set()
        .bind_acceleration_structure(&scene.acc.tlas.accel, vk::ShaderStageFlags::RAYGEN_KHR)
        .bind_buffer(
            fdinfo,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_image(
            outimg_descinfo,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_buffer(
            vertex_buffer_info,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            index_buffer_info,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            instance_buffer_info,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .build();
    let pipeline =
        build_raytracing_pipeline(&rploader, device.logical_clone(), &[descriptor.layout]);
    let sbt = build_sbt(
        instance.as_ref(),
        &rploader,
        &mut tcmdm,
        &pipeline,
        1,
        &mut unf,
    );
    unf.wait_completion();
    RayTraceRenderer {
        scene,
        extent,
        out_img,
        descriptor,
        sbt,
        pipeline,
        _frame_data: frame_data,
        dm,
        gcmdm,
        tcmdm,
        ccmdm,
        asloader: loader,
        rploader,
        instance,
    }
}

fn create_storage_image<T: Instance>(
    instance: &T,
    tcmdm: &mut CommandManager,
    extent: vk::Extent2D,
    unf: &mut UnfinishedExecutions,
) -> AllocatedImage {
    let mm = instance.allocator();
    let device = instance.device();
    let out_img = mm.create_image_gpu(
        "RT out image",
        vk::Format::R32G32B32A32_SFLOAT,
        extent,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let barrier_use = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::GENERAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: out_img.image,
        subresource_range,
    };
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_use],
            );
        }
    };
    let cmd = tcmdm.get_cmd_buffer();
    let fence = device.immediate_execute(cmd, device.transfer_queue(), command);
    unf.add_fence(fence);
    out_img
}

fn roundup_alignment(size: u64, align_to: u64) -> u64 {
    (size + (align_to - 1)) & !(align_to - 1)
}

fn build_sbt<T: Instance>(
    instance: &T,
    rploader: &RTPipelineLoader,
    tcmdm: &mut CommandManager,
    pipeline: &Pipeline,
    hit_groups: u32,
    unf: &mut UnfinishedExecutions,
) -> ShaderBindingTable {
    let device = instance.device();
    let properties =
        unsafe { RTPipelineLoader::get_properties(instance.instance(), device.physical().device) };
    let align_handle = properties.shader_group_handle_alignment as u64;
    let align_group = properties.shader_group_base_alignment as u64;
    let size_handle = properties.shader_group_handle_size as u64;
    let size_handle_aligned = roundup_alignment(size_handle, align_handle);
    let mut data = Vec::new();
    // load single raygen group
    let rgen_group = unsafe {
        rploader.get_ray_tracing_shader_group_handles(pipeline.pipeline, 0, 1, size_handle as usize)
    }
    .expect("Failed to retrieve shader handle");
    data.extend_from_slice(&rgen_group);
    let missing_bytes = padding(data.len() as u64, align_group) as usize;
    data.extend_from_slice(&vec![0; missing_bytes]);
    // in the NVIDIA example it's written that stride and size for rgen must be the same
    let mut rgen_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: 0,
        stride: roundup_alignment(align_handle, align_group),
        size: roundup_alignment(align_handle, align_group),
    };
    // load single miss group
    let miss_offset = data.len() as u64;
    let miss_group = unsafe {
        rploader.get_ray_tracing_shader_group_handles(pipeline.pipeline, 1, 1, size_handle as usize)
    }
    .expect("Failed to retrieve shader handle");
    data.extend_from_slice(&miss_group);
    let missing_bytes = padding(data.len() as u64, align_group) as usize;
    data.extend_from_slice(&vec![0; missing_bytes]);
    let mut miss_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: miss_offset,
        stride: size_handle,
        size: size_handle_aligned,
    };
    // load hit groups
    let hit_group_base_index = 2; // 0 is raygen and 1 is miss
    let hit_offset = data.len() as u64;
    for hit_group in 0..hit_groups {
        let shader_group = unsafe {
            rploader.get_ray_tracing_shader_group_handles(
                pipeline.pipeline,
                hit_group_base_index + hit_group,
                1,
                size_handle as usize,
            )
        }
        .expect("Failed to retrieve shader handle");
        data.extend_from_slice(&shader_group);
        if align_handle != size_handle as u64 {
            let missing_bytes = padding(data.len() as u64, align_handle) as usize;
            data.extend_from_slice(&vec![0; missing_bytes]);
        }
    }
    let mut hit_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: hit_offset,
        stride: size_handle,
        size: (hit_groups as u64 * size_handle_aligned),
    };
    let call_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: 0,
        stride: 0,
        size: 0,
    };
    // now upload everything to a buffer
    let mm = instance.allocator();
    let cpu_buf = mm.create_buffer(
        "SBT CPU",
        data.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let mapped = cpu_buf
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len()) };
    let gpu_buf = mm.create_buffer(
        "SBT GPU",
        data.len() as u64,
        vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
            | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        MemoryLocation::GpuOnly,
    );
    let buffer_copy = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: cpu_buf.size,
    };
    let transfer_queue = device.transfer_queue();
    let cmd = tcmdm.get_cmd_buffer();
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_copy_buffer(cmd, cpu_buf.buffer, gpu_buf.buffer, &[buffer_copy]);
        }
    };
    let fence = device.immediate_execute(cmd, transfer_queue, command);
    unf.add(fence, cpu_buf);
    // calculates the groups addresses
    let sbt_addr_info = vk::BufferDeviceAddressInfo {
        s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
        p_next: ptr::null(),
        buffer: gpu_buf.buffer,
    };
    let base_addr = unsafe { device.logical().get_buffer_device_address(&sbt_addr_info) };
    rgen_addr.device_address += base_addr;
    miss_addr.device_address += base_addr;
    hit_addr.device_address += base_addr;
    ShaderBindingTable {
        rgen_addr,
        miss_addr,
        hit_addr,
        call_addr,
        buffer: gpu_buf,
    }
}

#[cfg(test)]
mod tests {
    use super::RayTraceRenderer;
    use crate::{parse, RayTraceInstance};
    use std::path::PathBuf;
    use std::sync::{mpsc, Arc};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn load_raytrace() {
        init();
        if let Some(instance) = RayTraceInstance::new() {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("resources")
                .join("cube.glaze");
            let parsed = parse(path).unwrap();
            let _ = RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 2, 2)
                .unwrap();
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }

    #[test]
    fn draw_outlive() {
        init();
        if let Some(instance) = RayTraceInstance::new() {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("resources")
                .join("cube.glaze");
            let parsed = parse(path).unwrap();
            let renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 2, 2)
                    .unwrap();
            let (write, _read) = mpsc::channel();
            let _ = renderer.draw(write);
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }

    #[test]
    fn save_to_disk() {
        init();
        if let Some(instance) = RayTraceInstance::new() {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("resources")
                .join("cube.glaze");
            let parsed = parse(path).unwrap();
            let renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 800, 600)
                    .unwrap();
            let (write, _read) = mpsc::channel();
            let _image = renderer.draw(write).export();
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }
}
