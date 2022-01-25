use super::cmd::CommandManager;
use super::descriptor::DescriptorSetManager;
use super::instance::Instance;
use super::memory::AllocatedBuffer;
use super::scene::RayTraceScene;
use super::{AllocatedImage, Descriptor};
use crate::vulkan::pipeline::build_raytracing_pipeline;
use crate::{
    ParsedScene, Pipeline, PresentInstance, RayTraceInstance, TextureFormat, TextureInfo,
    TextureLoaded, VulkanScene,
};
use ash::extensions::khr::{
    AccelerationStructure as AccelerationLoader, RayTracingPipeline as RTPipelineLoader,
};
use ash::vk;
use cgmath::{Point3, Vector3 as Vec3};
use gpu_allocator::MemoryLocation;
use std::ptr;
use std::sync::mpsc::Sender;
use std::sync::Arc;

pub const RAYTRACER_MAX_RECURSION: u32 = 6;

#[repr(C)]
struct FrameDataRT {
    cam_pos: Point3<f32>,
    cam_tgt: Point3<f32>,
    cam_upp: Vec3<f32>,
    fovx: f32,
    ar: f32,
}

impl FrameDataRT {
    fn new(scene: &RayTraceScene, extent: vk::Extent2D) -> Self {
        let fovx = match &scene.camera {
            crate::Camera::Perspective(persp) => persp.fovx,
            crate::Camera::Orthographic(_) => 0.0,
        };
        FrameDataRT {
            cam_pos: scene.camera.position(),
            cam_tgt: scene.camera.target(),
            cam_upp: scene.camera.up(),
            fovx,
            ar: extent.width as f32 / extent.height as f32,
        }
    }
}

pub const RAYTRACE_SPLIT_SIZE: u32 = 64;

pub struct RayTraceRenderer<T: Instance + Send + Sync> {
    scene: RayTraceScene,
    extent: vk::Extent2D,
    out_img: AllocatedImage,
    descriptor: Descriptor,
    dm: DescriptorSetManager,
    _frame_data: AllocatedBuffer,
    tcmdm: CommandManager,
    ccmdm: CommandManager,
    pipeline: Pipeline,
    asloader: Arc<AccelerationLoader>,
    rploader: RTPipelineLoader,
    instance: Arc<T>,
}

impl<T: Instance + Send + Sync> RayTraceRenderer<T> {
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
        let mm = instance.allocator();
        let scene = RayTraceScene::from(loader.clone(), scene, mm, &mut ccmdm)?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(instance, loader, ccmdm, scene, extent))
    }

    pub fn draw(mut self, channel: Sender<String>) -> TextureLoaded {
        // if the other end disconnected, this thread can die anyway, so unwrap()
        channel.send("Starting rendering".to_string()).unwrap();
        channel.send("Rendering finished".to_string()).unwrap();
        channel.send("Copying result".to_string()).unwrap();
        let out_image = copy_storage_to_output(
            self.instance.as_ref(),
            self.tcmdm.get_cmd_buffer(),
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
        }
    }
}

fn copy_storage_to_output<T: Instance>(
    instance: &T,
    tcmd: vk::CommandBuffer,
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
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
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
    let barrier_transfer = vk::ImageMemoryBarrier {
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
    let barrier_use = vk::ImageMemoryBarrier {
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
    let transfer_queue = device.transfer_queue();
    let command = |vkdevice: &ash::Device, cmd: vk::CommandBuffer| unsafe {
        vkdevice.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_transfer],
        );
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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_use],
        );
    };
    let fence = device.immediate_execute(tcmd, transfer_queue, command);
    device.wait_completion(&[fence]);
    retval
}

fn setup_frame_data<T: Instance>(
    instance: &T,
    tcmd: vk::CommandBuffer,
    extent: vk::Extent2D,
    scene: &RayTraceScene,
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
    //TODO: for when the raytracer will be finished:
    //at the time of writing, this is the only deferred upload I do at setup time.
    //After the renderer will be finished, if there are more deferred upload (like SSBOs or such)
    //consider returning the UnfinishedExecutions here instead of waiting on the Fence.
    //(I can't just return the fence otherwise the CPU buf will be deallocated before the GPU
    //finishes the actual copy)
    let device = instance.device();
    let tqueue = device.transfer_queue();
    let fence = device.immediate_execute(tcmd, tqueue, command);
    device.wait_completion(&[fence]);
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
    let rploader = RTPipelineLoader::new(instance.instance(), device.logical());
    let transfer_queue = device.transfer_queue();
    let mut tcmdm = CommandManager::new(device.logical_clone(), transfer_queue.idx, 1);
    let mut dm = DescriptorSetManager::new(
        device.logical_clone(),
        &AVG_DESC,
        instance.desc_layout_cache(),
    );
    let vbinfo = vk::DescriptorBufferInfo {
        buffer: scene.vertex_buffer.buffer,
        offset: 0,
        range: scene.vertex_buffer.size,
    };
    let ibinfo = vk::DescriptorBufferInfo {
        buffer: scene.index_buffer.buffer,
        offset: 0,
        range: scene.index_buffer.size,
    };
    let out_img = instance.allocator().create_image_gpu(
        "RT out image",
        vk::Format::R32G32B32A32_SFLOAT,
        extent,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let outimg_descinfo = vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: out_img.image_view,
        image_layout: vk::ImageLayout::GENERAL,
    };
    let tcmd = tcmdm.get_cmd_buffer();
    let frame_data = setup_frame_data(instance.as_ref(), tcmd, extent, &scene);
    let fdinfo = vk::DescriptorBufferInfo {
        buffer: frame_data.buffer,
        offset: 0,
        range: frame_data.size,
    };
    let descriptor = dm
        .new_set()
        .bind_buffer(
            fdinfo,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_buffer(
            vbinfo,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_buffer(
            ibinfo,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_acceleration_structure(&scene.acc.tlas.accel, vk::ShaderStageFlags::RAYGEN_KHR)
        .bind_image(
            outimg_descinfo,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .build();
    let pipeline =
        build_raytracing_pipeline(&rploader, device.logical_clone(), &[descriptor.layout]);
    RayTraceRenderer {
        scene,
        extent,
        out_img,
        descriptor,
        dm,
        _frame_data: frame_data,
        tcmdm,
        ccmdm,
        pipeline,
        asloader: loader,
        rploader,
        instance,
    }
}

#[cfg(test)]
mod tests {
    use super::RayTraceRenderer;
    use crate::{parse, RayTraceInstance};
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn load_raytrace() {
        env_logger::init();
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
}
