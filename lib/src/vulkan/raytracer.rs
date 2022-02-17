use super::cmd::CommandManager;
use super::descriptor::DescriptorSetManager;
use super::instance::Instance;
use super::memory::AllocatedBuffer;
use super::scene::{padding, RayTraceScene};
use super::{AllocatedImage, Descriptor, UnfinishedExecutions};
use crate::geometry::{SBT_LIGHT_STRIDE, SBT_LIGHT_TYPES};
use crate::materials::{SBT_MATERIAL_STRIDE, SBT_MATERIAL_TYPES};
use crate::vulkan::pipeline::build_raytracing_pipeline;
#[cfg(feature = "vulkan-interactive")]
use crate::PresentInstance;
#[cfg(feature = "vulkan-interactive")]
use crate::VulkanScene;
use crate::{
    Camera, ParsedScene, Pipeline, RayTraceInstance, TextureFormat, TextureInfo, TextureLoaded,
};
use ash::extensions::khr::{
    AccelerationStructure as AccelerationLoader, RayTracingPipeline as RTPipelineLoader,
};
use ash::vk;
use cgmath::{SquareMatrix, Vector2 as Vec2};
use gpu_allocator::MemoryLocation;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128PlusPlus;
use std::iter::repeat;
use std::ptr;
use std::sync::mpsc::Sender;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RTFrameData {
    seed: u32,
    lights_no: u32,
    pixel_offset: Vec2<f32>,
    scene_radius: f32,
    exposure: f32,
}

impl Default for RTFrameData {
    fn default() -> Self {
        Self {
            seed: 0,
            lights_no: 0,
            scene_radius: 0.0,
            pixel_offset: Vec2::new(0.0, 0.0),
            exposure: 1.0,
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
    pub(super) scene: RayTraceScene<T>,
    camera: Camera,
    push_constants: [u8; 128],
    frame_data: Vec<AllocatedBuffer>,
    rng: Xoshiro128PlusPlus,
    cumulative_img: AllocatedImage,
    out_img: AllocatedImage,
    sample_scheduler: WorkScheduler,
    pub(crate) request_new_frame: bool,
    // contains also the out_img
    frame_desc: Vec<Descriptor>,
    extent: vk::Extent2D,
    sbt: ShaderBindingTable,
    pipeline: Pipeline,
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
        let scene = RayTraceScene::<RayTraceInstance>::new(
            instance.clone(),
            loader.clone(),
            scene,
            &mut ccmdm,
        )?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(instance, loader, ccmdm, scene, extent, 1))
    }

    pub fn set_exposure(&mut self, exposure: f32) {
        if exposure >= 0.0 {
            // the higher the value, the brighter the image.
            self.scene.meta.exposure = exposure;
        }
        // no need to restart the frame, as only the weight for each sample is affected.
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn from_realtime(
        instance: Arc<PresentInstance>,
        scene: &mut VulkanScene,
        width: u32,
        height: u32,
        frames_in_flight: usize,
    ) -> Result<RayTraceRenderer<PresentInstance>, std::io::Error> {
        let device = instance.device();
        let compute = device.compute_queue();
        let mut ccmdm = CommandManager::new(device.logical_clone(), compute.idx, 15);
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let scene = RayTraceScene::<PresentInstance>::from(loader.clone(), scene, &mut ccmdm)?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(
            instance,
            loader,
            ccmdm,
            scene,
            extent,
            frames_in_flight as u8,
        ))
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn change_resolution(&mut self, width: u32, height: u32) {
        self.extent = vk::Extent2D { width, height };
        let mut unf = UnfinishedExecutions::new(self.instance.device());
        self.out_img = create_storage_image(
            self.instance.as_ref(),
            &mut self.tcmdm,
            self.extent,
            &mut unf,
            false,
        );
        self.cumulative_img = create_storage_image(
            self.instance.as_ref(),
            &mut self.tcmdm,
            self.extent,
            &mut unf,
            true,
        );
        self.frame_desc = build_descriptor(
            &mut self.dm,
            &self.frame_data,
            &self.cumulative_img,
            &self.out_img,
        );
        unf.wait_completion();
        self.update_camera(&self.camera.clone());
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn update_camera(&mut self, camera: &Camera) {
        self.push_constants = build_push_constants(camera, self.extent);
        self.request_new_frame = true;
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn draw_frame(
        &mut self,
        wait: vk::Semaphore,
        signal: vk::Semaphore,
        out_img: &AllocatedImage,
        frame_no: usize,
    ) {
        let device = self.instance.device();
        let vkdevice = device.logical();
        let cmd_begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),
        };
        let cmds = [self.ccmdm.get_cmd_buffer()];
        let cmd = cmds[0];
        let wait = [wait];
        let signal = [signal];
        let submit_ci = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait.len() as u32,
            p_wait_semaphores: wait.as_ptr(),
            p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_buffer_count: cmds.len() as u32,
            p_command_buffers: cmds.as_ptr(),
            signal_semaphore_count: signal.len() as u32,
            p_signal_semaphores: signal.as_ptr(),
        };
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
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: blit_subresource,
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ],
        }];
        let barrier_src_general_to_transfer = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::MEMORY_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.out_img.image,
            subresource_range,
        };
        let barrier_dst_undefined_to_transfer = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::MEMORY_READ,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: out_img.image,
            subresource_range,
        };
        let barrier_src_transfer_to_general = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.out_img.image,
            subresource_range,
        };
        let barrier_dst_transfer_to_shader = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::MEMORY_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: out_img.image,
            subresource_range,
        };
        let frame_index = frame_no % self.frame_desc.len();
        if self.request_new_frame {
            self.sample_scheduler.rewind();
        }
        let fd = RTFrameData {
            seed: self.rng.gen(),
            lights_no: self.scene.lights_no,
            pixel_offset: self.sample_scheduler.next().unwrap(),
            scene_radius: self.scene.meta.scene_radius,
            exposure: self.scene.meta.exposure,
        };
        update_frame_data(fd, &mut self.frame_data[frame_index]);
        let queue = device.compute_queue();
        unsafe {
            vkdevice
                .begin_command_buffer(cmd, &cmd_begin)
                .expect("Failed to begin command");
            if self.request_new_frame {
                self.request_new_frame = false;
                let color = vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                };
                vkdevice.cmd_clear_color_image(
                    cmd,
                    self.cumulative_img.image,
                    vk::ImageLayout::GENERAL,
                    &color,
                    &[subresource_range],
                );
            }
            vkdevice.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                &self.push_constants,
            );
            vkdevice.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline.pipeline,
            );
            vkdevice.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline.layout,
                0,
                &[self.frame_desc[frame_index].set, self.scene.descriptor.set],
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
                self.out_img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                out_img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blit_regions,
                vk::Filter::NEAREST,
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

            vkdevice
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");
            vkdevice
                .queue_submit(queue.queue, &[submit_ci], vk::Fence::null())
                .expect("Failed to submit to queue");
        }
    }

    pub fn draw(mut self, channel: Sender<String>) -> TextureLoaded {
        // if the other end disconnected, this thread can die anyway, so unwrap()
        channel.send("Tracing rays".to_string()).unwrap();
        if self.request_new_frame {
            self.sample_scheduler.rewind();
        }
        let fd = RTFrameData {
            seed: self.rng.gen(),
            lights_no: self.scene.lights_no,
            pixel_offset: self.sample_scheduler.next().unwrap(),
            scene_radius: self.scene.meta.scene_radius,
            exposure: self.scene.meta.exposure,
        };
        self.request_new_frame = false;
        update_frame_data(fd, &mut self.frame_data[0]);
        let cmd = self.ccmdm.get_cmd_buffer();
        let device = self.instance.device();
        let new_frame = self.request_new_frame;
        self.request_new_frame = false;
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                if new_frame {
                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let color = vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    };
                    device.cmd_clear_color_image(
                        cmd,
                        self.cumulative_img.image,
                        vk::ImageLayout::GENERAL,
                        &color,
                        &[subresource_range],
                    );
                }
                device.cmd_push_constants(
                    cmd,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                    0,
                    &self.push_constants,
                );
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
                    &[self.frame_desc[0].set, self.scene.descriptor.set],
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
            instance: self.instance.clone(),
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
        src_access_mask: vk::AccessFlags::MEMORY_WRITE,
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
        src_access_mask: vk::AccessFlags::MEMORY_READ,
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
        dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
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
        dst_access_mask: vk::AccessFlags::MEMORY_READ,
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
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            retval.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &blit_regions,
            vk::Filter::NEAREST,
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

fn init_rt<T: Instance + Send + Sync>(
    instance: Arc<T>,
    loader: Arc<AccelerationLoader>,
    ccmdm: CommandManager,
    scene: RayTraceScene<T>,
    extent: vk::Extent2D,
    frames_in_flight: u8,
) -> RayTraceRenderer<T> {
    const AVG_DESC: [(vk::DescriptorType, f32); 2] = [
        (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
        (vk::DescriptorType::STORAGE_IMAGE, 1.0),
    ];
    let device = instance.device();
    let mut unf = UnfinishedExecutions::new(device);
    let rploader = RTPipelineLoader::new(instance.instance(), device.logical());
    let graphic_queue = device.graphic_queue();
    let transfer_queue = device.transfer_queue();
    let gcmdm = CommandManager::new(device.logical_clone(), graphic_queue.idx, 1);
    let sample_scheduler = WorkScheduler::new();
    let mut tcmdm = CommandManager::new(device.logical_clone(), transfer_queue.idx, 1);
    let mut dm = DescriptorSetManager::new(
        device.logical_clone(),
        &AVG_DESC,
        instance.desc_layout_cache(),
    );
    let out_img = create_storage_image(instance.as_ref(), &mut tcmdm, extent, &mut unf, false);
    let cumulative_img =
        create_storage_image(instance.as_ref(), &mut tcmdm, extent, &mut unf, true);
    let camera = scene.camera.clone();
    let push_constants = build_push_constants(&camera, extent);
    let frame_data = (0..frames_in_flight)
        .into_iter()
        .map(|_| {
            instance.allocator().create_buffer(
                "raytrace framedata",
                std::mem::size_of::<RTFrameData>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            )
        })
        .collect::<Vec<_>>();
    let frame_desc = build_descriptor(&mut dm, &frame_data, &cumulative_img, &out_img);
    let pipeline = build_raytracing_pipeline(
        &rploader,
        device.logical_clone(),
        &[frame_desc[0].layout, scene.descriptor.layout],
    );
    let sbt = build_sbt(
        instance.as_ref(),
        &rploader,
        &mut tcmdm,
        &pipeline,
        &mut unf,
    );
    let rng = Xoshiro128PlusPlus::from_entropy();
    unf.wait_completion();
    RayTraceRenderer {
        scene,
        camera,
        push_constants,
        frame_data,
        rng,
        cumulative_img,
        out_img,
        sample_scheduler,
        request_new_frame: true,
        frame_desc,
        extent,
        sbt,
        pipeline,
        dm,
        gcmdm,
        tcmdm,
        ccmdm,
        asloader: loader,
        rploader,
        instance,
    }
}

fn build_descriptor(
    dm: &mut DescriptorSetManager,
    frame_data: &[AllocatedBuffer],
    cumulative_img: &AllocatedImage,
    out_img: &AllocatedImage,
) -> Vec<Descriptor> {
    frame_data
        .iter()
        .map(|fd| {
            dm.new_set()
                .bind_buffer(
                    fd,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CALLABLE_KHR,
                )
                .bind_image(
                    cumulative_img,
                    vk::ImageLayout::GENERAL,
                    vk::Sampler::null(),
                    vk::DescriptorType::STORAGE_IMAGE,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                )
                .bind_image(
                    out_img,
                    vk::ImageLayout::GENERAL,
                    vk::Sampler::null(),
                    vk::DescriptorType::STORAGE_IMAGE,
                    vk::ShaderStageFlags::RAYGEN_KHR,
                )
                .build()
        })
        .collect()
}

fn create_storage_image<T: Instance>(
    instance: &T,
    tcmdm: &mut CommandManager,
    extent: vk::Extent2D,
    unf: &mut UnfinishedExecutions,
    clearable: bool,
) -> AllocatedImage {
    let mm = instance.allocator();
    let device = instance.device();
    let clearable_flags = if clearable {
        // one of the two image (self.cumulative) requires clearing, which in turn requires
        // transfer_dst
        vk::ImageUsageFlags::TRANSFER_DST
    } else {
        vk::ImageUsageFlags::empty()
    };
    let out_img = mm.create_image_gpu(
        "RT out image",
        vk::Format::R32G32B32A32_SFLOAT,
        extent,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | clearable_flags,
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
    data.extend(repeat(0).take(missing_bytes));
    // in the NVIDIA example it's written that stride and size for rgen must be the same
    let mut rgen_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: 0,
        stride: roundup_alignment(align_handle, align_group),
        size: roundup_alignment(align_handle, align_group),
    };

    // load two miss groups: normal ray and shadow ray miss
    let miss_offset = data.len() as u64;
    let miss_groups = unsafe {
        rploader.get_ray_tracing_shader_group_handles(
            pipeline.pipeline,
            1,
            2,
            2 * size_handle as usize,
        )
    }
    .expect("Failed to retrieve shader handle");
    for miss_group in miss_groups.chunks_exact(size_handle as usize) {
        data.extend_from_slice(miss_group);
        // ensures every member is aligned properly
        if align_handle != size_handle as u64 {
            let missing_bytes = padding(data.len() as u64, align_handle) as usize;
            data.extend(repeat(0).take(missing_bytes));
        }
    }
    let missing_bytes = padding(data.len() as u64, align_group) as usize;
    data.extend(repeat(0).take(missing_bytes));
    let mut miss_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: miss_offset,
        stride: size_handle,
        size: 2 * size_handle_aligned,
    };

    // load single hit group
    let hit_offset = data.len() as u64;
    let shader_group = unsafe {
        rploader.get_ray_tracing_shader_group_handles(pipeline.pipeline, 3, 1, size_handle as usize)
    }
    .expect("Failed to retrieve shader handle");
    data.extend_from_slice(&shader_group);
    let missing_bytes = padding(data.len() as u64, align_group) as usize;
    data.extend(repeat(0).take(missing_bytes));
    let mut hit_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: hit_offset,
        stride: size_handle,
        size: size_handle_aligned,
    };

    // load multiple callables. Retrieve them all at once
    let call_offset = data.len() as u64;
    let callables_base_index = 4;
    let total_call = SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE + SBT_MATERIAL_TYPES * SBT_MATERIAL_STRIDE;
    let callables_group = unsafe {
        rploader.get_ray_tracing_shader_group_handles(
            pipeline.pipeline,
            callables_base_index,
            total_call as u32,
            total_call * size_handle as usize,
        )
    }
    .expect("Failed to retrieve shader handle");
    for callable in callables_group.chunks_exact(size_handle as usize) {
        data.extend_from_slice(callable);
        if align_handle != size_handle as u64 {
            let missing_bytes = padding(data.len() as u64, align_handle) as usize;
            data.extend(repeat(0).take(missing_bytes));
        }
    }
    let missing_bytes = padding(data.len() as u64, align_group) as usize;
    data.extend(repeat(0).take(missing_bytes));
    let mut call_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: call_offset,
        stride: size_handle,
        size: total_call as u64 * size_handle_aligned,
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
    call_addr.device_address += base_addr;
    ShaderBindingTable {
        rgen_addr,
        miss_addr,
        hit_addr,
        call_addr,
        buffer: gpu_buf,
    }
}

fn build_push_constants(camera: &Camera, extent: vk::Extent2D) -> [u8; 128] {
    // build matrices
    let ar = extent.width as f32 / extent.height as f32;
    let view_inv = camera.look_at_rh().invert().unwrap();
    let mut proj = camera.projection(ar);
    proj[1][1] *= -1.0; // cgmath is made for openGL and vulkan projection is slightly different
    let proj_inv = proj.invert().unwrap();
    // save matrices to byte array
    let mut retval = [0; 128];
    let mut index = 0;
    let matrices = [view_inv, proj_inv];
    for matrix in matrices {
        let vals: &[f32; 16] = matrix.as_ref();
        for val in vals {
            let bytes = f32::to_le_bytes(*val);
            retval[index] = bytes[0];
            retval[index + 1] = bytes[1];
            retval[index + 2] = bytes[2];
            retval[index + 3] = bytes[3];
            index += 4;
        }
    }
    retval
}

fn update_frame_data(fd: RTFrameData, buf: &mut AllocatedBuffer) {
    let mapped = buf
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    unsafe { std::ptr::copy_nonoverlapping(&fd, mapped, 1) };
}

/// Iterator used to schedule the samples in a pixel.
/// Starting from the entire pixel area ((0,0),(1,1)) this iterator subdivides the pixel
/// and choses which area to sample.
/// In the realtime raytracing the number of samples per pixel is not fixed so this process has to
/// be incremental.
///
/// This iterator is unlimited and will never produce None.
struct WorkScheduler {
    current: Vec<(Vec2<f32>, Vec2<f32>)>,
    next: Vec<(Vec2<f32>, Vec2<f32>)>,
}

impl WorkScheduler {
    pub fn new() -> Self {
        WorkScheduler {
            current: vec![(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0))],
            next: Vec::new(),
        }
    }

    /// Resets this iterator
    pub fn rewind(&mut self) {
        *self = WorkScheduler::new()
    }
}

impl Iterator for WorkScheduler {
    type Item = Vec2<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(area) = self.current.pop() {
            let middle = Vec2::new((area.0.x + area.1.x) / 2.0, (area.0.y + area.1.y) / 2.0);
            self.next.push((area.0, middle));
            self.next.push((middle, area.1));
            self.next
                .push((Vec2::new(middle.x, area.0.y), Vec2::new(area.1.x, middle.y)));
            self.next
                .push((Vec2::new(area.0.x, middle.y), Vec2::new(middle.x, area.1.y)));
            Some(middle)
        } else {
            self.current.append(&mut self.next);
            self.next()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RayTraceRenderer;
    use crate::{parse, ParsedScene, RayTraceInstance};
    use std::path::PathBuf;
    use std::sync::{mpsc, Arc};
    use tempfile::tempdir;

    fn init() -> Box<dyn ParsedScene + Send> {
        let _ = env_logger::builder().is_test(true).try_init();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("cube.glaze");
        parse(path).unwrap()
    }

    #[test]
    fn load_raytrace() {
        if let Some(instance) = RayTraceInstance::new() {
            let parsed = init();
            let _ = RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 2, 2)
                .unwrap();
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }

    #[test]
    fn draw_outlive() {
        if let Some(instance) = RayTraceInstance::new() {
            let parsed = init();
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
    fn save_to_disk() -> Result<(), std::io::Error> {
        if let Some(instance) = RayTraceInstance::new() {
            let parsed = init();
            let dir = tempdir()?;
            let file = dir.path().join("save.png");
            let renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 2, 2)
                    .unwrap();
            let (write, _read) = mpsc::channel();
            let image = renderer.draw(write).export();
            image.save(file.clone()).unwrap();
            assert!(file.exists());
        } else {
            // SKIPPED does not exists in cargo test...
        }
        Ok(())
    }

    #[test]
    fn change_resolution() -> Result<(), std::io::Error> {
        if let Some(instance) = RayTraceInstance::new() {
            let parsed = init();
            let mut renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 2, 2)
                    .unwrap();
            renderer.change_resolution(4, 4);
            let (write, _read) = mpsc::channel();
            let image = renderer.draw(write).export();
            assert_eq!(image.width(), 4);
            assert_eq!(image.height(), 4);
        } else {
            // SKIPPED does not exists in cargo test...
        }
        Ok(())
    }
}
