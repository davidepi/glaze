use super::cmd::CommandManager;
use super::descriptor::DescriptorSetManager;
use super::instance::Instance;
use super::memory::AllocatedBuffer;
use super::raytrace_structures::{PTLastVertex, RTFrameData, PT_STEPS};
use super::scene::{padding, RayTraceScene};
use super::sync::{create_fence, create_semaphore};
use super::{export, AllocatedImage, Descriptor, UnfinishedExecutions};
use crate::geometry::{SBT_LIGHT_STRIDE, SBT_LIGHT_TYPES};
use crate::materials::{SBT_MATERIAL_STRIDE, SBT_MATERIAL_TYPES};
use crate::parser::NoScene;
use crate::vulkan::pipeline::build_raytracing_pipeline;
use crate::vulkan::FRAMES_IN_FLIGHT;
#[cfg(feature = "vulkan-interactive")]
use crate::PresentInstance;
#[cfg(feature = "vulkan-interactive")]
use crate::RealtimeRenderer;
use crate::{Camera, Light, Material, Pipeline, RayTraceInstance, Texture};
use ash::extensions::khr::RayTracingPipeline as RTPipelineLoader;
use ash::vk;
use cgmath::SquareMatrix;
use gpu_allocator::MemoryLocation;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128PlusPlus;
use std::collections::VecDeque;
use std::iter::repeat;
use std::ptr;
use std::sync::Arc;

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Integrators available to solve the Rendering Equation.
///
/// The integrators contained in this enum have an implementation in the [RayTraceRenderer] and can
/// be used to solve the rendering equation.
pub enum Integrator {
    /// An integrator using only direct lighting.
    ///
    /// For each light path, a single ray is traced from the camera. After finding the intersection
    /// point, a single shadow ray is traced for occlusion testing and the path is considered
    /// terminated.
    DIRECT,
    /// An integrator implementing forward path tracing.
    ///
    /// For each light path, a ray is traced multiple times in the scene. At each intersection
    /// point, occlusion testing is done with a shadow ray. Every draw call calculates a single ray
    /// intersection + occlusion testing, so a single light path is composed of multiple draw
    /// calls. The final step of the light path does not trace any light (like in bidirectional
    /// methods, but simply drops the ray). Light paths with low contribution may be dropped
    /// randomly (russian roulette).
    #[default]
    PATH_TRACE,
}

impl Integrator {
    /// Returns all the values of this integrator.
    pub fn values() -> [Integrator; 2] {
        [Integrator::DIRECT, Integrator::PATH_TRACE]
    }

    /// Returns the name of the integrator as string.
    pub fn name(&self) -> &'static str {
        match self {
            Integrator::DIRECT => "Direct light only",
            Integrator::PATH_TRACE => "Path tracing",
        }
    }

    /// How many raygen shaders are necessary for each integrator
    fn raygen_count(&self) -> u32 {
        match self {
            Integrator::DIRECT | Integrator::PATH_TRACE => 1,
        }
    }

    /// Returns how many draw calls are required for each light path.
    ///
    /// In probabilistic integrators (e.g. [Integrator::PATH_TRACE]), returns the worst case
    /// amount.
    pub fn steps_per_sample(&self) -> usize {
        match self {
            Integrator::DIRECT => 1,
            Integrator::PATH_TRACE => PT_STEPS, // worst case
        }
    }
}

/// Contains addresses of the various raytrace shaders.
struct ShaderBindingTable {
    /// Starting address of the raygen shaders.
    ///
    /// Only a single raygen shader can be used. In order to use multiple of them we need to store
    /// the address of each one instead of only the first one (like for the other shaders).
    rgen_addrs: Vec<vk::StridedDeviceAddressRegionKHR>,
    /// Starting address of the miss shaders.
    miss_addr: vk::StridedDeviceAddressRegionKHR,
    /// Starting address of the hit/anyhit shaders.
    hit_addr: vk::StridedDeviceAddressRegionKHR,
    /// Starting address of the callable shaders.
    call_addr: vk::StridedDeviceAddressRegionKHR,
    /// Buffer containing all the shader handles (the actual SBT).
    _buffer: AllocatedBuffer,
}

/// Renderer capable of producing a fully raytraced image using different integrators.
///
/// This renderer does not support presenting to surface, so the resulting image MUST be passed to
/// the [RealtimeRenderer] for presentation.
pub struct RayTraceRenderer<T: Instance + Send + Sync> {
    /// Raytraced scene.
    scene: RayTraceScene<T>,
    /// Push constants data, updated before starting each draw.
    push_constants: [u8; 128],
    /// Additional per-frame data.
    frame_data: Vec<AllocatedBuffer>,
    /// Descriptor for the current frame. Contains also out8_img.
    frame_desc: Vec<Descriptor>,
    /// Random number generator, used to generate the random numbers required by the integrators.
    rng: Xoshiro128PlusPlus,
    /// Image used to accumulate pixel values. Frames are not independent in this renderer and the
    /// same one is reused as much as possible.
    cumulative_img: AllocatedImage,
    /// The output image, after dividing the cumulative image by the number of samples.
    out32_img: AllocatedImage,
    /// Same as out32 but uses R8G8B8A8 instead of R32G32B32A32.
    out8_img: AllocatedImage,
    /// The integrator used for rendering.
    integrator: Integrator,
    /// Additional buffer required by the integrator to store data between frames.
    integrator_data: AllocatedBuffer,
    /// Sampler, transform a discrete pixel into a continuous area to sample.
    sample_scheduler: WorkScheduler,
    /// True if a new frame should be requested (e.g. camera moved) and cumulative_img must reset.
    request_new_frame: bool,
    /// Extent of the rendered image.
    extent: vk::Extent2D,
    /// Raytracer SBT.
    sbt: ShaderBindingTable,
    /// Raytracer pipeline.
    pipeline: Pipeline,
    /// Descriptor manager for this renderer.
    dm: DescriptorSetManager,
    /// Transfer Queue command manager for this renderer.
    tcmdm: CommandManager,
    /// Compute Queue command manager for this renderer.
    ccmdm: CommandManager,
    /// Loader for RayTracePipeline.
    rploader: RTPipelineLoader,
    /// Vulkan instance.
    instance: Arc<T>,
}

impl<T: Instance + Send + Sync + 'static> RayTraceRenderer<T> {
    /// Creates a new raytrace renderer.
    ///
    /// Takes the instance, an optional scene, width and height of the expected output.
    /// # Examples
    /// Basic usage:
    /// ```no_run
    /// let instance = glaze::RayTraceInstance::new().expect("No GPU found");
    /// let arc_instance = std::sync::Arc::new(instance);
    /// let renderer = glaze::RaytraceRenderer::create(arc_instance, None, 1920, 1080);
    /// ```
    pub fn new(
        instance: Arc<RayTraceInstance>,
        scene: Option<RayTraceScene<RayTraceInstance>>,
        width: u32,
        height: u32,
    ) -> RayTraceRenderer<RayTraceInstance> {
        let scene = if let Some(scene) = scene {
            scene
        } else {
            RayTraceScene::<RayTraceInstance>::new(Arc::clone(&instance), Box::new(NoScene))
        };
        let extent = vk::Extent2D { width, height };
        init_rt(instance, scene, extent)
    }

    /// Waits until all the queued frames have been rendered.
    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    /// Sets the exposure for the rendered image.
    pub fn set_exposure(&mut self, exposure: f32) {
        if exposure >= 0.0 {
            // the higher the value, the brighter the image.
            self.scene.meta.exposure = exposure;
        }
        // no need to restart the frame, as only the weight for each sample is affected.
    }

    /// Sets the integrator used by the current renderer.
    ///
    /// A description of the possible values can be found in the [Integrator] enum.
    pub fn set_integrator(&mut self, integrator: Integrator) {
        if self.integrator != integrator {
            self.wait_idle();
            let instance = &self.instance;
            let device = instance.device();
            let mut unf = UnfinishedExecutions::new(device);
            let pipeline = build_raytracing_pipeline(
                &self.rploader,
                device.logical_clone(),
                &[self.frame_desc[0].layout, self.scene.descriptor.layout],
                integrator,
            );
            let sbt = build_sbt(
                instance.as_ref(),
                &self.rploader,
                &mut self.tcmdm,
                &pipeline,
                integrator,
                &mut unf,
            );
            self.integrator_data =
                create_integrator_data(self.instance.as_ref(), integrator, self.extent);
            self.frame_desc = build_descriptor(
                &mut self.dm,
                &self.frame_data,
                &self.integrator_data,
                &self.cumulative_img,
                &self.out32_img,
            );
            unf.wait_completion();
            self.pipeline = pipeline;
            self.sbt = sbt;
            self.integrator = integrator;
            self.request_new_frame = true;
        }
    }

    /// Changes the rendered scene.
    pub fn change_scene(&mut self, scene: RayTraceScene<T>) {
        let instance = &self.instance;
        let device = instance.device();
        let mut unf = UnfinishedExecutions::new(device);
        let pipeline = build_raytracing_pipeline(
            &self.rploader,
            device.logical_clone(),
            &[self.frame_desc[0].layout, scene.descriptor.layout],
            self.integrator,
        );
        let sbt = build_sbt(
            instance.as_ref(),
            &self.rploader,
            &mut self.tcmdm,
            &pipeline,
            self.integrator,
            &mut unf,
        );
        self.scene = scene;
        self.pipeline = pipeline;
        self.sbt = sbt;
        unf.wait_completion();
        self.request_new_frame = true;
    }

    /// Changes the rendered image resolution.
    pub fn change_resolution(&mut self, width: u32, height: u32) {
        self.extent = vk::Extent2D { width, height };
        let mut unf = UnfinishedExecutions::new(self.instance.device());
        self.out32_img = create_storage_image(
            self.instance.as_ref(),
            &mut self.tcmdm,
            self.extent,
            &mut unf,
            true,
        );
        self.out8_img = self.instance.allocator().create_image_gpu(
            "32bit image raytracer",
            vk::Format::R8G8B8A8_SRGB,
            self.extent,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        self.cumulative_img = create_storage_image(
            self.instance.as_ref(),
            &mut self.tcmdm,
            self.extent,
            &mut unf,
            true,
        );
        self.integrator_data =
            create_integrator_data(self.instance.as_ref(), self.integrator, self.extent);
        self.frame_desc = build_descriptor(
            &mut self.dm,
            &self.frame_data,
            &self.integrator_data,
            &self.cumulative_img,
            &self.out32_img,
        );
        unf.wait_completion();
        self.update_camera(self.scene.camera);
    }

    /// Changes the camera values.
    pub fn update_camera(&mut self, camera: Camera) {
        self.push_constants = build_push_constants(&camera, self.extent);
        self.request_new_frame = true;
    }

    /// Updates materials, cameras or textures in the renderer scene.
    ///
    /// The scene is owned by the renderer (as it requires tight synchronization with the GPU) and
    /// cannot be updated otherwise.
    // materials and lights must be handled together because of emissive materials.
    pub fn update_materials_and_lights(
        &mut self,
        materials: &[Material],
        lights: &[Light],
        textures: &[Texture],
    ) {
        let mut unf = UnfinishedExecutions::new(self.instance.device());
        self.scene.update_materials_and_lights(
            materials,
            lights,
            textures,
            &mut self.tcmdm,
            &mut unf,
        );
        unf.wait_completion();
        self.request_new_frame = true;
    }

    /// Refresh binded resources.
    ///
    /// Must be called after updating textures shared with the realtime renderer.
    ///
    /// This involves rebuilding the pipeline and the descriptors and is quite costly in term of
    /// performance.
    pub fn refresh_binded_textures(&mut self) {
        self.scene.refresh_descriptors();
        let mut unf = UnfinishedExecutions::new(self.instance.device());
        let instance = &self.instance;
        let device = instance.device();
        let pipeline = build_raytracing_pipeline(
            &self.rploader,
            device.logical_clone(),
            &[self.frame_desc[0].layout, self.scene.descriptor.layout],
            self.integrator,
        );
        let sbt = build_sbt(
            instance.as_ref(),
            &self.rploader,
            &mut self.tcmdm,
            &pipeline,
            self.integrator,
            &mut unf,
        );
        self.pipeline = pipeline;
        self.sbt = sbt;
        unf.wait_completion();
    }

    /// Returns a reference to the output image.
    pub(crate) fn output_image(&self) -> &AllocatedImage {
        &self.out8_img
    }

    /// Draws a frame.
    ///
    /// This calculates a single light path step for each pixel. Some integrators require multiple
    /// calls to complete a light path.
    ///
    /// Check the [Integrator::steps_per_sample] function.
    pub(crate) fn draw_frame(
        &mut self,
        wait: &[vk::Semaphore],
        signal: &[vk::Semaphore],
        fence: vk::Fence,
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
            image: self.out32_img.image,
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
            image: self.out8_img.image,
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
            image: self.out32_img.image,
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
            image: self.out8_img.image,
            subresource_range,
        };
        let frame_index = frame_no % self.frame_desc.len();
        if self.request_new_frame {
            self.sample_scheduler.rewind();
        }
        let camera_persp = match self.scene.camera {
            Camera::Perspective(_) => true,
            Camera::Orthographic(_) => false,
        };
        let fd = RTFrameData {
            seed: self.rng.gen(),
            lights_no: self.scene.lights_no,
            pixel_offset: self.sample_scheduler.next().unwrap(),
            scene_radius: self.scene.meta.scene_radius,
            exposure: self.scene.meta.exposure,
            scene_size: [self.extent.width as f32, self.extent.height as f32],
            scene_centre: [
                self.scene.meta.scene_centre[0],
                self.scene.meta.scene_centre[1],
                self.scene.meta.scene_centre[2],
                0.0,
            ],
            camera_persp,
        };
        update_frame_data(fd, &mut self.frame_data[frame_index]);
        unsafe {
            vkdevice
                .begin_command_buffer(cmd, &cmd_begin)
                .expect("Failed to begin command");
            if self.request_new_frame {
                self.request_new_frame = false;
                let color = vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                };
                vkdevice.cmd_fill_buffer(
                    cmd,
                    self.integrator_data.buffer,
                    0,
                    self.integrator_data.size,
                    0,
                );
                vkdevice.cmd_clear_color_image(
                    cmd,
                    self.cumulative_img.image,
                    vk::ImageLayout::GENERAL,
                    &color,
                    &[subresource_range],
                );
                vkdevice.cmd_clear_color_image(
                    cmd,
                    self.out32_img.image,
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
                &self.sbt.rgen_addrs[0],
                &self.sbt.miss_addr,
                &self.sbt.hit_addr,
                &self.sbt.call_addr,
                self.extent.width,
                self.extent.height,
                1,
            );
            vkdevice.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
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
                self.out32_img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.out8_img.image,
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

            vkdevice
                .end_command_buffer(cmd)
                .expect("Failed to end command buffer");
        }
        device.submit(&self.ccmdm, &[submit_ci], fence);
    }

    /// Draws the raytraced image and produces a frame.
    ///
    /// Requires the amount of samples per pixel and a callback function that will be called after
    /// each sample per pixel is finished.
    ///
    /// Samples per pixel are analogous to completed light paths.
    ///
    /// This method is **NOT** expected to work in realtime, and a proper realtime variant is
    /// available with crate visibility. The realtime variant produces faster results by displaying
    /// unfinished light paths, whereas this function shows only completed ones.
    pub fn draw<F>(&mut self, spp: usize, callback: F) -> image::RgbaImage
    where
        F: Fn(),
    {
        self.request_new_frame = true;
        let mut sync = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                (
                    create_semaphore(self.instance.device().logical()),
                    create_fence(self.instance.device().logical(), true),
                )
            })
            .collect::<VecDeque<_>>();
        let mut last_semaphore = vk::Semaphore::null();
        let substep = spp * self.integrator.steps_per_sample();
        for i in 0..substep {
            let (signal_sem, signal_fence) = sync.pop_front().unwrap();
            unsafe {
                self.instance
                    .device()
                    .logical()
                    .wait_for_fences(&[signal_fence], true, u64::MAX)
                    .expect("Failed to wait on fence");
                self.instance
                    .device()
                    .logical()
                    .reset_fences(&[signal_fence])
                    .expect("Failed to reset fence");
            }
            if i == 0 {
                self.draw_frame(&[], &[signal_sem], signal_fence, i);
            } else {
                self.draw_frame(&[last_semaphore], &[signal_sem], signal_fence, i);
            }
            last_semaphore = signal_sem;
            sync.push_back((signal_sem, signal_fence));
            if i % self.integrator.steps_per_sample() == 0 {
                callback();
            }
        }
        let fences = sync
            .iter()
            .map(|(_, fence)| fence)
            .copied()
            .collect::<Vec<_>>();
        unsafe {
            self.instance
                .device()
                .logical()
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("Failed to wait on fences");
            for (sem, fence) in sync {
                self.instance
                    .device()
                    .logical()
                    .destroy_semaphore(sem, None);
                self.instance.device().logical().destroy_fence(fence, None);
            }
        }
        let mut gcmdm = CommandManager::new(
            self.instance.device().logical_clone(),
            self.instance.device().graphic_queue(),
            1,
        );
        export(
            self.instance.as_ref(),
            self.output_image(),
            &mut gcmdm,
            self.extent.width,
            self.extent.height,
        )
    }
}

#[cfg(feature = "vulkan-interactive")]
impl TryFrom<&RealtimeRenderer> for RayTraceRenderer<PresentInstance> {
    type Error = std::io::Error;

    fn try_from(value: &RealtimeRenderer) -> Result<Self, Self::Error> {
        if value.instance().supports_raytrace() {
            let scene = RayTraceScene::<PresentInstance>::from(value.scene());
            let (width, height) = value.render_size();
            let extent = vk::Extent2D { width, height };
            Ok(init_rt(value.instance(), scene, extent))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "GPU does not support raytracing",
            ))
        }
    }
}

/// Initialize the raytracer.
fn init_rt<T: Instance + Send + Sync>(
    instance: Arc<T>,
    scene: RayTraceScene<T>,
    extent: vk::Extent2D,
) -> RayTraceRenderer<T> {
    const AVG_DESC: [(vk::DescriptorType, f32); 2] = [
        (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
        (vk::DescriptorType::STORAGE_IMAGE, 1.0),
    ];
    let device = instance.device();
    let mut unf = UnfinishedExecutions::new(device);
    let rploader = RTPipelineLoader::new(instance.instance(), device.logical());
    let transfer_queue = device.transfer_queue();
    let compute_queue = device.compute_queue();
    let mut tcmdm = CommandManager::new(device.logical_clone(), transfer_queue, 1);
    let ccmdm = CommandManager::new(device.logical_clone(), compute_queue, 15);
    let sample_scheduler = WorkScheduler::new();
    let mut dm = DescriptorSetManager::new(
        device.logical_clone(),
        &AVG_DESC,
        instance.desc_layout_cache(),
    );
    let out32_img = create_storage_image(instance.as_ref(), &mut tcmdm, extent, &mut unf, false);
    let out8_img = instance.allocator().create_image_gpu(
        "Present image raytracer",
        vk::Format::R8G8B8A8_SRGB,
        extent,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC // src is used to export the rendered image
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let cumulative_img =
        create_storage_image(instance.as_ref(), &mut tcmdm, extent, &mut unf, true);
    let push_constants = build_push_constants(&scene.camera, extent);
    let frame_data = (0..FRAMES_IN_FLIGHT)
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
    let integrator = Integrator::default();
    let integrator_data = create_integrator_data(instance.as_ref(), integrator, extent);
    let frame_desc = build_descriptor(
        &mut dm,
        &frame_data,
        &integrator_data,
        &cumulative_img,
        &out32_img,
    );
    let pipeline = build_raytracing_pipeline(
        &rploader,
        device.logical_clone(),
        &[frame_desc[0].layout, scene.descriptor.layout],
        integrator,
    );
    let sbt = build_sbt(
        instance.as_ref(),
        &rploader,
        &mut tcmdm,
        &pipeline,
        integrator,
        &mut unf,
    );
    let rng = Xoshiro128PlusPlus::from_entropy();
    unf.wait_completion();
    RayTraceRenderer {
        scene,
        push_constants,
        frame_data,
        rng,
        cumulative_img,
        out32_img,
        out8_img,
        integrator,
        integrator_data,
        sample_scheduler,
        request_new_frame: true,
        frame_desc,
        extent,
        sbt,
        pipeline,
        dm,
        tcmdm,
        ccmdm,
        rploader,
        instance,
    }
}

/// Build the engine-level descriptors.
/// A single one for each one of the FRAMES_IN_FLIGHT.
fn build_descriptor(
    dm: &mut DescriptorSetManager,
    frame_data: &[AllocatedBuffer],
    integrator_data: &AllocatedBuffer,
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
                .bind_buffer(
                    integrator_data,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::RAYGEN_KHR,
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
        // one of the two images (self.cumulative) requires clearing, which in turn requires
        // transfer_dst
        vk::ImageUsageFlags::TRANSFER_DST
    } else {
        vk::ImageUsageFlags::empty()
    };
    let out_img = mm.create_image_gpu(
        "RT storage image",
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
    let fence = device.submit_immediate(tcmdm, command);
    unf.add_fence(fence);
    out_img
}

/// Increases `size` until it is aligned to `align_to`
fn roundup_alignment(size: u64, align_to: u64) -> u64 {
    (size + (align_to - 1)) & !(align_to - 1)
}

/// Calculates and allocates the Shader Binding Table (SBT).
fn build_sbt<T: Instance>(
    instance: &T,
    rploader: &RTPipelineLoader,
    tcmdm: &mut CommandManager,
    pipeline: &Pipeline,
    integrator: Integrator,
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
    let mut group_index = 0;

    let group_count = integrator.raygen_count();
    let rgen_groups = unsafe {
        rploader.get_ray_tracing_shader_group_handles(
            pipeline.pipeline,
            group_index,
            group_count,
            group_count as usize * size_handle as usize,
        )
    }
    .expect("Failed to retrieve shader handle");
    group_index += group_count;
    let mut rgen_addrs = Vec::with_capacity(group_count as usize);
    for rgen_group in rgen_groups.chunks_exact(size_handle as usize) {
        rgen_addrs.push(
            // in the NVIDIA example it's written that stride and size for rgen must be the same
            vk::StridedDeviceAddressRegionKHR {
                device_address: data.len() as u64,
                stride: roundup_alignment(align_handle, align_group),
                size: roundup_alignment(align_handle, align_group),
            },
        );
        data.extend_from_slice(rgen_group);
        //slightly different from the miss/hit/callable: each rgen is treated as a single group
        let missing_bytes = padding(data.len() as u64, align_group) as usize;
        data.extend(repeat(0).take(missing_bytes));
    }
    // load two miss groups: normal ray and shadow ray miss
    let miss_offset = data.len() as u64;
    let miss_groups = unsafe {
        rploader.get_ray_tracing_shader_group_handles(
            pipeline.pipeline,
            group_index,
            2,
            2 * size_handle as usize,
        )
    }
    .expect("Failed to retrieve shader handle");
    group_index += 2;
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
        rploader.get_ray_tracing_shader_group_handles(
            pipeline.pipeline,
            group_index,
            1,
            2 * size_handle as usize, // 1 group but has 2 shaders
        )
    }
    .expect("Failed to retrieve shader handle");
    group_index += 1;
    for hit in shader_group.chunks_exact(size_handle as usize) {
        data.extend_from_slice(hit);
        if align_handle != size_handle as u64 {
            let missing_bytes = padding(data.len() as u64, align_handle) as usize;
            data.extend(repeat(0).take(missing_bytes));
        }
    }
    let missing_bytes = padding(data.len() as u64, align_group) as usize;
    data.extend(repeat(0).take(missing_bytes));
    let mut hit_addr = vk::StridedDeviceAddressRegionKHR {
        device_address: hit_offset,
        stride: size_handle,
        size: 2 * size_handle_aligned,
    };

    // load multiple callables. Retrieve them all at once
    let call_offset = data.len() as u64;
    let total_call = SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE + SBT_MATERIAL_TYPES * SBT_MATERIAL_STRIDE;
    let callables_group = unsafe {
        rploader.get_ray_tracing_shader_group_handles(
            pipeline.pipeline,
            group_index,
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
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_copy_buffer(cmd, cpu_buf.buffer, gpu_buf.buffer, &[buffer_copy]);
        }
    };
    let fence = device.submit_immediate(tcmdm, command);
    unf.add(fence, cpu_buf);
    // calculates the groups addresses
    let sbt_addr_info = vk::BufferDeviceAddressInfo {
        s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
        p_next: ptr::null(),
        buffer: gpu_buf.buffer,
    };
    let base_addr = unsafe { device.logical().get_buffer_device_address(&sbt_addr_info) };
    for rgen_addr in &mut rgen_addrs {
        rgen_addr.device_address += base_addr;
    }
    miss_addr.device_address += base_addr;
    hit_addr.device_address += base_addr;
    call_addr.device_address += base_addr;
    ShaderBindingTable {
        rgen_addrs,
        miss_addr,
        hit_addr,
        call_addr,
        _buffer: gpu_buf,
    }
}

/// Calculates push constants.
fn build_push_constants(camera: &Camera, extent: vk::Extent2D) -> [u8; 128] {
    // build matrices
    let view_inv = camera.look_at_rh().invert().unwrap();
    let mut proj = camera.projection(extent.width, extent.height);
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

/// Creates the buffer used to store temporary integrator data.
fn create_integrator_data<T: Instance>(
    instance: &T,
    integrator: Integrator,
    extent: vk::Extent2D,
) -> AllocatedBuffer {
    let allocator = instance.allocator();
    // TRANSFER_DST required to clear the buffer
    match integrator {
        Integrator::DIRECT => allocator.create_buffer(
            "unused",
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        ),
        Integrator::PATH_TRACE => {
            let size =
                (extent.width * extent.height) as u64 * std::mem::size_of::<PTLastVertex>() as u64;
            allocator.create_buffer(
                "PT storage",
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
        }
    }
}

/// Updates the buffer containing per-frame data.
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
/// In realtime raytracing the number of samples per pixel is not fixed so this process has to
/// be incremental.
///
/// This iterator is unlimited and will never produce None.
struct WorkScheduler {
    current: Vec<([f32; 2], [f32; 2])>,
    next: Vec<([f32; 2], [f32; 2])>,
}

impl WorkScheduler {
    /// Creates a new iterator.
    pub fn new() -> Self {
        WorkScheduler {
            current: vec![([0.0, 0.0], [1.0, 1.0])],
            next: Vec::new(),
        }
    }

    /// Resets this iterator
    pub fn rewind(&mut self) {
        *self = WorkScheduler::new()
    }
}

impl Iterator for WorkScheduler {
    type Item = [f32; 2];

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(area) = self.current.pop() {
            let middle = [(area.0[0] + area.1[0]) / 2.0, (area.0[1] + area.1[1]) / 2.0];
            self.next.push((area.0, middle));
            self.next.push((middle, area.1));
            self.next
                .push(([middle[0], area.0[1]], [area.1[0], middle[1]]));
            self.next
                .push(([area.0[0], middle[1]], [middle[0], area.1[1]]));
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
    use crate::vulkan::scene::RayTraceScene;
    use crate::{parse, RayTraceInstance};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn init(instance: Arc<RayTraceInstance>) -> Option<RayTraceScene<RayTraceInstance>> {
        let _ = env_logger::builder().is_test(true).try_init();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("cube.glaze");
        Some(RayTraceScene::<RayTraceInstance>::new(
            Arc::clone(&instance),
            parse(path).unwrap(),
        ))
    }

    #[test]
    fn load_raytrace() {
        if let Some(instance) = RayTraceInstance::new() {
            let _ = RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), None, 2, 2);
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }

    #[test]
    fn draw_outlive() {
        if let Some(instance) = RayTraceInstance::new() {
            let instance = Arc::new(instance);
            let parsed = init(Arc::clone(&instance));
            let mut renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::clone(&instance), parsed, 2, 2);
            let _ = renderer.draw(1, || {});
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }

    #[test]
    fn save_to_disk() -> Result<(), std::io::Error> {
        if let Some(instance) = RayTraceInstance::new() {
            let instance = Arc::new(instance);
            let parsed = init(Arc::clone(&instance));
            let dir = tempdir()?;
            let file = dir.path().join("save.png");
            let mut renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::clone(&instance), parsed, 2, 2);
            let image = renderer.draw(1, || {});
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
            let instance = Arc::new(instance);
            let parsed = init(Arc::clone(&instance));
            let mut renderer =
                RayTraceRenderer::<RayTraceInstance>::new(Arc::clone(&instance), parsed, 2, 2);
            renderer.change_resolution(4, 4);
            let image = renderer.draw(1, || {});
            assert_eq!(image.width(), 4);
            assert_eq!(image.height(), 4);
        } else {
            // SKIPPED does not exists in cargo test...
        }
        Ok(())
    }
}
