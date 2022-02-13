use super::cmd::CommandManager;
use super::descriptor::{Descriptor, DescriptorSetManager};
use super::imgui::ImguiRenderer;
use super::instance::{Instance, PresentInstance};
use super::memory::AllocatedBuffer;
use super::pipeline::{Pipeline, PipelineBuilder};
use super::renderpass::RenderPass;
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use super::sync::PresentSync;
use super::{AllocatedImage, UnfinishedExecutions};
use crate::{include_shader, Camera, Light, Material, RayTraceRenderer};
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use std::io::ErrorKind;
use std::ptr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Number of frames prepared by the CPU while waiting for the GPU.
const FRAMES_IN_FLIGHT: usize = 2;

/// Stats about the renderer.
#[derive(Debug, Copy, Clone)]
pub struct Stats {
    /// Average frames per second.
    pub fps: f32,
    /// Average draw calls per frame in the last second.
    pub avg_draw_calls: f32,
}

/// Per-frame data passed to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FrameData {
    /// Projection matrix * View matrix.
    pub projview: Matrix4<f32>,
    /// Time elapsed since scene loading, in seconds.
    pub frame_time: f32,
}

/// Contains the required helpers to send a FrameData struct to the GPU shaders.
#[derive(Debug)]
struct FrameDataBuf {
    /// Buffer containing the FrameData.
    buffer: AllocatedBuffer,
    /// Actual data on the CPU.
    data: FrameData,
    /// Descriptor set containing the per-frame data (set 0).
    descriptor: Descriptor,
}

/// Realtime renderer capable of rendering a scene to a presentation surface.
pub struct RealtimeRenderer {
    /// In case the previewed image will use the raytraced renderer.
    /// The RayTraceRenderer lacks the support for handling presentation, so it cannot do it on its
    /// own. This field is None if raytracing is not supported.
    raytracer: Option<RayTraceRenderer<PresentInstance>>,
    /// True if the presented image should come from the raytracer.
    pub use_raytracer: bool,
    /// Stores the images that will be used to display raytrace outputs.
    raytrace_output: AllocatedImage,
    /// Descriptor containing the raytrace_output, will be passed to the copy_pipeline.
    raytrace_copy_desc: Descriptor,
    /// Scene to be rendered.
    scene: Option<VulkanScene>,
    /// Swapchain of the renderer.
    swapchain: Swapchain,
    /// Sampler used to copy the forward pass attachment to the swapchain image.
    copy_sampler: vk::Sampler,
    /// Pipeline used to copy the forward pass attachment to the swapchain image.
    copy_pipeline: Pipeline,
    /// Forward pass.
    forward_pass: RenderPass,
    /// UI Renderer.
    imgui_renderer: ImguiRenderer,
    /// Manager for the descriptor pools and sets.
    dm: DescriptorSetManager,
    /// Graphic Command Manager. Manager for the command pools and buffers of a graphic queue.
    gcmdm: CommandManager,
    /// Transfer Commmand Manager. Used solely to update materials.
    tcmdm: CommandManager,
    /// Synchronization structures for each frame.
    sync: PresentSync<FRAMES_IN_FLIGHT>,
    /// Per-frame data, for each frame.
    frame_data: [FrameDataBuf; FRAMES_IN_FLIGHT],
    /// Clear color of the forward pass.
    clear_color: [f32; 4],
    /// Instant when the rendering of the current scene started.
    start_time: Instant,
    /// Scaling factor of the rendered area (the renderered area is swapchain image * scale).
    render_scale: f32,
    /// Current frame index.
    frame_no: usize,
    /// Stats about the renderer.
    stats: InternalStats,
    /// Vulkan instance.
    instance: Arc<PresentInstance>,
}

impl RealtimeRenderer {
    /// Creates a new realtime renderer.
    /// Takes a vukan instance, an imgui context, the window dimensions and the render scale.
    /// # Examples
    /// basic usage:
    /// ``` no_run
    /// let mut event_loop = winit::event_loop::EventLoop::new();
    /// let window = winit::window::Window::new(&event_loop).unwrap();
    /// let mut imgui = imgui::Context::create();
    /// let instance = glaze::PresentInstance::new(&window).expect("No GPU found");
    /// let arc_instance = std::sync::Arc::new(instance);
    /// let renderer = glaze::RealtimeRenderer::create(
    ///     arc_instance,
    ///     &mut imgui,
    ///     window.inner_size().width,
    ///     window.inner_size().height,
    ///     1.0,
    /// );
    /// ```
    pub fn create(
        instance: Arc<PresentInstance>,
        imgui: &mut imgui::Context,
        window_width: u32,
        window_height: u32,
        render_scale: f32,
    ) -> Self {
        let avg_desc = [
            (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1.0),
        ];
        let mut dm = DescriptorSetManager::new(
            instance.device().logical_clone(),
            &avg_desc,
            instance.desc_layout_cache(),
        );
        let mm = instance.allocator();
        let gcmdm = CommandManager::new(
            instance.device().logical_clone(),
            instance.device().graphic_queue().idx,
            15,
        );
        let tcmdm = CommandManager::new(
            instance.device().logical_clone(),
            instance.device().transfer_queue().idx,
            1,
        );
        let swapchain = Swapchain::create(instance.clone(), window_width, window_height);
        let render_size = vk::Extent2D {
            width: (window_width as f32 * render_scale) as u32,
            height: (window_height as f32 * render_scale) as u32,
        };
        let mut frame_data = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let buffer = mm.create_buffer(
                "FrameData",
                std::mem::size_of::<FrameData>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            );
            let descriptor = dm
                .new_set()
                .bind_buffer(
                    &buffer,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                )
                .build();
            let data = FrameData {
                frame_time: 0.0,
                projview: Matrix4::identity(),
            };
            frame_data.push(FrameDataBuf {
                buffer,
                data,
                descriptor,
            });
        }
        let sync = PresentSync::create(instance.device().logical_clone());
        let copy_sampler = create_copy_sampler(instance.device().logical());
        let clear_color = [0.15, 0.15, 0.15, 1.0];
        let mut forward_pass = RenderPass::forward(
            instance.device().logical_clone(),
            copy_sampler,
            mm,
            &mut dm,
            render_size,
        );
        forward_pass.clear_color[0].color.float32 = clear_color;
        let imgui_renderer = ImguiRenderer::new(imgui, instance.clone(), dm.cache(), &swapchain);
        let copy_pipeline = create_copy_pipeline(
            instance.device().logical_clone(),
            swapchain.extent(),
            swapchain.renderpass(),
            &[forward_pass.copy_descriptor.layout],
        );
        let (raytrace_output, raytrace_copy_desc) =
            raytracer_copy_helpers(&instance, &mut dm, copy_sampler, render_size);
        RealtimeRenderer {
            raytracer: None,
            use_raytracer: false,
            raytrace_output,
            raytrace_copy_desc,
            scene: None,
            swapchain,
            copy_sampler,
            copy_pipeline,
            forward_pass,
            imgui_renderer,
            dm,
            gcmdm,
            tcmdm,
            sync,
            frame_data: frame_data.try_into().unwrap(),
            clear_color,
            start_time: Instant::now(),
            render_scale,
            frame_no: 0,
            stats: InternalStats::default(),
            instance,
        }
    }

    /// Waits until all the queued frames have been rendered.
    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    /// Returns the current render scale.
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    /// Returns the current background color.
    ///
    /// Color is expressed as RGB floats in the range [0, 1].
    pub fn get_clear_color(&self) -> [f32; 3] {
        [
            self.clear_color[0],
            self.clear_color[1],
            self.clear_color[2],
        ]
    }

    /// Sets the background color.
    ///
    /// Color is expressed as RGB floats in the range [0, 1].
    pub fn set_clear_color(&mut self, color: [f32; 3]) {
        self.clear_color = [color[0], color[1], color[2], 1.0];
        self.forward_pass.clear_color[0].color.float32 = self.clear_color;
    }

    /// Returns the current stats of the renderer.
    pub fn stats(&self) -> Stats {
        self.stats.last_val
    }

    /// Reuturns the vulkan instance used by the renderer.
    pub fn instance(&self) -> &PresentInstance {
        &self.instance
    }

    /// Returns the current scene being rendered.
    ///
    /// Returns None if no scene is loaded.
    pub fn scene(&self) -> Option<&VulkanScene> {
        self.scene.as_ref()
    }

    /// Returns a mutable reference to the scene being renderered.
    ///
    /// Returns None if no scene is loaded.
    pub fn scene_mut(&mut self) -> Option<&mut VulkanScene> {
        self.scene.as_mut()
    }

    /// Returns the camera of the current scene.
    ///
    /// Returns None if no scene is loaded.
    pub fn camera(&self) -> Option<&Camera> {
        if let Some(scene) = &self.scene {
            Some(&scene.current_cam)
        } else {
            None
        }
    }

    /// Updates the current camera.
    /// Updates also the raytracer camera, if this exists
    pub fn set_camera(&mut self, camera: Camera) {
        if let Some(rt) = &mut self.raytracer {
            rt.update_camera(&camera);
        }
        if let Some(scene) = &mut self.scene {
            scene.current_cam = camera;
        }
    }

    /// Changes the render size of the renderer.
    ///
    /// The render size will be (window_width * render_scale, window_height * render_scale).
    pub fn update_render_size(&mut self, window_width: u32, window_height: u32, scale: f32) {
        self.wait_idle();
        self.render_scale = scale;
        self.swapchain.recreate(window_width, window_height);
        self.imgui_renderer.update_swapchain(&self.swapchain);
        let render_size = vk::Extent2D {
            width: (window_width as f32 * scale) as u32,
            height: (window_height as f32 * scale) as u32,
        };
        let mm = self.instance.allocator();
        let mut forward_pass = RenderPass::forward(
            self.instance.device().logical_clone(),
            self.copy_sampler,
            mm,
            &mut self.dm,
            render_size,
        );
        forward_pass.clear_color[0].color.float32 = self.clear_color;
        self.forward_pass = forward_pass;
        self.copy_pipeline = create_copy_pipeline(
            self.instance.device().logical_clone(),
            self.swapchain.extent(),
            self.swapchain.renderpass(),
            &[self.forward_pass.copy_descriptor.layout],
        );
        let (rt_out, rt_desc) =
            raytracer_copy_helpers(&self.instance, &mut self.dm, self.copy_sampler, render_size);
        self.raytrace_output = rt_out;
        self.raytrace_copy_desc = rt_desc;
        if let Some(scene) = &mut self.scene {
            scene.deinit_pipelines();
            scene.init_pipelines(
                render_size,
                self.instance.device().logical_clone(),
                self.forward_pass.renderpass,
                self.frame_data[0].descriptor.layout,
            );
        }
        if let Some(raytracer) = &mut self.raytracer {
            raytracer.change_resolution(render_size.width, render_size.height);
        }
    }

    /// Loads a new scene in the renderer.
    ///
    /// The scene must be parsed and loaded externally.
    /// # Examples
    /// Basic usage:
    /// ``` no_run
    /// // init renderer
    /// let mut event_loop = winit::event_loop::EventLoop::new();
    /// let window = winit::window::Window::new(&event_loop).unwrap();
    /// let mut imgui = imgui::Context::create();
    /// let instance = glaze::PresentInstance::new(&window).expect("No GPU found");
    /// let arc_instance = std::sync::Arc::new(instance);
    /// let mut renderer = glaze::RealtimeRenderer::create(
    ///     arc_instance.clone(),
    ///     &mut imgui,
    ///     window.inner_size().width,
    ///     window.inner_size().height,
    ///     1.0,
    /// );
    /// // parse scene
    /// let scene_path = "/path/to/scene";
    /// let parsed = glaze::parse(scene_path).expect("Failed to parse scene");
    /// let (wchan, rchan) = std::sync::mpsc::channel();
    /// let loaded = glaze::VulkanScene::load(arc_instance, parsed, wchan).unwrap();
    /// renderer.change_scene(loaded);
    /// ```
    pub fn change_scene(&mut self, mut loaded: VulkanScene) {
        self.wait_idle();
        self.scene.take(); // drop previous scene, if existing
        let render_size = vk::Extent2D {
            width: (self.swapchain.extent().width as f32 * self.render_scale) as u32,
            height: (self.swapchain.extent().height as f32 * self.render_scale) as u32,
        };
        loaded.init_pipelines(
            render_size,
            self.instance.device().logical_clone(),
            self.forward_pass.renderpass,
            self.frame_data[0].descriptor.layout,
        );
        self.raytracer = if self.instance.supports_raytrace() {
            Some(
                RayTraceRenderer::<PresentInstance>::from_realtime(
                    self.instance.clone(),
                    &mut loaded,
                    render_size.width,
                    render_size.height,
                    FRAMES_IN_FLIGHT,
                )
                .unwrap(),
            )
        } else {
            None
        };
        self.imgui_renderer.load_scene_textures(&loaded);
        self.scene = Some(loaded);
        self.start_time = Instant::now();
    }

    /// Updates a single material.
    ///
    /// `old` is the index of the old material.
    pub fn change_material(&mut self, old: u16, new: Material) {
        self.wait_idle();
        if let Some(scene) = &mut self.scene {
            let render_size = vk::Extent2D {
                width: (self.swapchain.extent().width as f32 * self.render_scale) as u32,
                height: (self.swapchain.extent().height as f32 * self.render_scale) as u32,
            };
            let mut unf = UnfinishedExecutions::new(self.instance.device());
            scene.update_material(
                old,
                new,
                &mut self.tcmdm,
                &mut unf,
                self.forward_pass.renderpass,
                self.frame_data[0].descriptor.layout,
                render_size,
            );
            if let Some(raytracer) = &mut self.raytracer {
                raytracer
                    .scene
                    .update_materials(scene.materials(), &mut self.tcmdm, &mut unf);
                raytracer.request_new_frame = true;
            }
            unf.wait_completion();
        }
    }

    /// Adds a new light to the scene.
    pub fn add_light(&mut self, new: Light) {
        self.wait_idle();
        if let Some(scene) = &mut self.scene {
            scene.add_light(new);
            if let Some(raytracer) = &mut self.raytracer {
                let mut unf = UnfinishedExecutions::new(self.instance.device());
                raytracer
                    .scene
                    .update_lights(scene.lights(), &mut self.tcmdm, &mut unf);
                unf.wait_completion();
            }
        }
    }

    /// Removes from the scene the light with the given id.
    ///
    /// If the light does not exist, nothing happens.
    pub fn remove_light(&mut self, id: usize) {
        self.wait_idle();
        if let Some(scene) = &mut self.scene {
            scene.remove_light(id);
            if let Some(raytracer) = &mut self.raytracer {
                let mut unf = UnfinishedExecutions::new(self.instance.device());
                raytracer
                    .scene
                    .update_lights(scene.lights(), &mut self.tcmdm, &mut unf);
                unf.wait_completion();
            }
        }
    }

    /// Replaces a light in the scene.
    pub fn change_light(&mut self, old: usize, new: Light) {
        self.wait_idle();
        if let Some(scene) = &mut self.scene {
            scene.update_light(old, new);
            if let Some(raytracer) = &mut self.raytracer {
                let mut unf = UnfinishedExecutions::new(self.instance.device());
                raytracer
                    .scene
                    .update_lights(scene.lights(), &mut self.tcmdm, &mut unf);
                raytracer.request_new_frame = true;
                unf.wait_completion();
            }
        }
    }

    pub fn get_raytrace(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<RayTraceRenderer<PresentInstance>, std::io::Error> {
        if self.instance.supports_raytrace() {
            if let Some(scene) = &mut self.scene {
                let instance = self.instance.clone();
                RayTraceRenderer::<PresentInstance>::from_realtime(
                    instance,
                    scene,
                    width,
                    height,
                    FRAMES_IN_FLIGHT,
                )
            } else {
                Err(std::io::Error::new(ErrorKind::InvalidData, "Missing scene"))
            }
        } else {
            Err(std::io::Error::new(
                ErrorKind::Unsupported,
                "The video card does not support raytraced rendering",
            ))
        }
    }

    /// Draws a single frame.
    ///
    /// If there is a GUI to draw, the `imgui_data` should contain the GUI data.
    pub fn draw_frame(&mut self, imgui_data: Option<&imgui::DrawData>) {
        self.stats.update();
        let frame_sync = self.sync.get(self.frame_no);
        let device = self.instance.device().logical();
        frame_sync.wait_acquire(device);
        if let Some(acquired) = self.swapchain.acquire_next_image(frame_sync) {
            let current_time = Instant::now();
            let frame_data = &mut self.frame_data[self.frame_no % FRAMES_IN_FLIGHT];
            let mut graphics_wait = [frame_sync.image_available];
            let compute_finished = [frame_sync.compute_finished];
            let present_ready = [frame_sync.render_finished];
            frame_data.data.frame_time = (current_time - self.start_time).as_secs_f32();
            if self.use_raytracer {
                self.raytracer.as_mut().unwrap().draw_frame(
                    frame_sync.image_available,
                    compute_finished[0],
                    &self.raytrace_output,
                    self.frame_no,
                );
                // swap signal: the graphic queue have to wait the raytrace, not the image
                // available
                graphics_wait[0] = compute_finished[0];
            }
            let cmd = self.gcmdm.get_cmd_buffer();
            let cmd_ci = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
            };
            unsafe {
                device
                    .begin_command_buffer(cmd, &cmd_ci)
                    .expect("Failed to begin command buffer");
                // if raytracer is not requested, draw the objects
                self.forward_pass.begin(cmd);
                if let Some(scene) = &self.scene {
                    if !self.use_raytracer {
                        let ar = self.swapchain.extent().width as f32
                            / self.swapchain.extent().height as f32;
                        draw_objects(scene, ar, frame_data, device, cmd, &mut self.stats);
                    }
                }
                self.forward_pass.end(cmd);
                acquired.renderpass.begin(cmd);
                if !self.use_raytracer {
                    copy_to_swapchain(
                        device,
                        cmd,
                        &self.copy_pipeline,
                        &self.forward_pass.copy_descriptor,
                        &mut self.stats,
                    );
                } else {
                    copy_to_swapchain(
                        device,
                        cmd,
                        &self.copy_pipeline,
                        &self.raytrace_copy_desc,
                        &mut self.stats,
                    );
                }
                // draw ui directly on the swapchain
                // tried doing it on its own attachment but results in blending problems
                if let Some(dd) = imgui_data {
                    if dd.total_vtx_count > 0 {
                        self.imgui_renderer.draw(cmd, dd, &mut self.stats);
                    }
                }
                acquired.renderpass.end(cmd);
                device
                    .end_command_buffer(cmd)
                    .expect("Failed to end command buffer");
            }
            let submit_ci = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: graphics_wait.len() as u32,
                p_wait_semaphores: graphics_wait.as_ptr(),
                p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                signal_semaphore_count: present_ready.len() as u32,
                p_signal_semaphores: present_ready.as_ptr(),
            };
            let swapchains = [self.swapchain.raw_handle()];
            let present_ci = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                p_next: ptr::null(),
                wait_semaphore_count: present_ready.len() as u32,
                p_wait_semaphores: present_ready.as_ptr(),
                swapchain_count: swapchains.len() as u32,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: &acquired.index,
                p_results: ptr::null_mut(),
            };
            let queue = self.instance.device().graphic_queue();
            unsafe {
                device
                    .queue_submit(queue.queue, &[submit_ci], frame_sync.acquire)
                    .expect("Failed to submit render task");
            }
            self.swapchain.queue_present(queue.queue, &present_ci);
            self.frame_no += 1;
            self.stats.done_frame();
        } else {
            // out of date swapchain. the resize is called by winit so wait next frame
        }
    }
}

impl Drop for RealtimeRenderer {
    fn drop(&mut self) {
        self.wait_idle();
        unsafe {
            self.instance
                .device()
                .logical()
                .destroy_sampler(self.copy_sampler, None);
        };
    }
}

/// Draws all objects belonging to the loaded VulkanScene.
unsafe fn draw_objects(
    scene: &VulkanScene,
    ar: f32,
    frame_data: &mut FrameDataBuf,
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    stats: &mut InternalStats,
) {
    let cam = &scene.current_cam;
    let mut proj = cam.projection(ar);
    proj[1][1] *= -1.0;
    let view = cam.look_at_rh();
    frame_data.data.projview = proj * view;
    let mut current_shader = None;
    //write frame_data to the buffer
    let buf_ptr = frame_data
        .buffer
        .allocation()
        .mapped_ptr()
        .expect("Failed to map buffer")
        .cast()
        .as_ptr();
    std::ptr::copy_nonoverlapping(&frame_data.data, buf_ptr, 1);
    device.cmd_bind_vertex_buffers(cmd, 0, &[scene.vertex_buffer.buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, scene.index_buffer.buffer, 0, vk::IndexType::UINT32); //bind once, use firts_index as offset
    for obj in &scene.meshes {
        let (shader, desc) = &scene.materials_desc[obj.material as usize];
        let pipeline = scene.pipelines.get(shader).unwrap(); // this definitely exists
        if current_shader.is_none() || shader != current_shader.unwrap() {
            current_shader = Some(shader);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
        }
        let empty_vec = Vec::with_capacity(0);
        let instances = scene.instances.get(&obj.mesh_id).unwrap_or(&empty_vec);
        for instance in instances {
            let (_, po_desc) = scene.transforms[*instance as usize];
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[frame_data.descriptor.set, desc.set, po_desc.set],
                &[],
            );
            device.cmd_draw_indexed(cmd, obj.index_count, 1, obj.index_offset, 0, 0);
            stats.done_draw_call();
        }
    }
}

/// Creates the sampler used to copy the forward pass color attachment into the swapchain image.
fn create_copy_sampler(device: &ash::Device) -> vk::Sampler {
    let ci = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SamplerCreateFlags::empty(),
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
        mip_lod_bias: 0.0,
        anisotropy_enable: vk::FALSE,
        max_anisotropy: 1.0,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        unnormalized_coordinates: vk::FALSE,
    };
    unsafe {
        device
            .create_sampler(&ci, None)
            .expect("Failed to create sampler")
    }
}

/// Creates the pipeline used to copy the forward pass color attachment into the swapchain image.
fn create_copy_pipeline(
    device: Arc<ash::Device>,
    extent: vk::Extent2D,
    renderpass: vk::RenderPass,
    dset_layout: &[vk::DescriptorSetLayout],
) -> Pipeline {
    let mut builder = PipelineBuilder::default();
    let vs = include_shader!("blit.vert");
    let fs = include_shader!("blit.frag");
    builder.push_shader(vs, "main", vk::ShaderStageFlags::VERTEX);
    builder.push_shader(fs, "main", vk::ShaderStageFlags::FRAGMENT);
    // vertices will be hard-coded (fullscreen)
    builder.binding_descriptions = Vec::with_capacity(0);
    builder.attribute_descriptions = Vec::with_capacity(0);
    // deactivate depth stencil
    builder.no_depth();
    builder.rasterizer.cull_mode = vk::CullModeFlags::NONE;
    builder.build(device, renderpass, extent, dset_layout)
}

/// Copies an image to the swapchain using a fragment shader
/// VkCmdCopy and VkCmdBlit are not used because the swapchain is not guaranteed to support
/// TRANSFER_DST
fn copy_to_swapchain(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    copy_pip: &Pipeline,
    desc: &Descriptor,
    stats: &mut InternalStats,
) {
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, copy_pip.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            copy_pip.layout,
            0,
            &[desc.set],
            &[],
        );
        device.cmd_draw(cmd, 3, 1, 0, 0);
        stats.done_draw_call();
    }
}

fn raytracer_copy_helpers(
    instance: &PresentInstance,
    dm: &mut DescriptorSetManager,
    sampler: vk::Sampler,
    extent: vk::Extent2D,
) -> (AllocatedImage, Descriptor) {
    let raytrace_output = instance.allocator().create_image_gpu(
        "Raytracer output",
        vk::Format::R8G8B8A8_SRGB,
        extent,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let raytrace_copy_desc = dm
        .new_set()
        .bind_image(
            &raytrace_output,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        )
        .build();
    (raytrace_output, raytrace_copy_desc)
}

// not frame-time precise (counts after the CPU time, not the GPU) but the average should be correct
pub struct InternalStats {
    time_start: Instant,
    frame_count: u32,
    draw_calls: u32,
    last_val: Stats,
}

impl Default for InternalStats {
    fn default() -> Self {
        Self {
            time_start: Instant::now(),
            frame_count: 0,
            draw_calls: 0,
            last_val: Stats {
                fps: 0.0,
                avg_draw_calls: 0.0,
            },
        }
    }
}

impl InternalStats {
    fn update(&mut self) {
        let elapsed = Duration::as_secs_f32(&(Instant::now() - self.time_start));
        if elapsed > 1.0 {
            self.last_val = Stats {
                fps: self.frame_count as f32 / elapsed,
                avg_draw_calls: self.draw_calls as f32 / self.frame_count as f32,
            };
            self.frame_count = 0;
            self.draw_calls = 0;
            self.time_start = Instant::now();
        }
    }

    pub fn done_draw_call(&mut self) {
        self.draw_calls += 1;
    }

    pub fn done_frame(&mut self) {
        self.frame_count += 1;
    }
}
