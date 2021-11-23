use super::cmd::CommandManager;
use super::descriptor::{
    Descriptor, DescriptorAllocator, DescriptorSetCreator, DescriptorSetLayoutCache,
};
use super::device::Device;
use super::imgui::ImguiDrawer;
use super::instance::{Instance, PresentInstance};
use super::memory::{AllocatedBuffer, MemoryManager};
use super::pipeline::{Pipeline, PipelineBuilder};
use super::renderpass::RenderPass;
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use super::sync::PresentSync;
use crate::{include_shader, Camera, ReadParsed};
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use std::ptr;
use std::time::{Duration, Instant};
use winit::window::Window;

const AVG_DESC: [(vk::DescriptorType, f32); 1] = [(vk::DescriptorType::UNIFORM_BUFFER, 1.0)];
const FRAMES_IN_FLIGHT: usize = 2;

#[derive(Debug, Copy, Clone)]
pub struct Stats {
    pub fps: f32,
    pub avg_draw_calls: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FrameData {
    pub projview: Matrix4<f32>,
    pub frame_time: f32,
}

#[derive(Debug)]
struct FrameDataBuf {
    buffer: AllocatedBuffer,
    data: FrameData,
    descriptor: Descriptor,
}

pub struct RealtimeRenderer {
    instance: PresentInstance,
    swapchain: Swapchain,
    descriptor_creator: DescriptorSetCreator,
    copy_sampler: vk::Sampler,
    copy_pipeline: Pipeline,
    forward_pass: RenderPass,
    imgui_renderer: ImguiDrawer,
    mm: MemoryManager,
    cmdm: CommandManager,
    sync: PresentSync<FRAMES_IN_FLIGHT>,
    scene: Option<VulkanScene>,
    frame_data: [FrameDataBuf; FRAMES_IN_FLIGHT],
    clear_color: [f32; 4],
    start_time: Instant,
    render_scale: f32,
    frame_no: usize,
    paused: bool,
    stats: InternalStats,
}

impl RealtimeRenderer {
    pub fn create(
        window: &Window,
        imgui: &mut imgui::Context,
        window_width: u32,
        window_height: u32,
        render_scale: f32,
    ) -> Self {
        let mut instance = PresentInstance::new(window);
        let descriptor_allocator =
            DescriptorAllocator::new(instance.device().logical().clone(), &AVG_DESC);
        let descriptor_cache = DescriptorSetLayoutCache::new(instance.device().logical().clone());
        let mut descriptor_creator =
            DescriptorSetCreator::new(descriptor_allocator, descriptor_cache);
        let mut mm = MemoryManager::new(
            instance.instance(),
            instance.device().logical(),
            instance.device().physical().device,
            FRAMES_IN_FLIGHT as u8,
        );
        let mut cmdm = CommandManager::new(
            instance.device().logical().clone(),
            instance.present_device().graphic_index(),
        );
        let swapchain = Swapchain::create(&instance, window_width, window_height);
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
            let buf_info = vk::DescriptorBufferInfo {
                buffer: buffer.buffer,
                offset: 0,
                range: std::mem::size_of::<FrameData>() as u64,
            };
            let descriptor = descriptor_creator
                .new_set()
                .bind_buffer(
                    buf_info,
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
        let sync = PresentSync::create(instance.device());
        let copy_sampler = create_copy_sampler(instance.device().logical());
        let clear_color = [0.15, 0.15, 0.15, 1.0];
        let mut forward_pass = RenderPass::forward(
            instance.device().logical(),
            copy_sampler,
            &mut mm,
            &mut descriptor_creator,
            render_size,
        );
        forward_pass.clear_color[0].color.float32 = clear_color;
        let imgui_renderer = ImguiDrawer::new(
            imgui,
            instance.device_mut(),
            &mut mm,
            &mut cmdm,
            &mut descriptor_creator,
            &swapchain,
        );
        let copy_pipeline = create_copy_pipeline(
            instance.device().logical(),
            swapchain.extent(),
            swapchain.renderpass(),
            &[forward_pass.copy_descriptor.layout],
        );
        RealtimeRenderer {
            instance,
            swapchain,
            descriptor_creator,
            copy_sampler,
            copy_pipeline,
            forward_pass,
            imgui_renderer,
            mm,
            cmdm,
            sync,
            scene: None,
            frame_data: frame_data.try_into().unwrap(),
            clear_color,
            start_time: Instant::now(),
            render_scale,
            frame_no: 0,
            paused: false,
            stats: InternalStats::default(),
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    pub fn pause(&mut self) {
        self.wait_idle();
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.wait_idle();
        self.paused = false;
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    pub fn get_clear_color(&self) -> [f32; 3] {
        [
            self.clear_color[0],
            self.clear_color[1],
            self.clear_color[2],
        ]
    }

    pub fn set_clear_color(&mut self, color: [f32; 3]) {
        self.clear_color = [color[0], color[1], color[2], 1.0];
        self.forward_pass.clear_color[0].color.float32 = self.clear_color;
    }

    pub fn stats(&self) -> Stats {
        self.stats.last_val
    }

    pub fn scene(&self) -> Option<&VulkanScene> {
        self.scene.as_ref()
    }

    pub fn scene_mut(&mut self) -> Option<&mut VulkanScene> {
        self.scene.as_mut()
    }

    pub fn camera_mut(&mut self) -> Option<&mut Camera> {
        if let Some(scene) = &mut self.scene {
            Some(&mut scene.current_cam)
        } else {
            None
        }
    }

    pub fn update_render_size(&mut self, window_width: u32, window_height: u32, scale: f32) {
        self.wait_idle();
        if let Some(scene) = &mut self.scene {
            scene.deinit_pipelines(self.instance.device().logical());
        }
        self.render_scale = scale;
        self.swapchain
            .recreate(&self.instance, window_width, window_height);
        self.imgui_renderer
            .update(self.instance.device().logical(), &self.swapchain);
        let render_size = vk::Extent2D {
            width: (window_width as f32 * scale) as u32,
            height: (window_height as f32 * scale) as u32,
        };
        let mut forward_pass = RenderPass::forward(
            self.instance.device().logical(),
            self.copy_sampler,
            &mut self.mm,
            &mut self.descriptor_creator,
            render_size,
        );
        forward_pass.clear_color[0].color.float32 = self.clear_color;
        std::mem::swap(&mut self.forward_pass, &mut forward_pass);
        forward_pass.destroy(self.instance.device().logical(), &mut self.mm);
        let mut copy_pipeline = create_copy_pipeline(
            self.instance.device().logical(),
            self.swapchain.extent(),
            self.swapchain.renderpass(),
            &[self.forward_pass.copy_descriptor.layout],
        );
        std::mem::swap(&mut self.copy_pipeline, &mut copy_pipeline);
        copy_pipeline.destroy(self.instance.device().logical());
        if let Some(scene) = &mut self.scene {
            scene.init_pipelines(
                render_size,
                self.instance.device().logical(),
                self.forward_pass.renderpass,
                self.frame_data[0].descriptor.layout,
            );
        }
    }

    pub fn change_scene(&mut self, parsed: Box<dyn ReadParsed>) {
        self.wait_idle();
        if let Some(mut scene) = self.scene.take() {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.unload(self.instance.device(), &mut self.mm);
        }
        if let Ok(mut new) = VulkanScene::load(
            self.instance.device_mut(),
            &mut self.mm,
            &mut self.cmdm,
            parsed,
            &mut self.descriptor_creator,
        ) {
            let render_size = vk::Extent2D {
                width: (self.swapchain.extent().width as f32 * self.render_scale) as u32,
                height: (self.swapchain.extent().height as f32 * self.render_scale) as u32,
            };
            new.init_pipelines(
                render_size,
                self.instance.device().logical(),
                self.forward_pass.renderpass,
                self.frame_data[0].descriptor.layout,
            );
            self.scene = Some(new);
            self.start_time = Instant::now();
        } else {
            log::error!("Failed to load scene");
        }
    }

    pub fn destroy(mut self) {
        self.wait_idle();
        self.imgui_renderer
            .destroy(self.instance.device().logical(), &mut self.mm);
        if let Some(mut scene) = self.scene.take() {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.unload(self.instance.device(), &mut self.mm);
        }
        for data in self.frame_data {
            self.mm.free_buffer(data.buffer);
        }
        unsafe {
            self.instance
                .device()
                .logical()
                .destroy_sampler(self.copy_sampler, None);
        };
        self.forward_pass
            .destroy(self.instance.device().logical(), &mut self.mm);
        self.copy_pipeline.destroy(self.instance.device().logical());
        self.descriptor_creator.destroy();
        self.swapchain.destroy(&self.instance);
        self.cmdm.destroy();
        self.mm.destroy();
        self.sync.destroy(self.instance.device());
        self.instance.destroy();
    }

    pub fn draw_frame(&mut self, imgui_data: Option<&imgui::DrawData>) {
        if self.paused {
            return;
        }
        self.stats.update();
        let frame_sync = self.sync.get(self.frame_no);
        frame_sync.wait_acquire(self.instance.device());
        if let Some(acquired) = self.swapchain.acquire_next_image(frame_sync) {
            let cmd = self.cmdm.get_cmd_buffer();
            let current_time = Instant::now();
            let frame_data = &mut self.frame_data[self.frame_no % FRAMES_IN_FLIGHT];
            frame_data.data.frame_time = (current_time - self.start_time).as_secs_f32();
            let device = self.instance.device().logical();
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
                self.forward_pass.begin(device, cmd);
                if let Some(scene) = &self.scene {
                    let ar = self.swapchain.extent().width as f32
                        / self.swapchain.extent().height as f32;
                    draw_objects(scene, ar, frame_data, device, cmd, &mut self.stats);
                }
                self.forward_pass.end(device, cmd);
                acquired.renderpass.begin(device, cmd);
                copy_renderpass_to_swapchain(
                    device,
                    cmd,
                    &self.copy_pipeline,
                    &[&self.forward_pass],
                    &mut self.stats,
                );
                // draw ui directly on the swapchain
                // tried doing it on its own attachment but results in blending problems
                if let Some(dd) = imgui_data {
                    if dd.total_vtx_count > 0 {
                        self.imgui_renderer.draw(
                            device,
                            cmd,
                            dd,
                            &mut self.mm,
                            &mut self.descriptor_creator,
                            self.scene.as_ref(),
                            &mut self.stats,
                        );
                    }
                }
                acquired.renderpass.end(device, cmd);
                device
                    .end_command_buffer(cmd)
                    .expect("Failed to end command buffer");
            }
            let wait_sem = [frame_sync.image_available()];
            let signal_sem = [frame_sync.render_finished()];
            let submit_ci = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: wait_sem.len() as u32,
                p_wait_semaphores: wait_sem.as_ptr(),
                p_wait_dst_stage_mask: &vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                signal_semaphore_count: signal_sem.len() as u32,
                p_signal_semaphores: signal_sem.as_ptr(),
            };
            let swapchains = [self.swapchain.swapchain_khr()];
            let present_ci = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                p_next: ptr::null(),
                wait_semaphore_count: signal_sem.len() as u32,
                p_wait_semaphores: signal_sem.as_ptr(),
                swapchain_count: swapchains.len() as u32,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: &acquired.index,
                p_results: ptr::null_mut(),
            };
            let queue = self.instance.device().graphic_queue();
            unsafe {
                device
                    .queue_submit(queue, &[submit_ci], frame_sync.acquire_fence())
                    .expect("Failed to submit render task");
            }
            self.swapchain.queue_present(queue, &present_ci);
            self.mm.frame_end_clean();
            self.frame_no += 1;
            self.stats.done_frame();
        } else {
            // out of date swapchain. the resize is called by winit so wait next frame
        }
    }
}

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
        .allocation
        .mapped_ptr()
        .expect("Failed to map buffer")
        .cast()
        .as_ptr();
    std::ptr::copy_nonoverlapping(&frame_data.data, buf_ptr, 1);
    //
    device.cmd_bind_vertex_buffers(cmd, 0, &[scene.vertex_buffer.buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, scene.index_buffer.buffer, 0, vk::IndexType::UINT32); //bind once, use firts_index as offset
    for obj in &scene.meshes {
        let (material, mat_descriptor) = scene.materials.get(&obj.material).unwrap(); //TODO: unwrap or default material
        if current_shader.is_none() || material.shader != current_shader.unwrap() {
            current_shader = Some(material.shader);
            let pipeline = scene.pipelines.get(&material.shader).unwrap(); //TODO: unwrap or load at runtime
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[frame_data.descriptor.set, mat_descriptor.set],
                &[],
            );
        }
        device.cmd_draw_indexed(cmd, obj.index_count, 1, obj.index_offset, 0, 0);
        stats.done_draw_call();
    }
}

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

fn create_copy_pipeline(
    device: &ash::Device,
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

fn copy_renderpass_to_swapchain(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    copy_pip: &Pipeline,
    rp: &[&RenderPass],
    stats: &mut InternalStats,
) {
    for renderpass in rp {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, copy_pip.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                copy_pip.layout,
                0,
                &[renderpass.copy_descriptor.set],
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            stats.done_draw_call();
        }
    }
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
