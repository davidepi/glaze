use super::descriptor::{
    Descriptor, DescriptorAllocator, DescriptorSetBuilder, DescriptorSetCreator,
    DescriptorSetLayoutCache,
};
use super::device::{Device, PresentDevice};
use super::imgui::ImguiDrawer;
use super::instance::{Instance, PresentInstance};
use super::memory::{AllocatedBuffer, AllocatedImage, MemoryManager};
use super::renderpass::{FinalRenderPass, RenderPass};
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use super::sync::PresentSync;
use crate::materials::{Pipeline, PipelineBuilder};
use crate::{include_shader, Scene};
use ash::vk::{self, DescriptorSetLayout};
use cgmath::{Matrix4, SquareMatrix};
use std::ptr;
use std::time::Instant;
use winit::window::Window;

const AVG_DESC: [(vk::DescriptorType, f32); 1] = [(vk::DescriptorType::UNIFORM_BUFFER, 1.0)];
const FRAMES_IN_FLIGHT: usize = 2;

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
    imgui: imgui::Context,
    imgui_renderer: ImguiDrawer,
    mm: MemoryManager,
    sync: PresentSync<FRAMES_IN_FLIGHT>,
    scene: Option<VulkanScene>,
    frame_data: [FrameDataBuf; FRAMES_IN_FLIGHT],
    start_time: Instant,
    render_width: u32,
    render_height: u32,
    frame_no: usize,
}

impl RealtimeRenderer {
    pub fn create(window: &Window, width: u32, height: u32) -> Self {
        let mut instance = PresentInstance::new(&window);
        let extent = vk::Extent2D { width, height };
        let descriptor_allocator =
            DescriptorAllocator::new(instance.device().logical().clone(), &AVG_DESC);
        let descriptor_cache = DescriptorSetLayoutCache::new(instance.device().logical().clone());
        let mut descriptor_creator =
            DescriptorSetCreator::new(descriptor_allocator, descriptor_cache);
        let mut mm = MemoryManager::new(
            instance.instance(),
            instance.device().logical(),
            instance.device().physical().device,
        );
        let swapchain = Swapchain::create(&mut instance, &mut mm, width, height);
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
        let forward_pass = RenderPass::forward(
            instance.device().logical(),
            copy_sampler,
            &mut mm,
            &mut descriptor_creator,
            extent,
        );
        let copy_pipeline = create_copy_pipeline(
            instance.device().logical(),
            swapchain.extent(),
            swapchain.renderpass(),
            &[forward_pass.copy_descriptor.layout],
        );
        let mut imgui = imgui::Context::create();
        let imgui_renderer = ImguiDrawer::new(
            &mut imgui,
            instance.device(),
            copy_sampler,
            &mut mm,
            &mut descriptor_creator,
            swapchain.extent(),
        );
        RealtimeRenderer {
            instance,
            swapchain,
            descriptor_creator,
            copy_sampler,
            copy_pipeline,
            forward_pass,
            imgui,
            imgui_renderer,
            mm,
            sync,
            scene: None,
            frame_data: frame_data.try_into().unwrap(),
            start_time: Instant::now(),
            render_width: width,
            render_height: height,
            frame_no: 0,
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    //TODO: readd change_render_size, when there is a good distinction between render size
    //      and window size

    pub fn change_scene(&mut self, scene: Scene) {
        self.wait_idle();
        if let Some(mut scene) = self.scene.take() {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.unload(self.instance.device(), &mut self.mm);
        }
        let mut new = VulkanScene::load(
            self.instance.device(),
            &mut self.mm,
            scene,
            &mut self.descriptor_creator,
        );
        new.init_pipelines(
            self.render_width,
            self.render_height,
            self.instance.device().logical(),
            self.forward_pass.renderpass,
            self.frame_data[0].descriptor.layout,
        );
        self.scene = Some(new);
        self.start_time = Instant::now();
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
            .destroy(&self.instance.device().logical(), &mut self.mm);
        self.copy_pipeline.destroy(self.instance.device().logical());
        self.descriptor_creator.destroy();
        self.swapchain.destroy(&self.instance, &mut self.mm);
        self.mm.destroy();
        self.sync.destroy(self.instance.device());
        self.instance.destroy();
    }

    pub fn draw_frame(&mut self) {
        let frame_sync = self.sync.get(self.frame_no);
        frame_sync.wait_acquire(self.instance.device());
        if let Some(acquired) = self.swapchain.acquire_next_image(frame_sync) {
            let current_time = Instant::now();
            let frame_data = &mut self.frame_data[self.frame_no % FRAMES_IN_FLIGHT];
            frame_data.data.frame_time = (current_time - self.start_time).as_secs_f32();
            let device = self.instance.device().logical();
            unsafe {
                device.reset_command_buffer(acquired.cmd, vk::CommandBufferResetFlags::empty())
            }
            .expect("Failed to reset command buffer");
            let cmd_ci = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
            };
            unsafe {
                device
                    .begin_command_buffer(acquired.cmd, &cmd_ci)
                    .expect("Failed to begin command buffer");
                self.forward_pass.begin(device, acquired.cmd);
                if let Some(scene) = &self.scene {
                    let ar = self.render_width as f32 / self.render_height as f32;
                    draw_objects(scene, ar, frame_data, device, acquired.cmd);
                }
                self.forward_pass.end(device, acquired.cmd);
                self.imgui_renderer.draw(
                    device,
                    acquired.cmd,
                    self.imgui.frame().render(),
                    &mut self.mm,
                );
                copy_renderpass_results(
                    device,
                    acquired.cmd,
                    &self.copy_pipeline,
                    &[&self.forward_pass, self.imgui_renderer.ui_pass()],
                    acquired.renderpass,
                );
                device
                    .end_command_buffer(acquired.cmd)
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
                p_command_buffers: &acquired.cmd,
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
            self.frame_no += 1;
        } else {
            return; // out of date swapchain. the resize is called by winit so wait next frame
        }
    }
}

unsafe fn draw_objects(
    scene: &VulkanScene,
    ar: f32,
    frame_data: &mut FrameDataBuf,
    device: &ash::Device,
    cmd: vk::CommandBuffer,
) {
    let cam = &scene.current_cam;
    let mut proj = cam.projection(ar);
    proj[1][1] *= -1.0;
    let view = cam.look_at_rh();
    frame_data.data.projview = proj * view;
    let mut current_shader_id = u8::MAX;
    //write frame_data to the buffer
    let buf_ptr = frame_data
        .buffer
        .allocation
        .mapped_ptr()
        .expect("Failed to map buffer")
        .cast()
        .as_ptr();
    std::ptr::copy_nonoverlapping(&frame_data.data, buf_ptr, std::mem::size_of::<FrameData>());
    //
    device.cmd_bind_vertex_buffers(cmd, 0, &[scene.vertex_buffer.buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, scene.index_buffer.buffer, 0, vk::IndexType::UINT32); //bind once, use firts_index as offset
    for obj in &scene.meshes {
        let (material, mat_descriptor) = scene.materials.get(&obj.material).unwrap(); //TODO: unwrap or default material
        if material.shader_id != current_shader_id {
            current_shader_id = material.shader_id;
            let pipeline = scene.pipelines.get(&material.shader_id).unwrap(); //TODO: unwrap or load at runtime
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
    dset_layout: &[DescriptorSetLayout],
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

fn copy_renderpass_results(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    copy_pip: &Pipeline,
    rp: &[&RenderPass],
    finalpass: &FinalRenderPass,
) {
    finalpass.begin(device, cmd);
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
        }
    }
    finalpass.end(device, cmd);
}
