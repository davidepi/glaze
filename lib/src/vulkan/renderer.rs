use std::ptr;

use super::{AllocatedBuffer, Device, Instance, PresentInstance, PresentSync, Swapchain};
use crate::Scene;
use ash::vk;
use winit::window::Window;

pub struct RealtimeRenderer {
    pub instance: PresentInstance,
    pub swapchain: Swapchain,
    pub sync: PresentSync,
    vertex_buffer: Option<AllocatedBuffer>,
    render_width: u32,
    render_height: u32,
}

impl RealtimeRenderer {
    pub fn create(window: &Window, width: u32, height: u32) -> Self {
        let mut instance = PresentInstance::new(&window);
        let swapchain = Swapchain::create(&mut instance, width, height);
        let sync = PresentSync::create(instance.device());
        RealtimeRenderer {
            instance,
            swapchain,
            sync,
            vertex_buffer: None,
            render_width: width,
            render_height: height,
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    pub fn change_render_size(&mut self, width: u32, height: u32) {
        self.wait_idle();
        self.render_width = width;
        self.render_height = height;
        self.swapchain.re_create(&mut self.instance, width, height);
    }

    pub fn change_scene(&mut self, scene: &Scene) {
        let device = self.instance.device_mut();
        if let Some(prev_buf) = self.vertex_buffer.take() {
            device.free_vertices(prev_buf);
        }
        self.vertex_buffer = Some(device.load_vertices(&scene.vertices));
    }

    pub fn destroy(mut self) {
        let device = self.instance.device_mut();
        if let Some(vb) = self.vertex_buffer.take() {
            device.free_vertices(vb);
        }
        self.sync.destroy(self.instance.device());
        self.swapchain.destroy(&self.instance);
    }

    pub fn draw_frame(&mut self) {
        let frame_sync = self.sync.next();
        frame_sync.wait_acquire(self.instance.device());
        if let Some(acquired) = self.swapchain.acquire_next_image(frame_sync) {
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
            let color = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0],
                },
            }];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent(),
            };
            let rp_ci = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass: self.swapchain.default_render_pass(),
                framebuffer: acquired.framebuffer,
                render_area,
                clear_value_count: color.len() as u32,
                p_clear_values: color.as_ptr(),
            };
            unsafe {
                device
                    .begin_command_buffer(acquired.cmd, &cmd_ci)
                    .expect("Failed to begin command buffer");
                device.cmd_begin_render_pass(acquired.cmd, &rp_ci, vk::SubpassContents::INLINE);
                device.cmd_end_render_pass(acquired.cmd);
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
        } else {
            return; // out of date swapchain. the resize is called by winit so wait next frame
        }
    }
}
