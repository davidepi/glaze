use std::ptr;

use crate::vulkan::{Device, Instance, RealtimeRenderer};
use ash::vk;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

pub const DEFAULT_WIDTH: u32 = 800;
pub const DEFAULT_HEIGHT: u32 = 600;

pub struct GlazeApp {
    window: Window,
    renderer: RealtimeRenderer,
}

impl GlazeApp {
    pub fn create(event_loop: &EventLoop<()>) -> GlazeApp {
        let render_width = DEFAULT_WIDTH;
        let render_height = DEFAULT_HEIGHT;
        let window = WindowBuilder::new()
            .with_title(env!("CARGO_PKG_NAME"))
            .with_inner_size(winit::dpi::LogicalSize::new(render_width, render_height))
            .with_resizable(false)
            .build(event_loop)
            .unwrap();
        let renderer = RealtimeRenderer::create(&window, render_width, render_height);
        GlazeApp { window, renderer }
    }

    pub fn draw_frame(&mut self) {
        let r = &mut self.renderer;
        let frame_sync = r.sync.next();
        frame_sync.wait_acquire(r.instance.device());
        if let Some(acquired) = r.swapchain.acquire_next_image(frame_sync) {
            let device = r.instance.device().logical();
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
                extent: r.swapchain.extent(),
            };
            let rp_ci = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass: r.swapchain.default_render_pass(),
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
            let swapchains = [r.swapchain.swapchain_khr()];
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
            let queue = r.instance.device().graphic_queue();
            unsafe {
                device
                    .queue_submit(queue, &[submit_ci], frame_sync.acquire_fence())
                    .expect("Failed to submit render task");
            }
            r.swapchain.queue_present(queue, &present_ci);
        } else {
            return; // out of date swapchain. the resize is called by winit so wait next frame
        }
    }

    pub fn main_loop(&mut self, mut event_loop: EventLoop<()>) {
        event_loop.run_return(|event, _, control_flow| match event {
            Event::WindowEvent {
                event,
                window_id: _,
            } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::MainEventsCleared => self.draw_frame(),
            Event::LoopDestroyed => self.renderer.wait_idle(),
            _ => (),
        });
    }

    pub fn destroy(self) {
        self.renderer.destroy();
    }
}
