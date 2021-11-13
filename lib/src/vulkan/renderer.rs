use super::descriptor::{DescriptorAllocator, DescriptorSetLayoutCache};
use super::device::Device;
use super::instance::{Instance, PresentInstance};
use super::memory::MemoryManager;
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use super::sync::PresentSync;
use crate::materials::Pipeline;
use crate::Scene;
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use std::ptr;
use winit::window::Window;

const AVG_DESC: [(vk::DescriptorType, f32); 1] = [(vk::DescriptorType::UNIFORM_BUFFER, 1.0)];
const FRAMES_IN_FLIGHT: usize = 2;

pub struct RealtimeRenderer {
    instance: PresentInstance,
    swapchain: Swapchain,
    descriptor_allocator: DescriptorAllocator,
    descriptor_cache: DescriptorSetLayoutCache,
    mm: MemoryManager,
    sync: PresentSync<FRAMES_IN_FLIGHT>,
    scene: Option<VulkanScene<FRAMES_IN_FLIGHT>>,
    render_width: u32,
    render_height: u32,
    frame_no: usize,
}

impl RealtimeRenderer {
    pub fn create(window: &Window, width: u32, height: u32) -> Self {
        let mut instance = PresentInstance::new(&window);
        let descriptor_allocator =
            DescriptorAllocator::new(instance.device().logical().clone(), &AVG_DESC);
        let descriptor_cache = DescriptorSetLayoutCache::new(instance.device().logical().clone());
        let mut mm = MemoryManager::new(
            instance.instance(),
            instance.device().logical(),
            instance.device().physical().device,
        );
        let swapchain = Swapchain::create(&mut instance, &mut mm, width, height);
        let sync = PresentSync::create(instance.device());
        RealtimeRenderer {
            instance,
            swapchain,
            descriptor_allocator,
            descriptor_cache,
            mm,
            sync,
            scene: None,
            render_width: width,
            render_height: height,
            frame_no: 0,
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    pub fn change_render_size(&mut self, width: u32, height: u32) {
        self.wait_idle();
        self.render_width = width;
        self.render_height = height;
        let mut new_swapchain = Swapchain::create(&mut self.instance, &mut self.mm, width, height);
        std::mem::swap(&mut self.swapchain, &mut new_swapchain);
        new_swapchain.destroy(&self.instance, &mut self.mm); //new swapchain is the old one!
        if let Some(scene) = &mut self.scene {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.init_pipelines(
                width,
                height,
                self.instance.device().logical(),
                self.swapchain.default_render_pass(),
            )
        }
    }

    pub fn change_scene(&mut self, scene: Scene) {
        self.wait_idle();
        if let Some(mut scene) = self.scene.take() {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.unload(&mut self.mm);
        }
        let mut new =
            VulkanScene::<FRAMES_IN_FLIGHT>::load(self.instance.device(), &mut self.mm, scene);
        new.init_pipelines(
            self.render_width,
            self.render_height,
            self.instance.device().logical(),
            self.swapchain.default_render_pass(),
        );
        self.scene = Some(new);
    }

    pub fn destroy(mut self) {
        self.wait_idle();
        if let Some(mut scene) = self.scene.take() {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.unload(&mut self.mm);
        }
        self.descriptor_cache.destroy();
        self.descriptor_allocator.destroy();
        self.swapchain.destroy(&self.instance, &mut self.mm);
        self.mm.destroy();
        self.sync.destroy(self.instance.device());
        self.instance.destroy();
    }

    pub fn draw_frame(&mut self) {
        let frame_sync = self.sync.get(self.frame_no);
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
            let color = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
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
                if let Some(scene) = &self.scene {
                    let ar = self.render_width as f32 / self.render_height as f32;
                    draw_objects(scene, ar, device, acquired.cmd);
                }
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
            self.frame_no += 1;
        } else {
            return; // out of date swapchain. the resize is called by winit so wait next frame
        }
    }
}

unsafe fn draw_objects(
    scene: &VulkanScene<FRAMES_IN_FLIGHT>,
    ar: f32,
    device: &ash::Device,
    cmd: vk::CommandBuffer,
) {
    let cam = &scene.current_cam;
    let mut proj = cam.projection(ar);
    proj[1][1] *= -1.0;
    let view = cam.look_at_rh();
    let viewproj = proj * view;
    let mut current_shader_id = u8::MAX;
    device.cmd_bind_vertex_buffers(cmd, 0, &[scene.vertex_buffer.buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, scene.index_buffer.buffer, 0, vk::IndexType::UINT32); //bind once, use firts_index as offset
    for obj in &scene.meshes {
        let material = scene.materials.get(&obj.material).unwrap(); //TODO: unwrap or default material
        if material.shader_id != current_shader_id {
            current_shader_id = material.shader_id;
            let pipeline = scene.pipelines.get(&material.shader_id).unwrap(); //TODO: unwrap or load at runtime
            device.cmd_push_constants(
                cmd,
                pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                as_u8_slice(&viewproj),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
        }
        device.cmd_draw_indexed(cmd, obj.index_count, 1, obj.index_offset, 0, 0);
    }
}

unsafe fn as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}
