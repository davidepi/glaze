use super::descriptor::{
    Descriptor, DescriptorAllocator, DescriptorSetBuilder, DescriptorSetLayoutCache,
};
use super::device::Device;
use super::instance::{Instance, PresentInstance};
use super::memory::{AllocatedBuffer, MemoryManager};
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use super::sync::PresentSync;
use crate::Scene;
use ash::vk;
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
    descriptor_allocator: DescriptorAllocator,
    descriptor_cache: DescriptorSetLayoutCache,
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
        let mut descriptor_allocator =
            DescriptorAllocator::new(instance.device().logical().clone(), &AVG_DESC);
        let mut descriptor_cache =
            DescriptorSetLayoutCache::new(instance.device().logical().clone());
        let mut mm = MemoryManager::new(
            instance.instance(),
            instance.device().logical(),
            instance.device().physical().device,
        );
        let swapchain = Swapchain::create(&mut instance, &mut mm, width, height);
        let mut frame_data = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for frame in 0..FRAMES_IN_FLIGHT {
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
            let descriptor =
                DescriptorSetBuilder::new(&mut descriptor_cache, &mut descriptor_allocator)
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
        RealtimeRenderer {
            instance,
            swapchain,
            descriptor_allocator,
            descriptor_cache,
            mm,
            sync,
            scene: None,
            start_time: Instant::now(),
            frame_data: frame_data.try_into().unwrap(),
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
                self.frame_data[0].descriptor.layout,
            )
        }
    }

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
            &mut self.descriptor_allocator,
            &mut self.descriptor_cache,
        );
        new.init_pipelines(
            self.render_width,
            self.render_height,
            self.instance.device().logical(),
            self.swapchain.default_render_pass(),
            self.frame_data[0].descriptor.layout,
        );
        self.scene = Some(new);
        self.start_time = Instant::now();
    }

    pub fn destroy(mut self) {
        self.wait_idle();
        if let Some(mut scene) = self.scene.take() {
            scene.deinit_pipelines(self.instance.device().logical());
            scene.unload(self.instance.device(), &mut self.mm);
        }
        for data in self.frame_data {
            self.mm.free_buffer(data.buffer);
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
                    draw_objects(scene, ar, frame_data, device, acquired.cmd);
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
