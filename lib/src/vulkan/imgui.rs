use super::descriptor::{Descriptor, DescriptorSetCreator};
use super::device::Device;
use super::memory::{AllocatedBuffer, AllocatedImage, MemoryManager};
use super::renderpass::RenderPass;
use super::swapchain::Swapchain;
use crate::include_shader;
use crate::materials::{Pipeline, PipelineBuilder};
use ash::vk;
use cgmath::Vector2 as Vec2;
use gpu_allocator::MemoryLocation;
use std::ptr;

const DEFAULT_VERTEX_SIZE: u64 = 512 * std::mem::size_of::<imgui::DrawVert>() as u64;
const DEFAULT_INDEX_SIZE: u64 = 512 * std::mem::size_of::<imgui::DrawIdx>() as u64;

#[repr(C)]
#[derive(Clone, Copy)]
struct ImguiPC {
    scale: Vec2<f32>,
    translate: Vec2<f32>,
}

pub struct ImguiDrawer {
    vertex_size: u64,
    vertex_buf: AllocatedBuffer,
    index_size: u64,
    index_buf: AllocatedBuffer,
    font: AllocatedImage,
    pipeline: Pipeline,
    sampler: vk::Sampler,
    descriptor: Descriptor,
    extent: vk::Extent2D,
}

impl ImguiDrawer {
    pub fn new<T: Device>(
        context: &mut imgui::Context,
        device: &T,
        copy_sampler: vk::Sampler,
        mm: &mut MemoryManager,
        descriptor_creator: &mut DescriptorSetCreator,
        swapchain: &Swapchain,
    ) -> Self {
        let vertex_size = DEFAULT_VERTEX_SIZE;
        let index_size = DEFAULT_INDEX_SIZE;
        let mut builder = PipelineBuilder::default();
        builder.binding_descriptions = vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<imgui::DrawVert>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        builder.attribute_descriptions = vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];
        builder.no_depth();
        builder.rasterizer.cull_mode = vk::CullModeFlags::NONE;
        builder.push_shader(
            include_shader!("imgui.vert"),
            "main",
            vk::ShaderStageFlags::VERTEX,
        );
        builder.push_shader(
            include_shader!("imgui.frag"),
            "main",
            vk::ShaderStageFlags::FRAGMENT,
        );
        builder.push_constants(std::mem::size_of::<ImguiPC>(), vk::ShaderStageFlags::VERTEX);
        builder.blending_settings = vec![vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            color_write_mask: vk::ColorComponentFlags::all(),
            alpha_blend_op: vk::BlendOp::ADD,
            color_blend_op: vk::BlendOp::ADD,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        }];
        let fonts_gpu_buf;
        {
            let mut fonts_ref = context.fonts();
            let fonts = fonts_ref.build_rgba32_texture();
            let fonts_size = (fonts.width * fonts.height * 4) as u64;
            let fonts_extent = vk::Extent2D {
                width: fonts.width as u32,
                height: fonts.height as u32,
            };
            let fonts_cpu_buf = mm.create_buffer(
                "Font atlas CPU",
                fonts_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            );
            fonts_gpu_buf = mm.create_image_gpu(
                "Font atals GPU",
                vk::Format::R8G8B8A8_UNORM,
                fonts_extent,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            );
            let mapped = fonts_cpu_buf
                .allocation
                .mapped_ptr()
                .expect("Failed to map buffer")
                .cast()
                .as_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(fonts.data.as_ptr(), mapped, fonts_size as usize);
            }
            upload_image(device, &fonts_cpu_buf, &fonts_gpu_buf, fonts_extent);
            mm.free_buffer(fonts_cpu_buf);
        }
        let vertex_buf = mm.create_buffer(
            "Imgui vertex buffer",
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
        );
        let index_buf = mm.create_buffer(
            "Imgui index buffer",
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::CpuToGpu,
        );
        let sampler_ci = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
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
        let sampler = unsafe { device.logical().create_sampler(&sampler_ci, None) }
            .expect("Failed to create sampler");
        let texture_binding = vk::DescriptorImageInfo {
            sampler,
            image_view: fonts_gpu_buf.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let descriptor = descriptor_creator
            .new_set()
            .bind_image(
                texture_binding,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
            .build();

        let extent = swapchain.extent();
        let pipeline = builder.build(
            device.logical(),
            swapchain.renderpass(),
            extent,
            &[descriptor.layout],
        );
        Self {
            vertex_size,
            vertex_buf,
            index_size,
            index_buf,
            font: fonts_gpu_buf,
            pipeline,
            sampler,
            descriptor,
            extent,
        }
    }

    pub fn destroy(self, device: &ash::Device, mm: &mut MemoryManager) {
        self.pipeline.destroy(device);
        unsafe { device.destroy_sampler(self.sampler, None) };
        mm.free_buffer(self.vertex_buf);
        mm.free_buffer(self.index_buf);
        mm.free_image(self.font);
    }

    pub fn draw(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        mm: &mut MemoryManager,
    ) {
        // reallocate vertex buffer if not enough
        let vert_required_mem =
            (draw_data.total_vtx_count as usize * std::mem::size_of::<imgui::DrawVert>()) as u64;
        if vert_required_mem > self.vertex_size {
            let new_size = std::cmp::max(vert_required_mem, self.vertex_size);
            let mut new_vertex_buf = mm.create_buffer(
                "Imgui vertex buffer",
                new_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            );
            self.vertex_size = new_size;
            std::mem::swap(&mut self.vertex_buf, &mut new_vertex_buf);
            mm.free_buffer(new_vertex_buf);
        }
        // reallocate index buffer if not enough
        let idx_required_mem =
            (draw_data.total_idx_count as usize * std::mem::size_of::<imgui::DrawIdx>()) as u64;
        if idx_required_mem > self.index_size {
            let new_size = std::cmp::max(idx_required_mem, self.index_size);
            let mut new_index_buf = mm.create_buffer(
                "Imgui index buffer",
                new_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            );
            self.index_size = new_size;
            std::mem::swap(&mut self.index_buf, &mut new_index_buf);
            mm.free_buffer(new_index_buf);
        }
        // setup pipeline
        let scale = Vec2::new(
            2.0 / draw_data.display_size[0],
            2.0 / draw_data.display_size[1],
        );
        let translate = Vec2::new(
            -1.0 - draw_data.display_pos[0] * scale[0],
            -1.0 - draw_data.display_pos[1] * scale[1],
        );
        let imguipc = ImguiPC { scale, translate };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                as_u8_slice(&imguipc),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[self.descriptor.set],
                &[],
            );
        }
        let mut vert_offset = 0;
        let mut idx_offset = 0;
        for draw_list in draw_data.draw_lists() {
            unsafe {
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.vertex_buf.buffer],
                    &[vert_offset as u64 * std::mem::size_of::<imgui::DrawVert>() as u64],
                )
            };
            unsafe {
                device.cmd_bind_index_buffer(
                    cmd,
                    self.index_buf.buffer,
                    idx_offset as u64 * std::mem::size_of::<imgui::DrawIdx>() as u64,
                    vk::IndexType::UINT16,
                )
            };
            let vertices = draw_list.vtx_buffer();
            let dst_ptr = self
                .vertex_buf
                .allocation
                .mapped_ptr()
                .unwrap()
                .cast::<imgui::DrawVert>()
                .as_ptr();
            let dst_ptr = unsafe { dst_ptr.offset(vert_offset) };
            unsafe { std::ptr::copy_nonoverlapping(vertices.as_ptr(), dst_ptr, vertices.len()) };
            vert_offset += vertices.len() as isize;
            let indices = draw_list.idx_buffer();
            let dst_ptr = self
                .index_buf
                .allocation
                .mapped_ptr()
                .unwrap()
                .cast::<imgui::DrawIdx>()
                .as_ptr();
            let dst_ptr = unsafe { dst_ptr.offset(idx_offset) };
            unsafe { std::ptr::copy_nonoverlapping(indices.as_ptr(), dst_ptr, indices.len()) };
            idx_offset += indices.len() as isize;
            // then draw everything in that list
            for command in draw_list.commands() {
                match command {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        unsafe {
                            device.cmd_draw_indexed(
                                cmd,
                                count as u32,
                                1,
                                cmd_params.idx_offset as u32,
                                cmd_params.vtx_offset as i32,
                                0,
                            )
                        };
                    }
                    _ => todo!(),
                }
            }
        }
    }
}

fn upload_image<T: Device>(
    device: &T,
    cpu_buf: &AllocatedBuffer,
    gpu_buf: &AllocatedImage,
    extent: vk::Extent2D,
) {
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    };
    let barrier_transfer = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: gpu_buf.image,
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
        image: gpu_buf.image,
        subresource_range,
    };
    let image_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let copy_region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,   // 0 = same as image width
        buffer_image_height: 0, // 0 = same as image height
        image_subresource,
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
    };
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_transfer],
            );
            device.cmd_copy_buffer_to_image(
                cmd,
                cpu_buf.buffer,
                gpu_buf.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_use],
            );
        }
    };
    device.immediate_execute(command);
}

unsafe fn as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}
