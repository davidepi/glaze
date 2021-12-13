use super::cmd::CommandManager;
use super::descriptor::{Descriptor, DescriptorSetManager};
use super::device::Device;
use super::memory::{AllocatedBuffer, AllocatedImage, MemoryManager};
use super::pipeline::{Pipeline, PipelineBuilder};
use super::renderer::InternalStats;
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use crate::{include_shader, TextureFormat};
use ash::vk;
use cgmath::Vector2 as Vec2;
use fnv::FnvHashMap;
use gpu_allocator::MemoryLocation;
use imgui::{DrawCmdParams, TextureId};
use std::ptr;

/// initial buffer size for the imgui vertex buffer
const INITIAL_VERTEX_SIZE: u64 = 512 * std::mem::size_of::<imgui::DrawVert>() as u64;
/// initial buffer size for the imgui index buffer
const INITIAL_INDEX_SIZE: u64 = 512 * std::mem::size_of::<imgui::DrawIdx>() as u64;
/// ID used to reference the imgui font atlas. This should not be used by any other texture.
const FONT_ATLAS_TEXTURE_ID: TextureId = TextureId::new(usize::MAX);

#[repr(C)]
#[derive(Clone, Copy)]
/// Push constants for the imgui pipeline
struct ImguiPC {
    scale: Vec2<f32>,
    translate: Vec2<f32>,
}

/// Imgui backend for vulkan
pub struct ImguiRenderer {
    /// size of the allocated vertex buffer (in bytes)
    vertex_size: u64,
    /// vertex buffer
    vertex_buf: AllocatedBuffer,
    /// size of the allocated index buffer (in bytes)
    index_size: u64,
    /// index buffer
    index_buf: AllocatedBuffer,
    /// font atlas image
    font: AllocatedImage,
    font_descriptor: Descriptor,
    pipeline: Pipeline,
    // pipeline for single_channel images (sampler takes .rrr instead of .rgb)
    pipeline_bw: Pipeline,
    sampler: vk::Sampler,
    tex_descs: FnvHashMap<u16, Descriptor>,
}

impl ImguiRenderer {
    /// Creates a new imgui renderer
    pub(super) fn new<T: Device>(
        context: &mut imgui::Context,
        device: &T,
        mm: &mut MemoryManager,
        cmdm: &mut CommandManager,
        descriptor_creator: &mut DescriptorSetManager,
        swapchain: &Swapchain,
    ) -> Self {
        let vertex_size = INITIAL_VERTEX_SIZE;
        let index_size = INITIAL_INDEX_SIZE;
        let fonts_gpu_buf;
        {
            let mut fonts_ref = context.fonts();
            // Outside the range of engine-assignable texture ids. So if it is not in the map,
            // it is definetly the texture atlas.
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
                1,
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
            upload_image(device, cmdm, &fonts_cpu_buf, &fonts_gpu_buf, fonts_extent);
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
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
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
        let font_descriptor = descriptor_creator
            .new_set()
            .bind_image(
                texture_binding,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
            .build();
        // assign tex_id AFTER building the font atlas or it gets reset
        let mut fonts = context.fonts();
        fonts.tex_id = FONT_ATLAS_TEXTURE_ID;
        let pipeline = build_imgui_pipeline(device.logical(), swapchain, &font_descriptor, false);
        let pipeline_bw = build_imgui_pipeline(device.logical(), swapchain, &font_descriptor, true);
        Self {
            vertex_size,
            vertex_buf,
            index_size,
            index_buf,
            font: fonts_gpu_buf,
            pipeline,
            pipeline_bw,
            sampler,
            font_descriptor,
            tex_descs: FnvHashMap::default(),
        }
    }

    /// Updates the underlying swapchain (and therefore the imgui render size)
    pub(super) fn update_swapchain(&mut self, device: &ash::Device, swapchain: &Swapchain) {
        let mut pipeline = build_imgui_pipeline(device, swapchain, &self.font_descriptor, false);
        let mut pipeline_bw = build_imgui_pipeline(device, swapchain, &self.font_descriptor, true);
        std::mem::swap(&mut self.pipeline, &mut pipeline);
        std::mem::swap(&mut self.pipeline_bw, &mut pipeline_bw);
        pipeline.destroy(device);
        pipeline_bw.destroy(device);
    }

    /// destroy the imgui renderer
    pub(super) fn destroy(self, device: &ash::Device, mm: &mut MemoryManager) {
        self.pipeline.destroy(device);
        self.pipeline_bw.destroy(device);
        unsafe { device.destroy_sampler(self.sampler, None) };
        mm.free_buffer(self.vertex_buf);
        mm.free_buffer(self.index_buf);
        mm.free_image(self.font);
    }

    /// draw the ui. This should be called inside an existing render pass.
    pub(super) fn draw(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        mm: &mut MemoryManager,
        dm: &mut DescriptorSetManager,
        scene: Option<&VulkanScene>,
        stats: &mut InternalStats,
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
            mm.deferred_free_buffer(new_vertex_buf);
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
            mm.deferred_free_buffer(new_index_buf);
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
                &[self.font_descriptor.set],
                &[],
            );
        }
        let mut bound_texture = FONT_ATLAS_TEXTURE_ID;
        let mut pipeline_bw = false;
        let mut vert_offset = 0;
        let mut idx_offset = 0;
        let clip_offset = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
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
                    imgui::DrawCmd::Elements {
                        count,
                        cmd_params:
                            DrawCmdParams {
                                clip_rect,
                                texture_id,
                                idx_offset,
                                vtx_offset,
                            },
                    } => {
                        let clip_x = (clip_rect[0] - clip_offset[0]) * clip_scale[0];
                        let clip_y = (clip_rect[1] - clip_offset[1]) * clip_scale[1];
                        let clip_w = (clip_rect[2] - clip_offset[0]) * clip_scale[0] - clip_x;
                        let clip_h = (clip_rect[3] - clip_offset[1]) * clip_scale[1] - clip_y;
                        let scissors = [vk::Rect2D {
                            offset: vk::Offset2D {
                                x: clip_x as _,
                                y: clip_y as _,
                            },
                            extent: vk::Extent2D {
                                width: clip_w as _,
                                height: clip_h as _,
                            },
                        }];
                        // change pipeline if needed + create descriptor for texture if needed
                        if texture_id != bound_texture {
                            if texture_id == FONT_ATLAS_TEXTURE_ID {
                                if pipeline_bw {
                                    // set default pipeline
                                    pipeline_bw = false;
                                    unsafe {
                                        device.cmd_bind_pipeline(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.pipeline.pipeline,
                                        );
                                        device.cmd_push_constants(
                                            cmd,
                                            self.pipeline.layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            as_u8_slice(&imguipc),
                                        );
                                    }
                                }
                                unsafe {
                                    device.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        self.pipeline.layout,
                                        0,
                                        &[self.font_descriptor.set],
                                        &[],
                                    );
                                }
                            } else if let Some(scene) = scene {
                                // texture_id != FONT_ATLAS_TEXTURE_ID implicitly assumed
                                let tid_u16 = texture_id.id() as u16;
                                // this should never fail
                                let texture = scene.textures.get(&tid_u16).unwrap();
                                let descriptor =
                                    self.tex_descs.entry(tid_u16).or_insert_with(|| {
                                        let info = vk::DescriptorImageInfo {
                                            sampler: self.sampler,
                                            image_view: texture.image.image_view,
                                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                        };
                                        dm.new_set()
                                            .bind_image(
                                                info,
                                                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                                vk::ShaderStageFlags::FRAGMENT,
                                            )
                                            .build()
                                    });
                                if texture.info.format == TextureFormat::Rgba && pipeline_bw {
                                    // set default pipeline
                                    pipeline_bw = false;
                                    unsafe {
                                        device.cmd_bind_pipeline(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.pipeline.pipeline,
                                        );
                                        device.cmd_push_constants(
                                            cmd,
                                            self.pipeline.layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            as_u8_slice(&imguipc),
                                        );
                                    }
                                } else if texture.info.format == TextureFormat::Gray && !pipeline_bw
                                {
                                    // set bw pipeline
                                    pipeline_bw = true;
                                    unsafe {
                                        device.cmd_bind_pipeline(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.pipeline_bw.pipeline,
                                        );
                                        device.cmd_push_constants(
                                            cmd,
                                            self.pipeline_bw.layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            as_u8_slice(&imguipc),
                                        );
                                    }
                                }
                                unsafe {
                                    device.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        self.pipeline.layout,
                                        0,
                                        &[descriptor.set],
                                        &[],
                                    );
                                }
                            } else {
                                panic!("Requestest to bound non-font texture but no scene loaded");
                            }
                            bound_texture = texture_id;
                        }
                        unsafe {
                            device.cmd_set_scissor(cmd, 0, &scissors);
                            device.cmd_draw_indexed(
                                cmd,
                                count as u32,
                                1,
                                idx_offset as u32,
                                vtx_offset as i32,
                                0,
                            );
                            stats.done_draw_call();
                        };
                    }
                    _ => todo!(),
                }
            }
        }
    }
}

/// Builds the imgui pipeline. font_descriptor is the descriptor set containing the font texture.
/// bw indicates that the pipeline uses a grayscale texture (uses .rrr instead of .rgb to avoid
/// having red tint on single channel textures).
fn build_imgui_pipeline(
    device: &ash::Device,
    swapchain: &Swapchain,
    font_descriptor: &Descriptor,
    bw: bool,
) -> Pipeline {
    let mut builder = PipelineBuilder {
        binding_descriptions: vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<imgui::DrawVert>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }],
        attribute_descriptions: vec![
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
        ],
        ..Default::default()
    };
    builder.no_depth();
    builder.rasterizer.cull_mode = vk::CullModeFlags::NONE;
    builder.push_shader(
        include_shader!("imgui.vert"),
        "main",
        vk::ShaderStageFlags::VERTEX,
    );
    if bw {
        builder.push_shader(
            include_shader!("imgui_bw.frag"),
            "main",
            vk::ShaderStageFlags::FRAGMENT,
        );
    } else {
        builder.push_shader(
            include_shader!("imgui.frag"),
            "main",
            vk::ShaderStageFlags::FRAGMENT,
        );
    };
    builder.dynamic_states = vec![vk::DynamicState::SCISSOR];
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
    builder.build(
        device,
        swapchain.renderpass(),
        swapchain.extent(),
        &[font_descriptor.layout],
    )
}

/// Uploads an image in the cpu_buf to the gpu_buf. The image is transitioned to the optimal layout
fn upload_image<T: Device>(
    device: &T,
    cmdm: &mut CommandManager,
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
    let cmd = cmdm.get_cmd_buffer();
    let fence = device.immediate_execute(cmd, command);
    device.wait_completion(&[fence]);
}

/// Reads a struct as a sequence of bytes
unsafe fn as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}
