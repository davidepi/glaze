use super::cmd::CommandManager;
use super::descriptor::{DLayoutCache, Descriptor, DescriptorSetManager};
use super::device::Device;
use super::instance::Instance;
use super::memory::{AllocatedBuffer, AllocatedImage};
use super::pipeline::{Pipeline, PipelineBuilder};
use super::renderer::InternalStats;
use super::scene::VulkanScene;
use super::swapchain::Swapchain;
use crate::materials::TextureLoaded;
use crate::{include_shader, PresentInstance, TextureFormat};
use ash::vk;
use cgmath::Vector2 as Vec2;
use font_kit::family_name::FamilyName;
use font_kit::properties::Properties;
use font_kit::source::SystemSource;
use gpu_allocator::MemoryLocation;
use imgui::{DrawCmdParams, FontSource, TextureId};
use std::ptr;
use std::sync::{Arc, RwLock};

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
    font: (AllocatedImage, Descriptor),
    /// pipeline for colored textures
    pipeline: Pipeline,
    /// pipeline for single_channel images (sampler takes .rrr instead of .rgb)
    pipeline_bw: Pipeline,
    /// sampler for the UI textures
    sampler: vk::Sampler,
    /// descriptor manager reserved for the UI resources
    dm: DescriptorSetManager,
    /// Reference to the scene textures, to avoid inconsistent states when the scene is
    /// deallocated.
    scene_textures: Option<Arc<RwLock<Vec<TextureLoaded>>>>,
    /// descriptors of the various textures, along with the texture info
    tex_descs: Vec<Descriptor>,
    /// Buffers that cannot be freed immediately, as they require the GPU to finish first.
    /// The first value is the number of invokation of the "draw" method before they are dropped.
    free_later: Vec<(u8, AllocatedBuffer)>,
    /// vulkan device handle
    instance: Arc<PresentInstance>,
}

impl ImguiRenderer {
    /// Creates a new imgui renderer
    pub(super) fn new(
        context: &mut imgui::Context,
        instance: Arc<PresentInstance>,
        layout_cache: DLayoutCache,
        swapchain: &Swapchain,
    ) -> Self {
        let device = instance.device();
        let mm = instance.allocator();
        let vertex_size = INITIAL_VERTEX_SIZE;
        let index_size = INITIAL_INDEX_SIZE;
        let avg_sizes = [(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1.0)];
        let mut dm = DescriptorSetManager::new(device.logical_clone(), &avg_sizes, layout_cache);
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 1);
        let fonts_gpu_buf;
        {
            let mut fonts_ref = context.fonts();
            // use a system font if possible (otherwise the default one is okay I guess)
            if let Ok(handle) =
                SystemSource::new().select_best_match(&[FamilyName::SansSerif], &Properties::new())
            {
                if let Ok(loaded) = handle.load() {
                    if let Some(raw) = loaded.copy_font_data() {
                        fonts_ref.clear_fonts();
                        fonts_ref.add_font(&[FontSource::TtfData {
                            data: &raw,
                            size_pixels: 13.0,
                            config: None,
                        }]);
                    }
                }
            }
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
                .allocation()
                .mapped_ptr()
                .expect("Failed to map buffer")
                .cast()
                .as_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(fonts.data.as_ptr(), mapped, fonts_size as usize);
            }
            upload_image(
                device,
                &mut tcmdm,
                &fonts_cpu_buf,
                &fonts_gpu_buf,
                fonts_extent,
            );
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
        let font_descriptor = dm
            .new_set()
            .bind_image(
                &fonts_gpu_buf,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
            .build();
        // assign tex_id AFTER building the font atlas or it gets reset
        let mut fonts = context.fonts();
        fonts.tex_id = FONT_ATLAS_TEXTURE_ID;
        let pipeline =
            build_imgui_pipeline(device.logical_clone(), swapchain, &font_descriptor, false);
        let pipeline_bw =
            build_imgui_pipeline(device.logical_clone(), swapchain, &font_descriptor, true);
        Self {
            vertex_size,
            vertex_buf,
            index_size,
            index_buf,
            font: (fonts_gpu_buf, font_descriptor),
            pipeline,
            pipeline_bw,
            sampler,
            dm,
            scene_textures: None,
            tex_descs: Vec::new(),
            free_later: Vec::new(),
            instance,
        }
    }

    /// Updates the underlying swapchain (and therefore the imgui render size)
    pub(super) fn update_swapchain(&mut self, swapchain: &Swapchain) {
        self.pipeline = build_imgui_pipeline(
            self.instance.device().logical_clone(),
            swapchain,
            &self.font.1,
            false,
        );
        self.pipeline_bw = build_imgui_pipeline(
            self.instance.device().logical_clone(),
            swapchain,
            &self.font.1,
            true,
        );
    }

    pub(super) fn load_scene_textures(&mut self, scene: &VulkanScene) {
        let scene_textures = Arc::clone(&scene.textures);
        self.scene_textures = Some(scene_textures);
        self.rebuild_texture_descriptors();
    }

    pub fn rebuild_texture_descriptors(&mut self) {
        self.tex_descs = if let Some(scene_textures) = &self.scene_textures {
            scene_textures
                .read()
                .unwrap()
                .iter()
                .map(|texture| {
                    self.dm
                        .new_set()
                        .bind_image(
                            &texture.image,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            self.sampler,
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            vk::ShaderStageFlags::FRAGMENT,
                        )
                        .build()
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// draw the ui. This should be called inside an existing render pass.
    /// cmd is a command buffer for a graphic queue.
    pub(super) fn draw(
        &mut self,
        cmd: vk::CommandBuffer,
        draw_data: &imgui::DrawData,
        stats: &mut InternalStats,
    ) {
        let device = self.instance.device();
        let vkdevice = device.logical();
        let mm = self.instance.allocator();
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
            self.free_later.push((5, new_vertex_buf));
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
            self.free_later.push((5, new_index_buf));
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
            vkdevice.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            vkdevice.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                as_u8_slice(&imguipc),
            );
            vkdevice.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[self.font.1.set],
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
                vkdevice.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[self.vertex_buf.buffer],
                    &[vert_offset as u64 * std::mem::size_of::<imgui::DrawVert>() as u64],
                )
            };
            unsafe {
                vkdevice.cmd_bind_index_buffer(
                    cmd,
                    self.index_buf.buffer,
                    idx_offset as u64 * std::mem::size_of::<imgui::DrawIdx>() as u64,
                    vk::IndexType::UINT16,
                )
            };
            let vertices = draw_list.vtx_buffer();
            let dst_ptr = self
                .vertex_buf
                .allocation()
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
                .allocation()
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
                                        vkdevice.cmd_bind_pipeline(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.pipeline.pipeline,
                                        );
                                        vkdevice.cmd_push_constants(
                                            cmd,
                                            self.pipeline.layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            as_u8_slice(&imguipc),
                                        );
                                    }
                                }
                                unsafe {
                                    vkdevice.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        self.pipeline.layout,
                                        0,
                                        &[self.font.1.set],
                                        &[],
                                    );
                                }
                            } else {
                                // texture_id != FONT_ATLAS_TEXTURE_ID implicitly assumed
                                let tex_id = texture_id.id();
                                let format = self.scene_textures.as_ref().unwrap().read().unwrap()
                                    [tex_id]
                                    .format;
                                let desc = &self.tex_descs[tex_id];
                                if format == TextureFormat::RgbaSrgb && pipeline_bw {
                                    // set default pipeline
                                    pipeline_bw = false;
                                    unsafe {
                                        vkdevice.cmd_bind_pipeline(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.pipeline.pipeline,
                                        );
                                        vkdevice.cmd_push_constants(
                                            cmd,
                                            self.pipeline.layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            as_u8_slice(&imguipc),
                                        );
                                    }
                                } else if format == TextureFormat::Gray && !pipeline_bw {
                                    // set bw pipeline
                                    pipeline_bw = true;
                                    unsafe {
                                        vkdevice.cmd_bind_pipeline(
                                            cmd,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            self.pipeline_bw.pipeline,
                                        );
                                        vkdevice.cmd_push_constants(
                                            cmd,
                                            self.pipeline_bw.layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            as_u8_slice(&imguipc),
                                        );
                                    }
                                }
                                unsafe {
                                    vkdevice.cmd_bind_descriptor_sets(
                                        cmd,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        self.pipeline.layout,
                                        0,
                                        &[desc.set],
                                        &[],
                                    );
                                }
                            }
                            bound_texture = texture_id;
                        }
                        unsafe {
                            vkdevice.cmd_set_scissor(cmd, 0, &scissors);
                            vkdevice.cmd_draw_indexed(
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
        self.free_deferred_buffers();
    }

    /// Decreases the counter of deferred frees and drops them when the counter is 0.
    ///
    /// Deferred frees are buffers that will be used in the next frame only. So they cannot be
    /// dropped immediately (the GPU has to complete the frame first).
    fn free_deferred_buffers(&mut self) {
        //TODO: replace with retain_mut once stable
        self.free_later.retain(|(t, _)| *t > 0);
        self.free_later.iter_mut().for_each(|(t, _)| *t -= 1);
    }
}

impl Drop for ImguiRenderer {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .device()
                .logical()
                .destroy_sampler(self.sampler, None)
        };
    }
}

/// Builds the imgui pipeline. font_descriptor is the descriptor set containing the font texture.
/// bw indicates that the pipeline uses a grayscale texture (uses .rrr instead of .rgb to avoid
/// having red tint on single channel textures).
fn build_imgui_pipeline(
    device: Arc<ash::Device>,
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
    let cwm = vk::ColorComponentFlags::R
        | vk::ColorComponentFlags::G
        | vk::ColorComponentFlags::B
        | vk::ColorComponentFlags::A;
    builder.blending_settings = vec![vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::TRUE,
        color_write_mask: cwm,
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
fn upload_image(
    device: &Device,
    tcmdm: &mut CommandManager,
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
    let cmd = tcmdm.get_cmd_buffer();
    let transfer_queue = device.transfer_queue();
    let fence = device.immediate_execute(cmd, transfer_queue, command);
    device.wait_completion(&[fence]);
}

/// Reads a struct as a sequence of bytes
pub(crate) unsafe fn as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}
