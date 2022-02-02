use super::descriptor::{Descriptor, DescriptorSetManager};
use super::memory::{AllocatedImage, MemoryManager};
use ash::vk;
use std::ptr;
use std::sync::Arc;

/// Wrapper for a render pass and its attachments.
/// This pass is expected to save its content to a shader optimal attachment that will be later
/// copied onto the swapchain image with a [FinalRenderPass]
pub struct RenderPass {
    /// The render pass raw handle.
    pub renderpass: vk::RenderPass,
    /// Framebuffer used for the render pass.
    pub framebuffer: vk::Framebuffer,
    /// Extent of the render pass.
    pub extent: vk::Extent2D,
    /// Descriptor used to blit this render pass onto the final pass.
    pub copy_descriptor: Descriptor,
    /// Clear color of this render pass.
    pub clear_color: Vec<vk::ClearValue>,
    /// Color attachment of this render pass.
    pub color: AllocatedImage,
    /// Depth attachment of this render pass.
    _depth: Option<AllocatedImage>,
    /// Vulkan device handle.
    device: Arc<ash::Device>,
}

impl RenderPass {
    /// Creates a forward pass with a color and a depth attachment.
    /// The color format is RGBA8_SRGB, the depth format is D32_SFLOAT.
    /// This pass has only a single subpass.
    pub fn forward(
        device: Arc<ash::Device>,
        copy_sampler: vk::Sampler,
        mm: &MemoryManager,
        descriptor_creator: &mut DescriptorSetManager,
        extent: vk::Extent2D,
    ) -> RenderPass {
        let color_format = vk::Format::R8G8B8A8_SRGB;
        let depth_format = vk::Format::D32_SFLOAT;
        let color_img = mm.create_image_gpu(
            "Forward pass color image",
            color_format,
            extent,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let depth_img = mm.create_image_gpu(
            "Forward pass depth image",
            depth_format,
            extent,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            1,
        );
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk::Format::R8G8B8A8_SRGB,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::CLEAR,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let attachments = [color_attachment, depth_attachment];
        let color_attachments_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachments_ref = [vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }];
        let subpass = [vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: color_attachments_ref.len() as u32,
            p_color_attachments: color_attachments_ref.as_ptr(),
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: depth_attachments_ref.as_ptr(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        }];
        let dependency = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];
        let render_pass_ci = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpass.len() as u32,
            p_subpasses: subpass.as_ptr(),
            dependency_count: dependency.len() as u32,
            p_dependencies: dependency.as_ptr(),
        };
        let renderpass = unsafe { device.create_render_pass(&render_pass_ci, None) }
            .expect("Failed to create render pass");
        let attachments = [color_img.image_view, depth_img.image_view];
        let fb_ci = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass: renderpass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: extent.width,
            height: extent.height,
            layers: 1,
        };
        let framebuffer = unsafe { device.create_framebuffer(&fb_ci, None) }
            .expect("Failed to create frambebuffer");
        let copy_descriptor = descriptor_creator
            .new_set()
            .bind_image(
                &color_img,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                copy_sampler,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
            .build();
        let clear_color = vec![
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
        RenderPass {
            renderpass,
            framebuffer,
            extent,
            copy_descriptor,
            clear_color,
            color: color_img,
            _depth: Some(depth_img),
            device,
        }
    }

    /// Starts the render pass
    pub fn begin(&self, cmd: vk::CommandBuffer) {
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        let rp_ci = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass: self.renderpass,
            framebuffer: self.framebuffer,
            render_area,
            clear_value_count: self.clear_color.len() as u32,
            p_clear_values: self.clear_color.as_ptr(),
        };
        unsafe {
            self.device
                .cmd_begin_render_pass(cmd, &rp_ci, vk::SubpassContents::INLINE);
        }
    }

    /// Ends the render pass
    pub fn end(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_end_render_pass(cmd);
        }
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.renderpass, None);
        }
    }
}

/// The final render pass.
/// This is the render pass that will be written to the swapchain image.
pub struct FinalRenderPass {
    /// The render pass raw handle
    pub renderpass: vk::RenderPass,
    /// The framebuffer for this render pass
    pub framebuffer: vk::Framebuffer,
    /// The extent of the render pass
    pub extent: vk::Extent2D,
    /// Image of the framebuffer,
    pub image: vk::Image,
    /// Image view of the framebuffer
    pub view: vk::ImageView,
    device: Arc<ash::Device>,
}

impl FinalRenderPass {
    /// Creates the final render pass.
    /// The parameters `format`, `view` and `extent` refers to the swapchain image format, view and
    /// extent.
    pub fn new(
        device: Arc<ash::Device>,
        format: vk::Format,
        image: vk::Image,
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> FinalRenderPass {
        let attachments = [vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }];
        let color_attachments_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpass = [vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: color_attachments_ref.len() as u32,
            p_color_attachments: color_attachments_ref.as_ptr(),
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        }];
        let dependency = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];
        let render_pass_ci = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpass.len() as u32,
            p_subpasses: subpass.as_ptr(),
            dependency_count: dependency.len() as u32,
            p_dependencies: dependency.as_ptr(),
        };
        let renderpass = unsafe { device.create_render_pass(&render_pass_ci, None) }
            .expect("Failed to create render pass");
        let fb_attachments = [view];
        let fb_ci = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass: renderpass,
            attachment_count: fb_attachments.len() as u32,
            p_attachments: fb_attachments.as_ptr(),
            width: extent.width,
            height: extent.height,
            layers: 1,
        };
        let framebuffer = unsafe { device.create_framebuffer(&fb_ci, None) }
            .expect("Failed to create frambebuffer");
        FinalRenderPass {
            renderpass,
            framebuffer,
            extent,
            image,
            view,
            device,
        }
    }

    ///Begins the render pass
    pub fn begin(&self, cmd: vk::CommandBuffer) {
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        let rp_ci = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass: self.renderpass,
            framebuffer: self.framebuffer,
            render_area,
            clear_value_count: 0,
            p_clear_values: ptr::null(),
        };
        unsafe {
            self.device
                .cmd_begin_render_pass(cmd, &rp_ci, vk::SubpassContents::INLINE);
        }
    }

    /// Ends the render pass
    pub fn end(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_end_render_pass(cmd);
        }
    }
}

impl Drop for FinalRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.renderpass, None);
        }
    }
}
