use std::ptr;

use super::descriptor::{Descriptor, DescriptorSetCreator};
use super::memory::{AllocatedImage, MemoryManager};
use ash::vk;

pub struct RenderPass {
    pub renderpass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub extent: vk::Extent2D,
    color: AllocatedImage,
    depth: Option<AllocatedImage>,
    pub copy_descriptor: Descriptor,
    clear_color: Vec<vk::ClearValue>,
}

impl RenderPass {
    pub fn forward(
        device: &ash::Device,
        copy_sampler: vk::Sampler,
        mm: &mut MemoryManager,
        descriptor_creator: &mut DescriptorSetCreator,
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
        );
        let depth_img = mm.create_image_gpu(
            "Forward pass depth image",
            depth_format,
            extent,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
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
        let image_info = vk::DescriptorImageInfo {
            sampler: copy_sampler,
            image_view: color_img.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let copy_descriptor = descriptor_creator
            .new_set()
            .bind_image(
                image_info,
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
            color: color_img,
            depth: Some(depth_img),
            copy_descriptor: copy_descriptor,
            clear_color,
        }
    }

    pub fn ui(
        device: &ash::Device,
        copy_sampler: vk::Sampler,
        mm: &mut MemoryManager,
        descriptor_creator: &mut DescriptorSetCreator,
        extent: vk::Extent2D,
    ) -> RenderPass {
        let color_format = vk::Format::R8G8B8A8_UNORM;
        let color_img = mm.create_image_gpu(
            "UI pass color image",
            color_format,
            extent,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        );
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let attachments = [color_attachment];
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
        let attachments = [color_img.image_view];
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
        let image_info = vk::DescriptorImageInfo {
            sampler: copy_sampler,
            image_view: color_img.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let copy_descriptor = descriptor_creator
            .new_set()
            .bind_image(
                image_info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
            .build();
        let clear_color = vec![vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }];
        RenderPass {
            renderpass,
            framebuffer,
            extent,
            color: color_img,
            depth: None,
            copy_descriptor: copy_descriptor,
            clear_color,
        }
    }

    pub fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
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
            device.cmd_begin_render_pass(cmd, &rp_ci, vk::SubpassContents::INLINE);
        }
    }

    pub fn end(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_end_render_pass(cmd);
        }
    }

    pub fn destroy(self, device: &ash::Device, mm: &mut MemoryManager) {
        mm.free_image(self.color);
        if let Some(depth) = self.depth {
            mm.free_image(depth);
        }
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.renderpass, None);
        }
    }
}

pub struct FinalRenderPass {
    pub renderpass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub extent: vk::Extent2D,
    pub view: vk::ImageView,
}

impl FinalRenderPass {
    pub fn new(
        device: &ash::Device,
        format: vk::Format,
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
            view,
        }
    }

    pub fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
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
            device.cmd_begin_render_pass(cmd, &rp_ci, vk::SubpassContents::INLINE);
        }
    }

    pub fn end(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_end_render_pass(cmd);
        }
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.renderpass, None);
        }
    }
}
