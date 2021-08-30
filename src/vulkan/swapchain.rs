use ash::vk;
use std::ptr;
use std::rc::Rc;

use super::Instance;
use super::PresentedInstance;

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    image_format: vk::Format,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    renderpass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    instance: Rc<PresentedInstance>,
}

impl Swapchain {
    pub fn create(instance: Rc<PresentedInstance>, width: u32, height: u32) -> Self {
        swapchain_init(instance, width, height, None)
    }

    pub fn image_format(&self) -> vk::Format {
        self.image_format
    }

    pub fn re_create(self, width: u32, height: u32) -> Self {
        swapchain_destroy(&self, true);
        swapchain_init(self.instance.clone(), width, height, Some(self.swapchain))
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        swapchain_destroy(&self, false);
    }
}

fn swapchain_destroy(swap: &Swapchain, partial: bool) {
    unsafe {
        swap.framebuffers
            .iter()
            .for_each(|fb| swap.instance.device().destroy_framebuffer(*fb, None));
        swap.instance
            .device()
            .destroy_render_pass(swap.renderpass, None);
        swap.image_views
            .iter()
            .for_each(|iw| swap.instance.device().destroy_image_view(*iw, None));
        if !partial {
            swap.loader.destroy_swapchain(swap.swapchain, None);
        }
    }
}

fn swapchain_init(
    instance: Rc<PresentedInstance>,
    width: u32,
    height: u32,
    old: Option<vk::SwapchainKHR>,
) -> Swapchain {
    let surface_cap = instance.surface_capabilities();
    let capabilities = surface_cap.capabilities;
    let queue_fam = instance.physical_device().queue_indices;
    let format = surface_cap
        .formats
        .iter()
        .find(|sf| {
            sf.format == vk::Format::B8G8R8A8_SRGB
                && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| surface_cap.formats.first().unwrap());
    let present_mode = *surface_cap
        .present_modes
        .iter()
        .find(|&x| *x == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(&vk::PresentModeKHR::FIFO);
    let extent = if surface_cap.capabilities.current_extent.width != u32::MAX {
        surface_cap.capabilities.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };
    let image_count = if surface_cap.capabilities.max_image_count == 0 {
        surface_cap.capabilities.min_image_count + 1
    } else {
        surface_cap
            .capabilities
            .max_image_count
            .max(surface_cap.capabilities.min_image_count + 1)
    };
    let queue_families_indices = [queue_fam.graphics_family, queue_fam.present_family];
    let (image_sharing_mode, queue_family_index_count, p_queue_family_indices) =
        if queue_fam.graphics_family != queue_fam.present_family {
            (
                vk::SharingMode::CONCURRENT,
                2,
                queue_families_indices.as_ptr(),
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
        };
    let ci = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: vk::SwapchainCreateFlagsKHR::empty(),
        surface: instance.surface().surface,
        min_image_count: image_count,
        image_format: format.format,
        image_color_space: format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode,
        queue_family_index_count,
        p_queue_family_indices,
        pre_transform: capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode: present_mode,
        clipped: vk::TRUE,
        old_swapchain: old.unwrap_or(vk::SwapchainKHR::null()),
    };
    let loader = ash::extensions::khr::Swapchain::new(instance.instance(), instance.device());
    let swapchain =
        unsafe { loader.create_swapchain(&ci, None) }.expect("Failed to create swapchain");
    let images = unsafe { loader.get_swapchain_images(swapchain) }.expect("Failed to get images");
    let image_views = images
        .into_iter()
        .map(|i| create_image_views(instance.device(), i, format.format))
        .collect::<Vec<_>>();
    let renderpass = create_renderpass(instance.device(), format.format);
    let framebuffers = create_framebuffers(instance.device(), extent, &image_views, renderpass);
    Swapchain {
        swapchain,
        loader,
        image_format: ci.image_format,
        extent: ci.image_extent,
        image_views,
        renderpass,
        framebuffers,
        instance,
    }
}

fn create_image_views(device: &ash::Device, image: vk::Image, format: vk::Format) -> vk::ImageView {
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let iw_ci = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        p_next: ptr::null(),
        flags: Default::default(),
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        components: vk::ComponentMapping::default(),
        subresource_range,
    };
    unsafe { device.create_image_view(&iw_ci, None) }.expect("Failed to create Image View")
}

fn create_renderpass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let color_attachment = [vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    }];
    let color_attachment_ref = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = [vk::SubpassDescription {
        flags: vk::SubpassDescriptionFlags::empty(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: color_attachment_ref.len() as u32,
        p_color_attachments: color_attachment_ref.as_ptr(),
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
        attachment_count: color_attachment.len() as u32,
        p_attachments: color_attachment.as_ptr(),
        subpass_count: subpass.len() as u32,
        p_subpasses: subpass.as_ptr(),
        dependency_count: dependency.len() as u32,
        p_dependencies: dependency.as_ptr(),
    };
    unsafe { device.create_render_pass(&render_pass_ci, None) }
        .expect("Failed to create render pass")
}

fn create_framebuffers(
    device: &ash::Device,
    extent: vk::Extent2D,
    image_views: &[vk::ImageView],
    render_pass: vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    let mut retval = Vec::with_capacity(image_views.len());
    for view in image_views {
        let attachments = [*view];
        let fb_ci = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: extent.width,
            height: extent.height,
            layers: 1,
        };
        let fb = unsafe { device.create_framebuffer(&fb_ci, None) }
            .expect("Failed to create frambebuffer");
        retval.push(fb);
    }
    retval
}
