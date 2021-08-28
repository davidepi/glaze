use ash::vk;
use std::{ptr, rc::Rc};

use super::{Instance, PresentedInstance};

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    image_format: vk::Format,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    instance: Rc<PresentedInstance>,
}

impl Swapchain {
    pub fn create(instance: Rc<PresentedInstance>, width: u32, height: u32) -> Self {
        swapchain_init(instance, width, height, None)
    }

    pub fn re_create(self, width: u32, height: u32) -> Self {
        unsafe {
            self.image_views
                .iter()
                .for_each(|iw| self.instance.device().destroy_image_view(*iw, None));
        }
        swapchain_init(self.instance.clone(), width, height, Some(self.swapchain))
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn destroy(&self) {
        unsafe {
            self.image_views
                .iter()
                .for_each(|iw| self.instance.device().destroy_image_view(*iw, None));
            self.loader.destroy_swapchain(self.swapchain, None);
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
    let old_swapchain = if let Some(old) = old {
        old
    } else {
        vk::SwapchainKHR::null()
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
        old_swapchain,
    };
    let loader = ash::extensions::khr::Swapchain::new(instance.instance(), instance.device());
    let swapchain =
        unsafe { loader.create_swapchain(&ci, None) }.expect("Failed to create swapchain");
    let images = unsafe { loader.get_swapchain_images(swapchain) }.expect("Failed to get images");
    let device = instance.device();
    let image_views = images
        .into_iter()
        .map(|i| create_image_views(device, i, format.format))
        .collect();
    Swapchain {
        swapchain,
        loader,
        image_format: ci.image_format,
        extent: ci.image_extent,
        image_views,
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
