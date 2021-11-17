use ash::vk;
use std::ptr;

use super::device::Device;
use super::instance::{Instance, PresentInstance};
use super::renderpass::FinalRenderPass;
use super::sync::PresentFrameSync;

pub struct AcquiredImage<'a> {
    pub index: u32,
    pub renderpass: &'a FinalRenderPass,
}

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    render_passes: Vec<FinalRenderPass>,
}

impl Swapchain {
    pub fn create(instance: &PresentInstance, width: u32, height: u32) -> Self {
        swapchain_init(instance, width, height, None)
    }

    pub fn recreate(&mut self, instance: &PresentInstance, width: u32, height: u32) {
        destroy(self, instance, true);
        let new = swapchain_init(instance, width, height, Some(self.swapchain));
        *self = new;
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn swapchain_khr(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn renderpass(&self) -> vk::RenderPass {
        self.render_passes[0].renderpass
    }

    pub fn queue_present(&self, queue: vk::Queue, present_info: &vk::PresentInfoKHR) {
        unsafe {
            match self.loader.queue_present(queue, present_info) {
                Ok(_) => (),
                Err(_val @ vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!("Refusing to present out of date swapchain");
                }
                _ => panic!("Failed to acquire next image"),
            }
        }
    }

    pub fn acquire_next_image<'a>(&'a self, sync: &PresentFrameSync) -> Option<AcquiredImage<'a>> {
        let acquired = unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                sync.image_available(),
                vk::Fence::null(),
            )
        };
        match acquired {
            Ok((index, _)) => Some(AcquiredImage {
                index,
                renderpass: &self.render_passes[index as usize],
            }),
            Err(_val @ vk::Result::ERROR_OUT_OF_DATE_KHR) => None,
            _ => panic!("Failed to acquire next image"),
        }
    }

    pub fn destroy(mut self, instance: &PresentInstance) {
        destroy(&mut self, instance, false);
    }
}

fn destroy(sc: &mut Swapchain, instance: &PresentInstance, partial: bool) {
    unsafe {
        sc.render_passes
            .drain(..)
            .for_each(|r| r.destroy(instance.device().logical()));
        sc.image_views
            .drain(..)
            .for_each(|iw| instance.device().logical().destroy_image_view(iw, None));
        if !partial {
            sc.loader.destroy_swapchain(sc.swapchain, None);
        }
    }
}

fn swapchain_init(
    instance: &PresentInstance,
    width: u32,
    height: u32,
    old: Option<vk::SwapchainKHR>,
) -> Swapchain {
    let surface_cap = instance.surface_capabilities();
    let device = instance.device();
    let capabilities = surface_cap.capabilities;
    let format = surface_cap
        .formats
        .iter()
        .find(|sf| {
            sf.format == vk::Format::B8G8R8A8_SRGB
                && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| {
            let default = surface_cap.formats.first().unwrap();
            log::warn!("Failed to find suitable surface format, using the first one available");
            default
        });
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
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        pre_transform: capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        old_swapchain: old.unwrap_or_else(vk::SwapchainKHR::null),
    };
    let loader = ash::extensions::khr::Swapchain::new(instance.instance(), device.logical());
    let swapchain =
        unsafe { loader.create_swapchain(&ci, None) }.expect("Failed to create swapchain");
    let images = unsafe { loader.get_swapchain_images(swapchain) }.expect("Failed to get images");
    let image_views = images
        .into_iter()
        .map(|i| create_image_views(device.logical(), i, format.format))
        .collect::<Vec<_>>();
    let render_passes = image_views
        .iter()
        .map(|iw| FinalRenderPass::new(device.logical(), format.format, *iw, extent))
        .collect();
    Swapchain {
        swapchain,
        loader,
        extent: ci.image_extent,
        image_views,
        render_passes,
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
