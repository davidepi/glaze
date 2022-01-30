use super::instance::{Instance, PresentInstance};
use super::renderpass::FinalRenderPass;
use super::sync::PresentFrameSync;
use ash::vk;
use std::ptr;
use std::sync::Arc;

/// The swapchain image used as target for the current frame
pub struct AcquiredImage<'a> {
    /// Frame index (in the swapchain images array)
    pub index: u32,
    /// Reference to the Final
    pub renderpass: &'a FinalRenderPass,
}

/// Wrapper for the swapchain, its image views, and the render pass writing to the images.
pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    extent: vk::Extent2D,
    render_passes: Vec<FinalRenderPass>,
    instance: Arc<PresentInstance>,
}

impl Swapchain {
    /// Creates a new swapchain with the given size.
    pub fn create(instance: Arc<PresentInstance>, width: u32, height: u32) -> Self {
        swapchain_init(instance, width, height, None)
    }

    /// Recreate the current swapchain with a new size.
    pub fn recreate(&mut self, width: u32, height: u32) {
        destroy(self, true);
        let new = swapchain_init(self.instance.clone(), width, height, Some(self.swapchain));
        *self = new;
    }

    /// Returns the swapchain image extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the swapchain raw handle.
    pub fn raw_handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    /// Returns the render pass writing to the swapchain image.
    pub fn renderpass(&self) -> vk::RenderPass {
        // all render passes are essentially the same
        self.render_passes[0].renderpass
    }

    /// Presents an image to the surface.
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

    /// Acquires the next swapchain frame to be renderered to.
    pub fn acquire_next_image<'a>(&'a self, sync: &PresentFrameSync) -> Option<AcquiredImage<'a>> {
        let acquired = unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                sync.image_available,
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
}

/// Destroys a swapchain.
/// The partial parameter is used to indicate that the swapchain will be resized (keeping the raw
/// handle as parameter for the ash::vk::SwapchainCreateInfoKHR).
fn destroy(sc: &mut Swapchain, partial: bool) {
    unsafe {
        if !partial {
            sc.loader.destroy_swapchain(sc.swapchain, None);
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        destroy(self, false);
    }
}

/// Creates a new swapchain with the given size. If an old swapchain is present, it will be recycled
/// for a faster initialization.
fn swapchain_init(
    instance: Arc<PresentInstance>,
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
        .iter()
        .map(|i| create_image_views(device.logical(), *i, format.format))
        .collect::<Vec<_>>();
    let render_passes = images
        .into_iter()
        .zip(image_views)
        .map(|(i, iw)| FinalRenderPass::new(device.logical_clone(), format.format, i, iw, extent))
        .collect();
    Swapchain {
        swapchain,
        loader,
        extent: ci.image_extent,
        render_passes,
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
