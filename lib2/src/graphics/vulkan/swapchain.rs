use super::instance::InstanceVulkan;
use super::physical::PhysicalDeviceVulkan;
use super::DeviceVulkan;
use crate::geometry::Extent2D;
use crate::graphics::error::{ErrorCategory, GraphicError};
use crate::graphics::format::{ColorSpace, ImageFormat, PresentMode};
use crate::graphics::swapchain::{PresentDevice, Swapchain};
use crate::graphics::vulkan::util::is_wayland;
use ash::vk;
use std::ptr;
use winit::window::Window;

struct SurfaceVulkan {
    surface: vk::SurfaceKHR,
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
    loader: ash::extensions::khr::Surface,
}

impl SurfaceVulkan {
    fn new(
        instance: &InstanceVulkan,
        physical: &PhysicalDeviceVulkan,
        window: &Window,
    ) -> Result<Self, GraphicError> {
        let surface = unsafe { create_surface(instance, window) }?;
        let loader =
            ash::extensions::khr::Surface::new(instance.vk_loader(), instance.vk_instance());
        let capabilities =
            unsafe { loader.get_physical_device_surface_capabilities(physical.device, surface) }?;
        let formats =
            unsafe { loader.get_physical_device_surface_formats(physical.device, surface) }?;
        let present_modes =
            unsafe { loader.get_physical_device_surface_present_modes(physical.device, surface) }?;
        Ok(SurfaceVulkan {
            surface,
            loader,
            capabilities,
            formats,
            present_modes,
        })
    }
}

impl Drop for SurfaceVulkan {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.surface, None) }
    }
}

unsafe fn create_surface(
    instance: &InstanceVulkan,
    window: &Window,
) -> Result<vk::SurfaceKHR, GraphicError> {
    #[cfg(target_os = "windows")]
    {
        use std::os::raw::c_void;
        use winapi::shared::windef::HWND;
        use winapi::um::libloaderapi::GetModuleHandleW;
        use winit::platform::windows::WindowExtWindows;

        let hwnd = window.hwnd() as HWND;
        let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
        let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
            s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: Default::default(),
            hinstance,
            hwnd: hwnd as *const c_void,
        };
        let win32_surface_loader =
            ash::extensions::khr::Win32Surface::new(instance.vk_loader(), instance.vk_instance());
        let surface = win32_surface_loader.create_win32_surface(&win32_create_info, None)?;
        Ok(surface)
    }
    #[cfg(target_os = "linux")]
    {
        use winit::platform::wayland::WindowExtWayland;
        use winit::platform::x11::WindowExtX11;

        if is_wayland() {
            let error_msg =
                "Failed to get wayland window. Maybe X11 is in use but $WAYLAND_DISPLAY is set?";
            let display = window
                .wayland_display()
                .ok_or_else(|| GraphicError::new(ErrorCategory::InitFailed, error_msg))?;
            let surface = window
                .wayland_surface()
                .ok_or_else(|| GraphicError::new(ErrorCategory::InitFailed, error_msg))?;
            let surface_ci = vk::WaylandSurfaceCreateInfoKHR {
                s_type: vk::StructureType::WAYLAND_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                display,
                surface,
            };
            let wayland_surface_loader = ash::extensions::khr::WaylandSurface::new(
                instance.vk_loader(),
                instance.vk_instance(),
            );
            let surface = wayland_surface_loader.create_wayland_surface(&surface_ci, None)?;
            Ok(surface)
        } else {
            let error_msg =
                "Failed to get X11 window. Maybe wayland is in use but $WAYLAND_DISPLAY is not set?";
            let x11_display = window
                .xlib_display()
                .ok_or_else(|| GraphicError::new(ErrorCategory::InitFailed, error_msg))?;
            let x11_window = window
                .xlib_window()
                .ok_or_else(|| GraphicError::new(ErrorCategory::InitFailed, error_msg))?;
            let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
                s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: Default::default(),
                window: x11_window as vk::Window,
                dpy: x11_display as *mut vk::Display,
            };
            let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(
                instance.vk_loader(),
                instance.vk_instance(),
            );
            let surface = xlib_surface_loader.create_xlib_surface(&x11_create_info, None)?;
            Ok(surface)
        }
    }
}

pub struct SwapchainVulkan {
    swapchain: vk::SwapchainKHR,
    surface: SurfaceVulkan,
    loader: ash::extensions::khr::Swapchain,
    mode: PresentMode,
    format: ImageFormat,
    color_space: ColorSpace,
    image_extent: vk::Extent2D,
    triple_buffering: bool,
}

impl PresentDevice for DeviceVulkan {
    type Swapchain = SwapchainVulkan;

    fn new_swapchain(
        &self,
        mode: PresentMode,
        format: ImageFormat,
        color_space: ColorSpace,
        size: Extent2D<u32>,
        window: &winit::window::Window,
        triple_buffering: bool,
    ) -> Result<Self::Swapchain, GraphicError> {
        let surface = SurfaceVulkan::new(self.instance(), self.physical(), window)?;
        let loader =
            ash::extensions::khr::Swapchain::new(self.instance().vk_instance(), self.logical());
        let mut image_extent = size.to_vk();
        let swapchain = swapchain_init(
            &loader,
            &surface,
            mode,
            format,
            color_space,
            &mut image_extent,
            triple_buffering,
            vk::SwapchainKHR::null(),
        )?;
        Ok(SwapchainVulkan {
            swapchain,
            surface,
            loader,
            mode,
            format,
            color_space,
            image_extent,
            triple_buffering,
        })
    }
}

impl Swapchain for SwapchainVulkan {
    fn size(&self) -> Extent2D<u32> {
        Extent2D {
            x: self.image_extent.width,
            y: self.image_extent.height,
        }
    }

    fn triple_buffering(&self) -> bool {
        self.triple_buffering
    }

    fn present_mode(&self) -> PresentMode {
        self.mode
    }
}

#[allow(clippy::too_many_arguments)]
fn swapchain_init(
    loader: &ash::extensions::khr::Swapchain,
    surface: &SurfaceVulkan,
    mode: PresentMode,
    format: ImageFormat,
    color_space: ColorSpace,
    image_extent: &mut vk::Extent2D,
    triple_buffering: bool,
    old_swapchain: vk::SwapchainKHR,
) -> Result<vk::SwapchainKHR, GraphicError> {
    let image_format = surface
        .formats
        .iter()
        .find(|sf| sf.format == format.to_vk() && sf.color_space == color_space.to_vk())
        .ok_or_else(|| {
            GraphicError::new(
                ErrorCategory::UnsupportedFeature,
                "The requested surface format and colorspace are not supported",
            )
        })?;
    let present_mode = surface
        .present_modes
        .iter()
        .find(|&x| *x == mode.to_vk())
        .ok_or_else(|| {
            GraphicError::new(
                ErrorCategory::UnsupportedFeature,
                "The requested present mode is not supported",
            )
        })?;
    image_extent.width = image_extent.width.clamp(
        surface.capabilities.min_image_extent.width,
        surface.capabilities.max_image_extent.width,
    );
    image_extent.height = image_extent.height.clamp(
        surface.capabilities.max_image_extent.height,
        surface.capabilities.max_image_extent.height,
    );
    let max_image_count = if surface.capabilities.max_image_count == 0 {
        u32::MAX
    } else {
        surface.capabilities.max_image_count
    };
    let image_count = if triple_buffering { 3 } else { 2 }
        .clamp(surface.capabilities.min_image_count, max_image_count);
    let ci = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        surface: surface.surface,
        min_image_count: image_count,
        image_format: image_format.format,
        image_color_space: image_format.color_space,
        image_extent: *image_extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        pre_transform: surface.capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::INHERIT,
        present_mode: *present_mode,
        clipped: vk::TRUE,
        old_swapchain,
    };
    Ok(unsafe { loader.create_swapchain(&ci, None) }?)
}
