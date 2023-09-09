use super::device::Device;
use super::format::{ImageFormat, PresentMode};
use crate::geometry::Extent2D;
use winit::window::Window;

pub trait PresentDevice: Device {
    type Swapchain;
    fn new_swapchain(
        &mut self,
        mode: PresentMode,
        format: ImageFormat,
        size: Extent2D<u32>,
        window: &Window,
        triple_buffering: bool,
        wayland: bool,
    ) -> Result<Self::Swapchain, Self::GraphicError>;
}

pub trait Swapchain {
    fn size(&self) -> Extent2D<u32>;
    fn set_size(&mut self, size: Extent2D<u32>);
}
