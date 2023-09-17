use super::device::Device;
use super::format::{ColorSpace, ImageFormat, PresentMode};
use crate::geometry::Extent2D;
use winit::window::Window;

pub trait PresentDevice: Device {
    type Swapchain;
    fn new_swapchain(
        &mut self,
        mode: PresentMode,
        format: ImageFormat,
        color_space: ColorSpace,
        size: Extent2D<u32>,
        window: &Window,
        triple_buffering: bool,
    ) -> Result<Self::Swapchain, Self::GraphicError>;
}

pub trait Swapchain {
    fn size(&self) -> Extent2D<u32>;
    fn triple_buffering(&self) -> bool;
    fn present_mode(&self) -> PresentMode;
}
