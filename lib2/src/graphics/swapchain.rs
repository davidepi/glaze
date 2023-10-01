use super::device::Device;
use super::error::GraphicError;
use super::format::{ImageFormat, PresentMode};
use crate::math::Extent2D;
use winit::window::Window;

pub trait PresentDevice: Device {
    type Swapchain;
    fn new_swapchain(
        &self,
        mode: PresentMode,
        format: ImageFormat,
        size: Extent2D<u32>,
        window: &Window,
        triple_buffering: bool,
    ) -> Result<Self::Swapchain, GraphicError>;
}

pub trait Swapchain {
    fn size(&self) -> Extent2D<u32>;
    fn triple_buffering(&self) -> bool;
    fn present_mode(&self) -> PresentMode;
}
