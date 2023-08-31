use crate::graphics::format::{ImageFormat, ImageUsage, PresentMode};

pub trait Device {
    type GraphicError;
    fn supports_raytracing(&self) -> bool;
    fn supports_image_format(&self, format: ImageFormat, usage: ImageUsage, optimal: bool) -> bool;
}

pub trait PresentDevice: Device {
    fn supports_swapchain(&self, format: ImageFormat) -> bool;
    fn support_present_mode(&self, present_mode: PresentMode) -> bool;
}
