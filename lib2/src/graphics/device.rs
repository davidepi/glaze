use crate::graphics::format::{ImageFormat, ImageUsage, PresentMode};

pub trait Device {
    type GraphicError;
    fn supports_raytracing(&self) -> bool;
}
