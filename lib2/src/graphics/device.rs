pub trait Device {
    type GraphicError;
    fn supports_raytracing(&self) -> bool;
}
