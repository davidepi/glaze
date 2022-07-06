#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod geometry;
mod materials;
mod parser;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use geometry::{
    Camera, ColorRGB, ColorXYZ, Light, LightType, Mesh, MeshInstance, OrthographicCam,
    PerspectiveCam, Spectrum, Transform, Vertex,
};
#[cfg(feature = "vulkan")]
pub use materials::{
    Material, MaterialType, Metal, Texture, TextureFormat, TextureGray, TextureInfo, TextureRGBA,
    DEFAULT_MATERIAL_ID, DEFAULT_TEXTURE_ID,
};
pub use parser::{converted_file, parse, Meta, ParsedScene, ParserVersion, Serializer};
#[cfg(feature = "vulkan")]
pub use vulkan::{
    DeviceInfo, Integrator, Pipeline, PipelineBuilder, RayTraceInstance, RayTraceRenderer,
};
#[cfg(feature = "vulkan-interactive")]
pub use vulkan::{PresentInstance, RayTraceScene, RealtimeRenderer, RealtimeScene};
