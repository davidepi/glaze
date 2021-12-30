#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod geometry;
mod materials;
mod parser;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
#[cfg(feature = "vulkan")]
pub use materials::TextureLoaded;
pub use materials::{
    Material, ShaderMat, Texture, TextureFormat, TextureGray, TextureInfo, TextureRGBA,
};
pub use parser::{converted_file, parse, serialize, ParsedScene, ParserVersion};
#[cfg(feature = "vulkan")]
pub use vulkan::{
    Pipeline, PipelineBuilder, PresentInstance, RayTraceInstance, RealtimeRenderer, VulkanScene,
};
