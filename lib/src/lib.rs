#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod geometry;
mod materials;
mod parser;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use geometry::{
    Camera, Mesh, MeshInstance, OrthographicCam, PerspectiveCam, Transform, Vertex,
};
#[cfg(feature = "vulkan")]
pub use materials::TextureLoaded;
pub use materials::{
    Material, ShaderMat, Texture, TextureFormat, TextureGray, TextureInfo, TextureRGBA,
};
pub use parser::{converted_file, parse, serialize, ParsedScene, ParserVersion};
#[cfg(feature = "vulkan")]
pub use vulkan::{DeviceInfo, Pipeline, PipelineBuilder, RayTraceInstance, RayTraceRenderer};
#[cfg(feature = "vulkan-interactive")]
pub use vulkan::{PresentInstance, RealtimeRenderer, VulkanScene};
