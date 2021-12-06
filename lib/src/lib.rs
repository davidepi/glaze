#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod geometry;
mod materials;
mod parser;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
pub use materials::{Material, ShaderMat, Texture, TextureFormat, TextureInfo};
pub use parser::{converted_file, parse, serialize, ParserVersion, ReadParsed};
#[cfg(feature = "vulkan")]
pub use vulkan::{PresentInstance, RealtimeRenderer, VulkanScene};
