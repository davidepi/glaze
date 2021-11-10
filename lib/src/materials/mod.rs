use image::RgbaImage;

pub type Texture = RgbaImage;
mod library;
pub use self::library::Library;
mod material;
pub use self::material::{Material, ShaderMat};
mod pipeline;
pub use self::pipeline::{Pipeline, PipelineBuilder};
