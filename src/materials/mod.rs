use image::RgbaImage;

pub type Texture = RgbaImage;
mod library;
pub use self::library::Library;
mod material;
mod pipeline;
pub use self::pipeline::PipelineBuilder;
