use image::RgbaImage;

pub type Texture = RgbaImage;
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct TextureInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub channels: image::ColorType,
}

mod library;
pub use self::library::Library;
mod material;
pub use self::material::Material;
mod shader;
pub use self::shader::ShaderMat;
