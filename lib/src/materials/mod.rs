use image::RgbaImage;

pub type Texture = RgbaImage;
mod library;
pub use self::library::Library;
mod material;
pub use self::material::Material;
mod shader;
pub use self::shader::ShaderMat;
