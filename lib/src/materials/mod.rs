mod material;
pub use self::material::Material;
mod shader;
pub use self::shader::ShaderMat;
mod texture;
#[cfg(feature = "vulkan")]
pub use self::texture::TextureLoaded;
pub use self::texture::{Texture, TextureFormat, TextureGray, TextureInfo, TextureRGBA};

pub const DEFAULT_TEXTURE_ID: u16 = 0;
pub const DEFAULT_MATERIAL_ID: u16 = 0;
