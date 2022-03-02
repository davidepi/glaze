mod material;
pub use self::material::Material;
mod shader;
pub use self::shader::ShaderMat;
mod metal;
mod texture;
pub use self::metal::Metal;
#[cfg(feature = "vulkan")]
pub use self::texture::TextureLoaded;
pub use self::texture::{Texture, TextureFormat, TextureGray, TextureInfo, TextureRGBA};

pub const DEFAULT_TEXTURE_ID: u16 = 0;
pub const DEFAULT_MATERIAL_ID: u16 = 0;

pub(crate) const SBT_MATERIAL_STRIDE: usize = 2;
pub(crate) const SBT_MATERIAL_TYPES: usize = 5;
