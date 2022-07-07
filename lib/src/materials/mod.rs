mod material;
pub use self::material::{Material, MaterialType};
mod metal;
pub use self::metal::Metal;
mod texture;
#[cfg(feature = "vulkan")]
pub use self::texture::TextureLoaded;
pub use self::texture::{Texture, TextureFormat, TextureGray, TextureInfo, TextureRGBA};

pub const DEFAULT_TEXTURE_ID: u16 = 0;
pub const DEFAULT_MATERIAL_ID: u16 = 0;

#[cfg(feature = "vulkan")]
pub use self::material::SBT_MATERIAL_STRIDE;
#[cfg(feature = "vulkan")]
pub use self::material::SBT_MATERIAL_TYPES;
