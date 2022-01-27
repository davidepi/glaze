use std::sync::Arc;

use image::imageops::{resize, FilterType};
use image::{GenericImageView, GrayImage, ImageBuffer, Pixel, RgbaImage};

#[cfg(feature = "vulkan")]
use crate::vulkan::AllocatedImage;
#[cfg(feature = "vulkan")]
use crate::vulkan::Instance;

/// Information about the texture.
// When loaded on the GPU the image is discarded and so width and height are lost.
// For this reason, additional information about the image are stored in this struct.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TextureInfo {
    /// Name of the texture.
    /// Unused in the engine, but may aid the user.
    /// Not guaranteed to be the name of the texture used by the used in the original file, or the
    /// path to the texture. Don't expect every file to follow the same convention.
    pub name: String,
    /// Width of the texture, in pixels.
    pub width: u16,
    /// Height of the texture, in pixels.
    pub height: u16,
    /// Format of the texture.
    pub format: TextureFormat,
}

/// A texture that has been loaded on the GPU.
#[cfg(feature = "vulkan")]
pub struct TextureLoaded {
    /// Information about the texture.
    pub info: TextureInfo,
    /// The allocated buffer in the GPU.
    pub(crate) image: AllocatedImage,
    /// The TextureLoaded is exposed outside the crate and cannot outlive the instance.
    pub(crate) instance: Arc<dyn Instance + Send + Sync>,
}

/// A RGBA texture stored in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureRGBA {
    /// Information about the texture.
    pub info: TextureInfo,
    /// Data of the texture. May contain MIP maps.
    pub data: Vec<image::RgbaImage>,
}

/// A Grayscale (single channel) texture stored in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureGray {
    /// Information about the texture.
    pub info: TextureInfo,
    /// Data of the texture. May contain MIP maps.
    pub data: Vec<image::GrayImage>,
}

/// Enum listing the type of textures available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// A grayscale texture.
    /// 8 bit per channel, single channel.
    Gray,
    /// A color texture with alpha channel.
    /// 8 bit per channel, 4 channels.
    /// The channels are ordered as Red, Green, Blue, Alpha.
    Rgba,
}

impl TextureFormat {
    /// Returns the color type of the texture.
    pub fn to_color_type(self) -> image::ColorType {
        match self {
            TextureFormat::Gray => image::ColorType::L8,
            TextureFormat::Rgba => image::ColorType::Rgba8,
        }
    }
}

/// A texture stored in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Texture {
    Rgba(TextureRGBA),
    Gray(TextureGray),
}

impl Texture {
    /// Creates a new grayscale texture from the given info and image.
    pub fn new_gray(info: TextureInfo, data: image::GrayImage) -> Self {
        Texture::Gray(TextureGray {
            info,
            data: vec![data],
        })
    }

    /// Creates a new grayscale texture from the given info and MIP maps chain.
    pub fn new_gray_with_mipmaps(info: TextureInfo, data: Vec<image::GrayImage>) -> Self {
        Texture::Gray(TextureGray { info, data })
    }

    /// Creates a new color texture from the given ingo and image.
    pub fn new_rgba(info: TextureInfo, data: image::RgbaImage) -> Self {
        Texture::Rgba(TextureRGBA {
            info,
            data: vec![data],
        })
    }

    /// Creates a new color texture from the given ingo and MIP maps chain.
    pub fn new_rgba_with_mipmaps(info: TextureInfo, data: Vec<image::RgbaImage>) -> Self {
        Texture::Rgba(TextureRGBA { info, data })
    }

    /// Returns the information about the texture.
    pub fn to_info(self) -> TextureInfo {
        match self {
            Texture::Rgba(img) => img.info,
            Texture::Gray(img) => img.info,
        }
    }

    /// Returns the name of the texture.
    pub fn name(&self) -> &str {
        match self {
            Texture::Rgba(t) => &t.info.name,
            Texture::Gray(t) => &t.info.name,
        }
    }

    /// Returns the raw bytes of a specific MIP map level, given the level index.
    pub fn raw(&self, level: usize) -> &[u8] {
        match self {
            Texture::Rgba(t) => t.data[level].as_raw(),
            Texture::Gray(t) => t.data[level].as_raw(),
        }
    }

    /// Returns the pointer to the raw bytes of a specific MIP map level, given the level index.
    pub fn ptr(&self, level: usize) -> *const u8 {
        match self {
            Texture::Rgba(t) => t.data[level].as_ptr(),
            Texture::Gray(t) => t.data[level].as_ptr(),
        }
    }

    /// Returns (width, height) of the texture in pixel.
    ///
    /// The parameter level refers to the specific MIP map level.
    pub fn dimensions(&self, level: usize) -> (u16, u16) {
        let (w, h) = match self {
            Texture::Rgba(t) => (t.info.width, t.info.height),
            Texture::Gray(t) => (t.info.width, t.info.height),
        };
        (std::cmp::max(1, w >> level), std::cmp::max(1, h >> level))
    }

    /// Returns the format of the texture.
    pub const fn format(&self) -> TextureFormat {
        match self {
            Texture::Rgba(_) => TextureFormat::Rgba,
            Texture::Gray(_) => TextureFormat::Gray,
        }
    }

    /// Returns the number of MIP map levels contained in the texture.
    ///
    /// Note, this is *not* the number of possible MIP maps.
    pub fn mipmap_levels(&self) -> usize {
        match self {
            Texture::Rgba(img) => img.data.len(),
            Texture::Gray(img) => img.data.len(),
        }
    }

    /// Returns the size in bytes of a specific MIP map level.
    pub fn size_bytes(&self, level: usize) -> usize {
        match self {
            Texture::Rgba(t) => t.data[level].len(), // len already returns the bytes
            Texture::Gray(t) => t.data[level].len(),
        }
    }

    /// Returns the number of bytes per pixel in the texture.
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Texture::Gray(_) => 1,
            Texture::Rgba(_) => 4,
        }
    }

    /// Returns true if the texture contains MIP maps.
    ///
    /// This method is equivalent to `[Texture::levels()] > 1`.
    pub fn has_mipmaps(&self) -> bool {
        match self {
            Texture::Rgba(img) => img.data.len() > 1,
            Texture::Gray(img) => img.data.len() > 1,
        }
    }

    /// Generates the MIP maps for this texture.
    ///
    /// A Catmull-Rom filtering is used.
    pub fn gen_mipmaps(&mut self) {
        if !self.has_mipmaps() {
            match self {
                Texture::Rgba(img) => img.data = gen_mipmaps::<RgbaImage>(img.data.pop().unwrap()),
                Texture::Gray(img) => img.data = gen_mipmaps::<GrayImage>(img.data.pop().unwrap()),
            }
        }
    }
}

impl Default for Texture {
    fn default() -> Self {
        const WIDTH: usize = 1;
        const HEIGHT: usize = 1;
        let buf = vec![255; WIDTH * HEIGHT * 4];
        let data = image::RgbaImage::from_raw(WIDTH as u32, HEIGHT as u32, buf).unwrap();
        let info = TextureInfo {
            name: "default".to_string(),
            width: WIDTH as u16,
            height: HEIGHT as u16,
            format: TextureFormat::Rgba,
        };
        Texture::Rgba(TextureRGBA {
            info,
            data: vec![data],
        })
    }
}

/// generate the mip maps for the texture
fn gen_mipmaps<I: GenericImageView>(
    img: ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>,
) -> Vec<ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>>
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let mut w = img.width();
    let mut h = img.height();
    debug_assert!((w & (w - 1)) == 0, "texture width must be a power of 2");
    debug_assert!((h & (h - 1)) == 0, "texture height must be a power of 2");
    let mip_levels = 1 + ilog2(std::cmp::max(w, h)) as usize;
    let mut mipmaps = Vec::with_capacity(mip_levels);
    mipmaps.push(img);
    for level in 1..mip_levels {
        w = std::cmp::max(1, w >> 1);
        h = std::cmp::max(1, h >> 1);
        let mipmap = resize(&mipmaps[level - 1], w, h, FilterType::CatmullRom);
        mipmaps.push(mipmap);
    }
    mipmaps
}

/// Calculates the log2 of a integer.
/// Works only if the input is known to be a power of 2
/// Implemented because the rust variant is currently unstable.
fn ilog2(x: u32) -> u32 {
    // from https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    const B: [u32; 5] = [0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000];
    let mut r = ((x & B[0]) != 0) as u32;
    r |= (((x & B[4]) != 0) as u32) << 4;
    r |= (((x & B[3]) != 0) as u32) << 3;
    r |= (((x & B[2]) != 0) as u32) << 2;
    r |= (((x & B[1]) != 0) as u32) << 1;
    r
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    #[test]
    fn ilog2() {
        for i in 0..32 {
            let val = 1 << i;
            let log = super::ilog2(val);
            assert_eq!(log, i);
        }
    }

    #[test]
    fn mipmaps() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("checker.jpg");
        let img = image::open(path).unwrap().to_rgba8();
        let mipmaps = super::gen_mipmaps::<image::RgbaImage>(img);
        assert_eq!(mipmaps.len(), 10);
        assert_eq!(mipmaps[0].width(), 512);
        assert_eq!(mipmaps[1].width(), 256);
        assert_eq!(mipmaps[2].width(), 128);
        assert_eq!(mipmaps[3].width(), 64);
        assert_eq!(mipmaps[4].width(), 32);
        assert_eq!(mipmaps[5].width(), 16);
        assert_eq!(mipmaps[6].width(), 8);
        assert_eq!(mipmaps[7].width(), 4);
        assert_eq!(mipmaps[8].width(), 2);
        assert_eq!(mipmaps[9].width(), 1);
    }

    #[test]
    fn mipmaps_nonuniform() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("checker_nu.jpg");
        let img = image::open(path).unwrap().to_rgba8();
        let mipmaps = super::gen_mipmaps::<image::RgbaImage>(img);
        assert_eq!(mipmaps.len(), 10);
        assert_eq!(mipmaps[0].width(), 64);
        assert_eq!(mipmaps[1].width(), 32);
        assert_eq!(mipmaps[2].width(), 16);
        assert_eq!(mipmaps[3].width(), 8);
        assert_eq!(mipmaps[4].width(), 4);
        assert_eq!(mipmaps[5].width(), 2);
        assert_eq!(mipmaps[6].width(), 1);
        assert_eq!(mipmaps[7].width(), 1);
        assert_eq!(mipmaps[8].width(), 1);
        assert_eq!(mipmaps[9].width(), 1);

        assert_eq!(mipmaps[0].height(), 512);
        assert_eq!(mipmaps[1].height(), 256);
        assert_eq!(mipmaps[2].height(), 128);
        assert_eq!(mipmaps[3].height(), 64);
        assert_eq!(mipmaps[4].height(), 32);
        assert_eq!(mipmaps[5].height(), 16);
        assert_eq!(mipmaps[6].height(), 8);
        assert_eq!(mipmaps[7].height(), 4);
        assert_eq!(mipmaps[8].height(), 2);
        assert_eq!(mipmaps[9].height(), 1);
    }
}
