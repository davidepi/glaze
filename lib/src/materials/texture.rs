#[cfg(feature = "vulkan")]
use crate::vulkan::AllocatedImage;

// when loaded on the GPU the image is discarded so also width and height are lost.
// for this reason we are storing the values in this struct
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TextureInfo {
    pub name: String,
    pub width: u16,
    pub height: u16,
    pub format: TextureFormat,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
pub struct TextureLoaded {
    pub info: TextureInfo,
    pub image: AllocatedImage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureRGBA {
    pub info: TextureInfo,
    pub data: image::RgbaImage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureGray {
    pub info: TextureInfo,
    pub data: image::GrayImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Gray,
    Rgba,
}

impl TextureFormat {
    pub fn to_color_type(self) -> image::ColorType {
        match self {
            TextureFormat::Gray => image::ColorType::L8,
            TextureFormat::Rgba => image::ColorType::Rgba8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Texture {
    Rgba(TextureRGBA),
    Gray(TextureGray),
}

impl Texture {
    pub fn new_gray(info: TextureInfo, data: image::GrayImage) -> Self {
        Texture::Gray(TextureGray { info, data })
    }

    pub fn new_rgba(info: TextureInfo, data: image::RgbaImage) -> Self {
        Texture::Rgba(TextureRGBA { info, data })
    }

    pub fn to_info(self) -> TextureInfo {
        match self {
            Texture::Rgba(img) => img.info,
            Texture::Gray(img) => img.info,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Texture::Rgba(t) => &t.info.name,
            Texture::Gray(t) => &t.info.name,
        }
    }

    pub fn raw(&self) -> &[u8] {
        match self {
            Texture::Rgba(t) => t.data.as_raw(),
            Texture::Gray(t) => t.data.as_raw(),
        }
    }

    pub fn ptr(&self) -> *const u8 {
        match self {
            Texture::Rgba(t) => t.data.as_ptr(),
            Texture::Gray(t) => t.data.as_ptr(),
        }
    }

    pub fn dimensions(&self) -> (u16, u16) {
        match self {
            Texture::Rgba(t) => (t.info.width, t.info.height),
            Texture::Gray(t) => (t.info.width, t.info.height),
        }
    }

    pub const fn format(&self) -> TextureFormat {
        match self {
            Texture::Rgba(_) => TextureFormat::Rgba,
            Texture::Gray(_) => TextureFormat::Gray,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Texture::Rgba(t) => t.data.len(), // len already returns the bytes
            Texture::Gray(t) => t.data.len(),
        }
    }

    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Texture::Gray(_) => 1,
            Texture::Rgba(_) => 4,
        }
    }
}

impl Default for Texture {
    fn default() -> Self {
        const WIDTH: usize = 2;
        const HEIGHT: usize = 2;
        let buf = vec![255; WIDTH * HEIGHT * 4];
        let data = image::RgbaImage::from_raw(WIDTH as u32, HEIGHT as u32, buf).unwrap();
        let info = TextureInfo {
            name: "default".to_string(),
            width: WIDTH as u16,
            height: HEIGHT as u16,
            format: TextureFormat::Rgba,
        };
        Texture::Rgba(TextureRGBA { info, data })
    }
}
