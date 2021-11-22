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
pub struct TextureRGB {
    pub info: TextureInfo,
    pub data: image::RgbImage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureGray {
    pub info: TextureInfo,
    pub data: image::GrayImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Gray,
    Rgb,
    Rgba,
}

impl TextureFormat {
    pub fn to_color_type(self) -> image::ColorType {
        match self {
            TextureFormat::Gray => image::ColorType::L8,
            TextureFormat::Rgb => image::ColorType::Rgb8,
            TextureFormat::Rgba => image::ColorType::Rgba8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Texture {
    Rgba(TextureRGBA),
    Rgb(TextureRGB),
    Gray(TextureGray),
}

impl Texture {
    pub fn new_gray(info: TextureInfo, data: image::GrayImage) -> Self {
        Texture::Gray(TextureGray { info, data })
    }

    pub fn new_rgb(info: TextureInfo, data: image::RgbImage) -> Self {
        Texture::Rgb(TextureRGB { info, data })
    }

    pub fn new_rgba(info: TextureInfo, data: image::RgbaImage) -> Self {
        Texture::Rgba(TextureRGBA { info, data })
    }

    pub fn to_info(self) -> TextureInfo {
        match self {
            Texture::Rgba(img) => img.info,
            Texture::Rgb(img) => img.info,
            Texture::Gray(img) => img.info,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Texture::Rgba(t) => &t.info.name,
            Texture::Rgb(t) => &t.info.name,
            Texture::Gray(t) => &t.info.name,
        }
    }

    pub fn raw(&self) -> &[u8] {
        match self {
            Texture::Rgba(t) => t.data.as_raw(),
            Texture::Rgb(t) => t.data.as_raw(),
            Texture::Gray(t) => t.data.as_raw(),
        }
    }

    pub fn ptr(&self) -> *const u8 {
        match self {
            Texture::Rgba(t) => t.data.as_ptr(),
            Texture::Rgb(t) => t.data.as_ptr(),
            Texture::Gray(t) => t.data.as_ptr(),
        }
    }

    pub fn dimensions(&self) -> (u16, u16) {
        match self {
            Texture::Rgba(t) => (t.info.width, t.info.height),
            Texture::Rgb(t) => (t.info.width, t.info.height),
            Texture::Gray(t) => (t.info.width, t.info.height),
        }
    }

    pub fn format(&self) -> TextureFormat {
        match self {
            Texture::Rgba(_) => TextureFormat::Rgba,
            Texture::Rgb(_) => TextureFormat::Rgb,
            Texture::Gray(_) => TextureFormat::Gray,
        }
    }
}
