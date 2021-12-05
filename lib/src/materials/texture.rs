use image::imageops::{resize, FilterType};
use image::{GenericImageView, GrayImage, ImageBuffer, Pixel, RgbaImage};

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
    pub data: Vec<image::RgbaImage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureGray {
    pub info: TextureInfo,
    pub data: Vec<image::GrayImage>,
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
        Texture::Gray(TextureGray {
            info,
            data: vec![data],
        })
    }

    pub fn new_gray_with_mipmaps(info: TextureInfo, data: Vec<image::GrayImage>) -> Self {
        Texture::Gray(TextureGray { info, data })
    }

    pub fn new_rgba(info: TextureInfo, data: image::RgbaImage) -> Self {
        Texture::Rgba(TextureRGBA {
            info,
            data: vec![data],
        })
    }

    pub fn new_rgba_with_mipmaps(info: TextureInfo, data: Vec<image::RgbaImage>) -> Self {
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

    pub fn raw(&self, level: usize) -> &[u8] {
        match self {
            Texture::Rgba(t) => t.data[level].as_raw(),
            Texture::Gray(t) => t.data[level].as_raw(),
        }
    }

    pub fn ptr(&self, level: usize) -> *const u8 {
        match self {
            Texture::Rgba(t) => t.data[level].as_ptr(),
            Texture::Gray(t) => t.data[level].as_ptr(),
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

    pub fn mipmap_levels(&self) -> usize {
        match self {
            Texture::Rgba(img) => img.data.len(),
            Texture::Gray(img) => img.data.len(),
        }
    }

    pub fn bytes(&self, level: usize) -> usize {
        match self {
            Texture::Rgba(t) => t.data[level].len(), // len already returns the bytes
            Texture::Gray(t) => t.data[level].len(),
        }
    }

    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Texture::Gray(_) => 1,
            Texture::Rgba(_) => 4,
        }
    }

    pub fn has_mipmaps(&self) -> bool {
        match self {
            Texture::Rgba(img) => img.data.len() > 1,
            Texture::Gray(img) => img.data.len() > 1,
        }
    }

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

fn gen_mipmaps<I: GenericImageView>(
    img: ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>,
) -> Vec<ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>>
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let mut w = img.width();
    let mut h = img.height();
    debug_assert!(w == h, "the texture must be square");
    debug_assert!(
        (w & (w - 1)) == 0,
        "texture dimensions must be a power of 2"
    );
    let mip_levels = ilog2(w) as usize;
    let mut mipmaps = Vec::with_capacity(mip_levels);
    mipmaps.push(img);
    for level in 1..=mip_levels {
        w >>= 1;
        h >>= 1;
        let mipmap = resize(&mipmaps[level - 1], w, h, FilterType::CatmullRom);
        mipmaps.push(mipmap);
    }
    mipmaps
}

fn ilog2(x: u32) -> u32 {
    // from https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    // because the rust variant is currently unstable
    // works only if the input is known to be a power of 2
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
}
