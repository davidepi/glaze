use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Representation of a visible color in the sRGB color space.
///
/// Color class contains an approximation of the visible spectrum. Instead of
/// storing several samples of the spectral wave, this class stores a
/// value composed of either Red Green and Blue ranging from `0.0` to `1.0`.
///
/// This sacrifices precision, however, the supplied input will most likely be
/// in RGB form.
#[derive(Debug, Copy, Clone)]
pub struct ColorRGB {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl ColorRGB {
    /// Creates a new color with the given `r`, `g` and `b` values.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        ColorRGB { r, g, b }
    }

    /// Converts the current color to the CIE 1931 XYZ with a D65/2° standard illuminant.
    pub fn to_xyz(self) -> ColorXYZ {
        ColorXYZ::from(self)
    }
}

/// Representation of a visible color in the CIE 1931 XYZ color space.
///
/// Color class contains an approximation of the visible spectrum.
/// Stores the color in the CIE 1931 XYZ color space, with X, Y, Z ranging from `0.0` to roughly
/// `100.0` and a D65/2° standard illuminant.
///
/// When converting this to a [ColorRGB], the standard illuminant D65 is assumed.
#[derive(Debug, Copy, Clone)]
pub struct ColorXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl ColorXYZ {
    /// Creates a new color with the given `x`, `y` and `z` values.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        ColorXYZ { x, y, z }
    }

    /// Converts the current color to the sRGB color space.
    pub fn to_rgb(self) -> ColorRGB {
        ColorRGB::from(self)
    }
}

impl From<ColorXYZ> for ColorRGB {
    fn from(col: ColorXYZ) -> Self {
        const EXP: f32 = 1.0 / 2.4;
        const INV_100: f32 = 1.0 / 100.0;
        let x = col.x * INV_100;
        let y = col.y * INV_100;
        let z = col.z * INV_100;
        let mut r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
        let mut g = x * -0.969266 + y * 1.8760108 + z * 0.0415560;
        let mut b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
        if r > 0.0031308 {
            r = 1.055 * f32::powf(r, EXP) - 0.055;
        } else {
            r *= 12.92;
        }
        if g > 0.0031308 {
            g = 1.055 * f32::powf(g, EXP) - 0.055;
        } else {
            g *= 12.92
        }
        if b > 0.0031308 {
            b = 1.055 * f32::powf(b, EXP) - 0.055;
        } else {
            b *= 12.92;
        }
        ColorRGB { r, g, b }
    }
}

impl From<ColorRGB> for ColorXYZ {
    fn from(col: ColorRGB) -> Self {
        const INV: f32 = 1.0 / 12.92;
        let mut r = if col.r > 0.04045 {
            f32::powf((col.r + 0.055) / 1.055, 2.4)
        } else {
            col.r * INV
        };
        let mut g = if col.g > 0.04045 {
            f32::powf((col.g + 0.055) / 1.055, 2.4)
        } else {
            col.g * INV
        };
        let mut b = if col.b > 0.04045 {
            f32::powf((col.b + 0.055) / 1.055, 2.4)
        } else {
            col.b * INV
        };
        r *= 100.0;
        g *= 100.0;
        b *= 100.0;
        let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        let z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;
        ColorXYZ { x, y, z }
    }
}

// Operators
impl Add<ColorRGB> for ColorRGB {
    type Output = ColorRGB;

    fn add(mut self, rhs: ColorRGB) -> Self::Output {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
        self
    }
}

impl Sub<ColorRGB> for ColorRGB {
    type Output = ColorRGB;

    fn sub(mut self, rhs: ColorRGB) -> Self::Output {
        self.r -= rhs.r;
        self.g -= rhs.g;
        self.b -= rhs.b;
        self
    }
}

impl Mul<ColorRGB> for ColorRGB {
    type Output = ColorRGB;

    fn mul(mut self, rhs: ColorRGB) -> Self::Output {
        self.r *= rhs.r;
        self.g *= rhs.g;
        self.b *= rhs.b;
        self
    }
}

impl Div<ColorRGB> for ColorRGB {
    type Output = ColorRGB;

    fn div(mut self, rhs: ColorRGB) -> Self::Output {
        self.r /= rhs.r;
        self.g /= rhs.g;
        self.b /= rhs.b;
        self
    }
}

impl AddAssign<ColorRGB> for ColorRGB {
    fn add_assign(&mut self, rhs: ColorRGB) {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
    }
}

impl SubAssign<ColorRGB> for ColorRGB {
    fn sub_assign(&mut self, rhs: ColorRGB) {
        self.r -= rhs.r;
        self.g -= rhs.g;
        self.b -= rhs.b;
    }
}

impl MulAssign<ColorRGB> for ColorRGB {
    fn mul_assign(&mut self, rhs: ColorRGB) {
        self.r *= rhs.r;
        self.g *= rhs.g;
        self.b *= rhs.b;
    }
}

impl DivAssign<ColorRGB> for ColorRGB {
    fn div_assign(&mut self, rhs: ColorRGB) {
        self.r /= rhs.r;
        self.g /= rhs.g;
        self.b /= rhs.b;
    }
}

impl Add<f32> for ColorRGB {
    type Output = ColorRGB;

    fn add(mut self, rhs: f32) -> Self::Output {
        self.r += rhs;
        self.g += rhs;
        self.b += rhs;
        self
    }
}

impl Sub<f32> for ColorRGB {
    type Output = ColorRGB;

    fn sub(mut self, rhs: f32) -> Self::Output {
        self.r -= rhs;
        self.g -= rhs;
        self.b -= rhs;
        self
    }
}

impl Mul<f32> for ColorRGB {
    type Output = ColorRGB;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.r *= rhs;
        self.g *= rhs;
        self.b *= rhs;
        self
    }
}

impl Div<f32> for ColorRGB {
    type Output = ColorRGB;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.r /= rhs;
        self.g /= rhs;
        self.b /= rhs;
        self
    }
}

impl AddAssign<f32> for ColorRGB {
    fn add_assign(&mut self, rhs: f32) {
        self.r += rhs;
        self.g += rhs;
        self.b += rhs;
    }
}

impl SubAssign<f32> for ColorRGB {
    fn sub_assign(&mut self, rhs: f32) {
        self.r -= rhs;
        self.g -= rhs;
        self.b -= rhs;
    }
}

impl MulAssign<f32> for ColorRGB {
    fn mul_assign(&mut self, rhs: f32) {
        self.r *= rhs;
        self.g *= rhs;
        self.b *= rhs;
    }
}

impl DivAssign<f32> for ColorRGB {
    fn div_assign(&mut self, rhs: f32) {
        self.r /= rhs;
        self.g /= rhs;
        self.b /= rhs;
    }
}

#[cfg(test)]
mod tests {
    use crate::{ColorRGB, ColorXYZ};
    use float_cmp::assert_approx_eq;

    #[test]
    fn xyz_to_rgb() {
        let xyz = ColorXYZ::new(23.954, 19.020, 13.234);
        let rgb = ColorRGB::from(xyz);
        assert_approx_eq!(f32, rgb.r, 0.67843, epsilon = 1e-5);
        assert_approx_eq!(f32, rgb.g, 0.39608, epsilon = 1e-5);
        assert_approx_eq!(f32, rgb.b, 0.37255, epsilon = 1e-5);
    }

    #[test]
    fn rgb_to_xyz() {
        let rgb = ColorRGB::new(0.67843, 0.39608, 0.37255);
        let xyz = ColorXYZ::from(rgb);
        assert_approx_eq!(f32, xyz.x, 23.954, epsilon = 1e-3);
        assert_approx_eq!(f32, xyz.y, 19.020, epsilon = 1e-3);
        assert_approx_eq!(f32, xyz.z, 13.234, epsilon = 1e-3);
    }

    #[test]
    fn color_rgb_add_color() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let res = c + c;
        assert_approx_eq!(f32, res.r, 1.0, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.6, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.4, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_add_assign_color() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        c += c;
        assert_approx_eq!(f32, c.r, 1.0, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.6, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.4, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_sub_color() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let res = c - c;
        assert_approx_eq!(f32, res.r, 0.0, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.0, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_sub_assign_color() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        c -= c;
        assert_approx_eq!(f32, c.r, 0.0, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.0, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_mul_color() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let res = c * c;
        assert_approx_eq!(f32, res.r, 0.25, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.09, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.04, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_mul_assign_color() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        c *= c;
        assert_approx_eq!(f32, c.r, 0.25, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.09, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.04, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_div_color() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let res = c / c;
        assert_approx_eq!(f32, res.r, 1.0, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 1.0, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_div_assign_color() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        c /= c;
        assert_approx_eq!(f32, c.r, 1.0, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 1.0, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_add_float() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.15;
        let res = c + val;
        assert_approx_eq!(f32, res.r, 0.65, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.45, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.35, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_add_assign_float() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.15;
        c += val;
        assert_approx_eq!(f32, c.r, 0.65, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.45, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.35, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_sub_float() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.15;
        let res = c - val;
        assert_approx_eq!(f32, res.r, 0.35, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.15, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.05, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_sub_assign_float() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.15;
        c -= val;
        assert_approx_eq!(f32, c.r, 0.35, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.15, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.05, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_mul_float() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.15;
        let res = c * val;
        assert_approx_eq!(f32, res.r, 0.075, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.045, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.03, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_mul_assign_float() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.15;
        c *= val;
        assert_approx_eq!(f32, c.r, 0.075, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.045, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.03, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_div_float() {
        let c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.6;
        let res = c / val;
        assert_approx_eq!(f32, res.r, 0.8333333, epsilon = 1e-5);
        assert_approx_eq!(f32, res.g, 0.5, epsilon = 1e-5);
        assert_approx_eq!(f32, res.b, 0.3333333, epsilon = 1e-5);
    }

    #[test]
    fn color_rgb_div_assign_float() {
        let mut c = ColorRGB::new(0.5, 0.3, 0.2);
        let val = 0.6;
        c /= val;
        assert_approx_eq!(f32, c.r, 0.8333333, epsilon = 1e-5);
        assert_approx_eq!(f32, c.g, 0.5, epsilon = 1e-5);
        assert_approx_eq!(f32, c.b, 0.3333333, epsilon = 1e-5);
    }
}
