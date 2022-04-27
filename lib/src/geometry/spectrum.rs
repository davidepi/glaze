#![allow(clippy::excessive_precision)]
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{ColorRGB, ColorXYZ};

/// The representation of a visibile colour
///
/// Spectrum class contains an approximation of the visible spectrum. It
/// represents a colour by sampling its EM spectrum. The samples span the range
/// 400nm to 700nm (included) with an interval of 20nm. For example, the first
/// sample is the value in the interval [400,420) nm
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Spectrum {
    pub(crate) wavelength: [f32; Spectrum::SAMPLES],
}

impl Spectrum {
    ///Number of samples.
    pub const SAMPLES: usize = 16;

    /// Interval in nanometers between each sample.
    pub const INTERVAL_NM: f32 = 20.0;

    /// Wavelength of the first sample, in nanometers
    pub const START_NM: f32 = 400.0;

    /// Returns a black spectrum composed of zero.
    pub fn black() -> Spectrum {
        Spectrum {
            wavelength: [0.0; 16],
        }
    }

    /// Returns a white spectrum (non light)
    pub fn white() -> Spectrum {
        SPECTRUM_WHITE
    }

    /// Initializes the spectrum given the temperature.
    ///
    /// Consider this spectrum as an emission from a blackbody and initialize it based on the
    /// temperature of the object.
    ///
    /// The unit of measurement for the temperature is Kelvin degrees.
    pub fn from_blackbody(temperature: f32) -> Self {
        if temperature <= 0.0 {
            Spectrum::black()
        } else {
            const PLANCK_H: f32 = 6.62606957e-34;
            const BOLTZMANN_K: f32 = 1.38064852e-23;
            // TODO: replace C with the speed of light in the given medium
            const C: f32 = 299792458.0;
            let mut current_wavelength = Spectrum::START_NM * 1e-9;
            let mut maxval = f32::MIN;
            let mut wavelength = [0.0; Spectrum::SAMPLES];
            for w in wavelength.iter_mut() {
                let first_term = 2.0 * PLANCK_H * C * C / current_wavelength.powi(5);
                let exp_term = PLANCK_H * C / (current_wavelength * temperature * BOLTZMANN_K);
                *w = first_term * 1.0 / f32::exp_m1(exp_term);
                maxval = w.max(maxval);
                current_wavelength += Spectrum::INTERVAL_NM * 1e-9;
            }
            let inv_maxval = 1.0 / maxval;
            for w in wavelength.iter_mut() {
                *w *= inv_maxval;
                *w = w.clamp(0.0, 1.0);
            }
            Spectrum { wavelength }
        }
    }

    /// Attempts to convert an sRGB color to a spectrum.
    ///
    /// Although an exact conversion is impossible, this function tries to
    /// create a spectrum given an RGB color. Note that multiple spectra can
    /// share the same RGB color, so this method being a one-to-many can lead
    /// to unexpected results.
    ///
    /// The `is_light` parameter, specifies if the color is emitted directly from a light.
    pub fn from_rgb(c: ColorRGB, is_light: bool) -> Self {
        let mut res = Spectrum::default();
        if !is_light {
            if c.r <= c.g && c.r <= c.b {
                res += SPECTRUM_WHITE * c.r;
                if c.g <= c.b {
                    res += SPECTRUM_CYAN * (c.g - c.r);
                    res += SPECTRUM_BLUE * (c.b - c.g);
                } else {
                    res += SPECTRUM_CYAN * (c.b - c.r);
                    res += SPECTRUM_GREEN * (c.g - c.b);
                }
            } else if c.g <= c.r && c.g <= c.b {
                res += SPECTRUM_WHITE * c.g;
                if c.r <= c.b {
                    res += SPECTRUM_MAGENTA * (c.r - c.g);
                    res += SPECTRUM_BLUE * (c.b - c.r);
                } else {
                    res += SPECTRUM_MAGENTA * (c.b - c.g);
                    res += SPECTRUM_RED * (c.r - c.b);
                }
            } else {
                res += SPECTRUM_WHITE * c.b;
                if c.r <= c.g {
                    res += SPECTRUM_YELLOW * (c.r - c.b);
                    res += SPECTRUM_GREEN * (c.g - c.r);
                } else {
                    res += SPECTRUM_YELLOW * (c.g - c.b);
                    res += SPECTRUM_RED * (c.r - c.g);
                }
            }
            res *= 0.94;
        } else {
            if c.r <= c.g && c.r <= c.b {
                res += SPECTRUM_WHITEL * c.r;
                if c.g <= c.b {
                    res += SPECTRUM_CYANL * (c.g - c.r);
                    res += SPECTRUM_BLUEL * (c.b - c.g);
                } else {
                    res += SPECTRUM_CYANL * (c.b - c.r);
                    res += SPECTRUM_GREENL * (c.g - c.b);
                }
            } else if c.g <= c.r && c.g <= c.b {
                res += SPECTRUM_WHITEL * c.g;
                if c.r <= c.b {
                    res += SPECTRUM_MAGENTAL * (c.r - c.g);
                    res += SPECTRUM_BLUEL * (c.b - c.r);
                } else {
                    res += SPECTRUM_MAGENTAL * (c.b - c.g);
                    res += SPECTRUM_REDL * (c.r - c.b);
                }
            } else {
                res += SPECTRUM_WHITEL * c.b;
                if c.r <= c.g {
                    res += SPECTRUM_YELLOWL * (c.r - c.b);
                    res += SPECTRUM_GREENL * (c.g - c.r);
                } else {
                    res += SPECTRUM_YELLOWL * (c.g - c.b);
                    res += SPECTRUM_REDL * (c.r - c.g);
                }
            }
            res *= 0.86445;
        }
        for i in 0..Spectrum::SAMPLES {
            res.wavelength[i] = res.wavelength[i].clamp(0.0, 1.0);
        }
        res
    }

    /// Converts the spectrum to a XYZ representation.
    pub fn to_xyz(&self) -> ColorXYZ {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        for i in 0..Spectrum::SAMPLES {
            x += self.wavelength[i] * X.wavelength[i];
            y += self.wavelength[i] * Y.wavelength[i];
            z += self.wavelength[i] * Z.wavelength[i];
        }
        x *= 100.0 * INVY_SUM;
        y *= 100.0 * INVY_SUM;
        z *= 100.0 * INVY_SUM;

        x = f32::max(x, 0.0);
        y = f32::max(y, 0.0);
        z = f32::max(z, 0.0);

        ColorXYZ { x, y, z }
    }

    /// Calculates the luminous intensity of the spectrum.
    ///
    /// This method calculates only the Y component of the XYZ representation, and can be used to
    /// express the luminous intensity.
    ///
    /// The value is clamped between 0.0 and 1.0
    pub fn luminance(&self) -> f32 {
        let mut y = 0.0;
        for i in 0..Spectrum::SAMPLES {
            y += self.wavelength[i] * Y.wavelength[i];
        }
        (y * INVY_SUM).clamp(0.0, 1.0)
    }

    /// Converts the spectrum to an array of bytes
    pub fn to_le_bytes(self) -> [u8; Spectrum::SAMPLES * 4] {
        let mut retval = [0; Spectrum::SAMPLES * 4];
        let mut index = 0;
        for val in self.wavelength {
            let bytes = f32::to_le_bytes(val);
            retval[index] = bytes[0];
            retval[index + 1] = bytes[1];
            retval[index + 2] = bytes[2];
            retval[index + 3] = bytes[3];
            index += 4;
        }
        retval
    }

    /// Creates the spectrum from an array of bytes
    pub fn from_bytes(bytes: [u8; Spectrum::SAMPLES * 4]) -> Spectrum {
        let mut wavelength = [0.0; Spectrum::SAMPLES];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            wavelength[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        Spectrum { wavelength }
    }
}

impl From<[f32; Spectrum::SAMPLES]> for Spectrum {
    fn from(wavelength: [f32; Spectrum::SAMPLES]) -> Self {
        Spectrum { wavelength }
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Self {
            wavelength: [0.0; 16],
        }
    }
}

const INVY_SUM: f32 = 0.17557178;

const X: Spectrum = Spectrum {
    wavelength: [
        0.048547909657160444,
        0.24864331478873888,
        0.33913105790813763,
        0.23759311571717262,
        0.068513086394717301,
        0.0074335845497747266,
        0.08596576422452927,
        0.30151855826377871,
        0.58514208267132439,
        0.88403650492429731,
        1.0476295638084412,
        0.91729557037353515,
        0.55824810008207959,
        0.24846323480208715,
        0.082662385882188882,
        0.023565863414357107,
    ],
};

const Y: Spectrum = Spectrum {
    wavelength: [
        0.0013634899475922187,
        0.010440415943351884,
        0.033597446996718648,
        0.077808559965342278,
        0.16970793592433134,
        0.38263264827430249,
        0.73335171341896055,
        0.95049857179323827,
        0.98971243401368458,
        0.88304891208807623,
        0.67035055945316946,
        0.43439004709323248,
        0.22639957609275976,
        0.093565923050045963,
        0.030277141408684354,
        0.0085307513410225518,
    ],
};

const Z: Spectrum = Spectrum {
    wavelength: [
        0.23185277074575425,
        1.2145957302053769,
        1.7617404375473658,
        1.4557389440139135,
        0.66370667571822806,
        0.24021451229850452,
        0.075098564475774759,
        0.020824563254912696,
        0.0045391401535986612,
        0.0017035374639090151,
        0.00090068979518643274,
        0.00025727244249234595,
        3.9191220954914268E-05,
        1.964333174934533E-06,
        0.0,
        0.0,
    ],
};

// A lot of constants needed for the various methods of the Spectrum class
const SPECTRUM_WHITE: Spectrum = Spectrum {
    wavelength: [
        1.0619347266616228,
        1.0623373513955183,
        1.0624330274817486,
        1.0624850787200137,
        1.0622213950288308,
        1.0613081599651542,
        1.0613058645182336,
        1.0618168659745209,
        1.0624642293010491,
        1.0624838864140043,
        1.0624682453762331,
        1.0625355983287506,
        1.0624016329348598,
        1.0622653248789862,
        1.060266533148627,
        1.0600420908765831,
    ],
};

const SPECTRUM_WHITEL: Spectrum = Spectrum {
    wavelength: [
        1.1560446394211681,
        1.1564162465744781,
        1.1567872929485827,
        1.1565328954114107,
        1.1565926830659454,
        1.1565499678850697,
        1.1472133116300325,
        1.1314209727068025,
        1.096408860125702,
        1.0338718350511178,
        0.96528604465789958,
        0.92067216838305188,
        0.90011672087937411,
        0.88940075523174911,
        0.88083842252481404,
        0.87810499922653207,
    ],
};

const SPECTRUM_CYAN: Spectrum = Spectrum {
    wavelength: [
        1.0240953312699979,
        1.0245612457188975,
        1.0463755555238345,
        1.0327846651059092,
        1.0478428969483209,
        1.0535090536305822,
        1.0534870576691449,
        1.0530663848751949,
        1.0549102750144981,
        0.94299173220279198,
        0.3100097408600444,
        0.0033711342032203243,
        -0.0048549813110745684,
        0.0018582205785167482,
        0.0039837672915054804,
        0.010507259067086385,
    ],
};

const SPECTRUM_CYANL: Spectrum = Spectrum {
    wavelength: [
        1.1352399582424499,
        1.1358531764433719,
        1.1362707169771014,
        1.1359364376354608,
        1.1361867189829913,
        1.135817770159788,
        1.1359519356976406,
        1.135423392708292,
        1.1224513886352236,
        0.87073337556349084,
        0.3803441995397272,
        0.051216852241201545,
        -0.011762638745943615,
        -0.01060685685959013,
        -0.006931473364874461,
        -0.0077818774183695668,
    ],
};

const SPECTRUM_MAGENTA: Spectrum = Spectrum {
    wavelength: [
        0.99302530302633674,
        1.0170691330352013,
        1.0143947530476214,
        1.0070517895374196,
        0.80112726913173504,
        0.077593476678434567,
        0.003229957831351733,
        -0.004352238640709956,
        0.0026944590704797754,
        0.28205531033673215,
        0.8570353689334701,
        0.99378492125784268,
        0.98449588288224388,
        0.8937980881442511,
        0.94958431903872431,
        0.9395992587226637,
    ],
};

const SPECTRUM_MAGENTAL: Spectrum = Spectrum {
    wavelength: [
        1.0765584064227334,
        1.0770490751029975,
        1.0731253134738323,
        1.0796647470180021,
        1.0024747756009726,
        0.4395828981593643,
        0.02042973274257508,
        -0.0015031343728669692,
        -6.099749699375323e-06,
        0.072151645981868115,
        0.48078616824947817,
        0.97313406556425108,
        1.0781818622728534,
        1.0327505540054573,
        1.0495214724241742,
        1.0257450908661028,
    ],
};

const SPECTRUM_YELLOW: Spectrum = Spectrum {
    wavelength: [
        -0.0059362362867909409,
        -0.0040293484704144403,
        0.034632747920561285,
        0.19407661745186114,
        0.45561541868250915,
        0.78117265145981962,
        1.0163873556505527,
        1.0511958466847318,
        1.0513470268321483,
        1.0515277720869929,
        1.0512298920801075,
        1.0515211534901903,
        1.0514264026060656,
        1.0513103386739624,
        1.0507004197273715,
        1.0485826837788901,
    ],
};

const SPECTRUM_YELLOWL: Spectrum = Spectrum {
    wavelength: [
        0.0001468672999305493,
        -0.00013161147654402951,
        -0.00016768424395723818,
        0.089519214436320216,
        0.74821476916582985,
        1.0340727288469598,
        1.0365778653585402,
        1.0367058054560021,
        1.0365194490895373,
        1.03661227107821,
        1.0361321399468379,
        1.0144985871415191,
        0.8293751396865352,
        0.6705682032005652,
        0.60059597683336108,
        0.58277723714307716,
    ],
};

const SPECTRUM_RED: Spectrum = Spectrum {
    wavelength: [
        0.11487922506830811,
        0.060141120462551691,
        0.0040665397109191335,
        0.010459427718803191,
        0.0035470993579631675,
        -0.0052706076654779289,
        -0.0062588252221244959,
        -0.0086496045197971341,
        0.00097200190739861079,
        0.14679380036909495,
        0.85847180162874637,
        0.99821493324988597,
        0.99605297040670981,
        1.0018494025816944,
        0.99593834054491903,
        0.9811979963396622,
    ],
};

const SPECTRUM_REDL: Spectrum = Spectrum {
    wavelength: [
        0.057139392791085111,
        0.043034047329456572,
        0.021260689526515806,
        0.001077172714861781,
        0.00057985241220036873,
        -0.00022486144117236386,
        -0.00012009820021618776,
        -0.0001991308173681336,
        0.012756076079520295,
        0.1832461591194777,
        0.51948819108311795,
        0.82120171360154059,
        0.96263010562297358,
        0.99410699787589729,
        0.99018057306059759,
        0.98278552726948454,
    ],
};

const SPECTRUM_GREEN: Spectrum = Spectrum {
    wavelength: [
        -0.010865527381003439,
        -0.010329458431599345,
        -0.0083431520558099291,
        0.083794233190453149,
        0.57500780803880425,
        0.95115677422179923,
        0.99948898769838934,
        0.99968078182605802,
        0.9988159758735875,
        0.88618140828021486,
        0.35690377193776984,
        0.01325598457467465,
        -0.0050991929756587905,
        -0.0083927995026960873,
        -0.0084413650357697944,
        -0.0047501377518373699,
    ],
};

const SPECTRUM_GREENL: Spectrum = Spectrum {
    wavelength: [
        0.0064830780912117957,
        0.00019032331867488899,
        -0.0081060480798639516,
        0.048161890183585902,
        0.66729637282872345,
        1.0307844454225901,
        1.0311600157417389,
        1.0265626896736526,
        1.0363099387922192,
        1.0120735391513225,
        0.32668720729539291,
        0.0033846154767388065,
        0.0081701266623202973,
        0.0088889810224966476,
        0.00036631914529600032,
        0.00099462806143045101,
    ],
};

const SPECTRUM_BLUE: Spectrum = Spectrum {
    wavelength: [
        0.99498216185557875,
        0.99569451590852531,
        0.99983310193704411,
        0.9648523926660395,
        0.67060127526424484,
        0.29157891777810102,
        0.044614561825850822,
        -6.7793271695393519e-06,
        0.00050597357489660954,
        0.0023497993510693772,
        0.00067442519549839989,
        0.016621955742817246,
        0.040211692914411255,
        0.049604490414015802,
        0.043574051087547458,
        0.027483432250758107,
    ],
};

const SPECTRUM_BLUEL: Spectrum = Spectrum {
    wavelength: [
        1.054236254920313,
        1.0576206026996142,
        1.058143833550661,
        1.0568818098511983,
        1.0207912014756255,
        0.2974275399820579,
        -0.0014770394250804989,
        -0.0013982161133251694,
        -0.00059190711447091779,
        -0.0010090527379278194,
        -0.0015479588813372375,
        0.0051106864601078716,
        0.047054873524993275,
        0.12827536395203271,
        0.15246421103968871,
        0.16615733676564479,
    ],
};

// All the operators
impl Add<Spectrum> for Spectrum {
    type Output = Spectrum;

    fn add(mut self, rhs: Spectrum) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] += rhs.wavelength[i];
        }
        self
    }
}

impl AddAssign<Spectrum> for Spectrum {
    fn add_assign(&mut self, rhs: Spectrum) {
        for i in 0..16 {
            self.wavelength[i] += rhs.wavelength[i];
        }
    }
}

impl Sub<Spectrum> for Spectrum {
    type Output = Spectrum;

    fn sub(mut self, rhs: Spectrum) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] -= rhs.wavelength[i];
        }
        self
    }
}

impl SubAssign<Spectrum> for Spectrum {
    fn sub_assign(&mut self, rhs: Spectrum) {
        for i in 0..16 {
            self.wavelength[i] -= rhs.wavelength[i];
        }
    }
}

impl Mul<Spectrum> for Spectrum {
    type Output = Spectrum;

    fn mul(mut self, rhs: Spectrum) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] *= rhs.wavelength[i];
        }
        self
    }
}

impl MulAssign<Spectrum> for Spectrum {
    fn mul_assign(&mut self, rhs: Spectrum) {
        for i in 0..16 {
            self.wavelength[i] *= rhs.wavelength[i];
        }
    }
}

impl Div<Spectrum> for Spectrum {
    type Output = Spectrum;

    fn div(mut self, rhs: Spectrum) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] /= rhs.wavelength[i];
        }
        self
    }
}

impl DivAssign<Spectrum> for Spectrum {
    fn div_assign(&mut self, rhs: Spectrum) {
        for i in 0..16 {
            self.wavelength[i] /= rhs.wavelength[i];
        }
    }
}

impl Add<f32> for Spectrum {
    type Output = Spectrum;

    fn add(mut self, rhs: f32) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] += rhs;
        }
        self
    }
}

impl AddAssign<f32> for Spectrum {
    fn add_assign(&mut self, rhs: f32) {
        for i in 0..16 {
            self.wavelength[i] += rhs;
        }
    }
}

impl Sub<f32> for Spectrum {
    type Output = Spectrum;

    fn sub(mut self, rhs: f32) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] -= rhs;
        }
        self
    }
}

impl SubAssign<f32> for Spectrum {
    fn sub_assign(&mut self, rhs: f32) {
        for i in 0..16 {
            self.wavelength[i] -= rhs;
        }
    }
}

impl Mul<f32> for Spectrum {
    type Output = Spectrum;

    fn mul(mut self, rhs: f32) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] *= rhs;
        }
        self
    }
}

impl MulAssign<f32> for Spectrum {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..16 {
            self.wavelength[i] *= rhs;
        }
    }
}

impl Div<f32> for Spectrum {
    type Output = Spectrum;

    fn div(mut self, rhs: f32) -> Self::Output {
        for i in 0..16 {
            self.wavelength[i] /= rhs;
        }
        self
    }
}

impl DivAssign<f32> for Spectrum {
    fn div_assign(&mut self, rhs: f32) {
        for i in 0..16 {
            self.wavelength[i] /= rhs;
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::{ColorRGB, Spectrum};

    use super::SPECTRUM_CYAN;

    #[test]
    fn spectrum_constructor_black() {
        let sp = Spectrum::black();
        let res = sp.to_xyz().to_rgb();
        assert!(res.r < 0.05);
        assert!(res.g < 0.05);
        assert!(res.b < 0.05);
    }

    #[test]
    fn spectrum_constructor_white() {
        let sp = Spectrum::white();
        let res = sp.to_xyz().to_rgb();
        assert!(res.r > 0.95);
        assert!(res.g > 0.95);
        assert!(res.b > 0.95);
    }

    #[test]
    fn spectrum_rgb_to_spectrum() {
        let rgb = ColorRGB::new(1.0, 1.0, 1.0);
        let sp = Spectrum::from_rgb(rgb, false);
        let res = sp.to_xyz().to_rgb();
        assert!(res.r > 0.95);
        assert!(res.g > 0.95);
        assert!(res.b > 0.95);
    }

    #[test]
    fn spectrum_to_xyz() {
        let rgb = ColorRGB::new(1.0, 0.0, 1.0);
        let sp = Spectrum::from_rgb(rgb, false);
        let res = sp.to_xyz();
        assert_approx_eq!(f32, res.x, 63.915726, epsilon = 1e-5);
        assert_approx_eq!(f32, res.y, 31.252344, epsilon = 1e-5);
        assert_approx_eq!(f32, res.z, 86.988340, epsilon = 1e-5);
    }

    #[test]
    fn spectrum_luminance() {
        let rgb = ColorRGB::new(1.0, 0.0, 1.0);
        let sp = Spectrum::from_rgb(rgb, false);
        assert_approx_eq!(f32, sp.luminance(), 0.31252345, epsilon = 1e-5);
    }

    #[test]
    fn spectrum_blackbody_zero() {
        let sp = Spectrum::from_blackbody(0.0);
        for i in 0..Spectrum::SAMPLES {
            assert_eq!(sp.wavelength[i], 0.0);
        }
    }

    #[test]
    fn spectrum_blackbody_red() {
        let sp = Spectrum::from_blackbody(800.0);
        let rgb = sp.to_xyz().to_rgb();
        assert_approx_eq!(f32, rgb.r, 0.4153538, epsilon = 1e-5);
        assert_eq!(rgb.g, 0.0);
        assert_eq!(rgb.b, 0.0);
    }

    #[test]
    fn spectrum_blackbody_white() {
        let sp = Spectrum::from_blackbody(6500.0);
        let rgb = sp.to_xyz().to_rgb();
        assert!(rgb.r > 0.9);
        assert!(rgb.g > 0.9);
        assert!(rgb.b > 0.9);
    }

    #[test]
    fn spectrum_blackbody_blue() {
        let sp = Spectrum::from_blackbody(20000.0);
        let rgb = sp.to_xyz().to_rgb();
        assert!(rgb.b > 0.8);
        assert!(rgb.b > rgb.r);
        assert!(rgb.b > rgb.g);
    }

    #[test]
    fn spectrum_from_to_bytes() {
        let sp = SPECTRUM_CYAN;
        let bytes = sp.to_le_bytes();
        let from = Spectrum::from_bytes(bytes);
        for i in 0..Spectrum::SAMPLES {
            assert_eq!(sp.wavelength[i], from.wavelength[i])
        }
    }

    #[test]
    fn spectrum_add_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp + sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] + sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_add_assign_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res += sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] + sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_sub_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp - sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] - sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_sub_assign_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res -= sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] - sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }
    #[test]
    fn spectrum_mul_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp * sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] * sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_mul_assign_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res *= sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] * sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_div_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp / sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] / sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_div_assign_spectrum() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res /= sp;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] / sp.wavelength[i],
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_add_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        //valid
        let sp_res = sp + 0.1;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] + 0.1,
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_add_assign_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res += 0.1;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] + 0.1,
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_sub_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp - 0.1;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] - 0.1,
                epsilon = 1e-5,
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_sub_assign_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res -= 0.1;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] - 0.1,
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_mul_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp * 0.1;
        for i in 0..16 {
            assert_approx_eq!(
                f32,
                sp_res.wavelength[i],
                sp.wavelength[i] * 0.1,
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn spectrum_mul_assign_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res *= 0.1;
        for i in 0..16 {
            assert_approx_eq!(f32, sp_res.wavelength[i], sp.wavelength[i] * 0.1);
        }
    }

    #[test]
    fn spectrum_div_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let sp_res = sp / 0.1;
        for i in 0..16 {
            assert_approx_eq!(f32, sp_res.wavelength[i], sp.wavelength[i] / 0.1);
        }
    }

    #[test]
    fn spectrum_div_assign_float() {
        let input = ColorRGB::new(0.5, 0.5, 0.5);
        let sp = Spectrum::from_rgb(input, false);

        let mut sp_res = sp;
        sp_res /= 0.1;
        for i in 0..16 {
            assert_approx_eq!(f32, sp_res.wavelength[i], sp.wavelength[i] / 0.1);
        }
    }
}
