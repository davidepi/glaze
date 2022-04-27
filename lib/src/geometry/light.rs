#[cfg(feature = "vulkan")]
use crate::include_shader;
use crate::Spectrum;
use cgmath::{Point3, Vector3 as Vec3};

#[derive(Debug, Clone, PartialEq)]
pub enum Light {
    Omni(OmniLight),
    Sun(SunLight),
    Area(AreaLight),
}

#[cfg(feature = "vulkan")]
pub const SBT_LIGHT_TYPES: usize = 3;
#[cfg(feature = "vulkan")]
pub const SBT_LIGHT_STRIDE: usize = 1;

/// Used to safely update all the light instances if new lights are added.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum LightType {
    OMNI,
    SUN,
    AREA,
}

impl LightType {
    pub fn all() -> [Self; 3] {
        [Self::OMNI, Self::SUN, Self::AREA]
    }

    pub fn name(&self) -> &'static str {
        match self {
            LightType::OMNI => "Omni",
            LightType::SUN => "Sun",
            LightType::AREA => "Area",
        }
    }

    pub fn has_position(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => false,
            LightType::AREA => false, // position for this light is given by the underlying model
        }
    }

    pub fn has_direction(&self) -> bool {
        match self {
            LightType::OMNI => false,
            LightType::SUN => true,
            LightType::AREA => false,
        }
    }

    pub fn has_intensity(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => false,
            LightType::AREA => true,
        }
    }

    pub fn has_spectrum(&self) -> bool {
        !matches!(self, LightType::AREA) // spectrum is given by the material
    }

    pub fn is_delta(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => true,
            LightType::AREA => false,
        }
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn callable_shaders() -> [Vec<u8>; SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE] {
        [
            include_shader!("light_omni_sample_visible.rcall").to_vec(),
            include_shader!("light_sun_sample_visible.rcall").to_vec(),
            include_shader!("light_area_sample_visible.rcall").to_vec(),
        ]
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn sbt_callable_index(&self) -> u32 {
        let light_index = match self {
            LightType::OMNI => 0,
            LightType::SUN => 1,
            LightType::AREA => 2,
        };
        (light_index * SBT_LIGHT_STRIDE) as u32
    }
}

impl TryFrom<u8> for LightType {
    type Error = std::io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(LightType::OMNI),
            1 => Ok(LightType::SUN),
            2 => Ok(LightType::AREA),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid enum value for LightType",
            )),
        }
    }
}

impl From<LightType> for u8 {
    fn from(ltype: LightType) -> Self {
        match ltype {
            LightType::OMNI => 0,
            LightType::SUN => 1,
            LightType::AREA => 2,
        }
    }
}

impl Light {
    pub fn new_omni(name: String, color: Spectrum, position: Point3<f32>, intensity: f32) -> Self {
        Light::Omni(OmniLight {
            name,
            color,
            position,
            intensity,
        })
    }

    pub fn new_sun(name: String, color: Spectrum, direction: Vec3<f32>) -> Self {
        Light::Sun(SunLight {
            name,
            color,
            direction,
        })
    }

    pub fn new_area(name: String, material_id: u32, intensity: f32) -> Self {
        Light::Area(AreaLight {
            name,
            material_id,
            intensity,
        })
    }

    pub fn ltype(&self) -> LightType {
        match self {
            Light::Omni(_) => LightType::OMNI,
            Light::Sun(_) => LightType::SUN,
            Light::Area(_) => LightType::AREA,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Light::Omni(l) => &l.name,
            Light::Sun(l) => &l.name,
            Light::Area(l) => &l.name,
        }
    }

    pub fn emission(&self) -> Spectrum {
        match self {
            Light::Omni(l) => l.color,
            Light::Sun(l) => l.color,
            _ => Spectrum::white(), // the emission is given by the instance material
        }
    }

    pub fn position(&self) -> Point3<f32> {
        match self {
            Light::Omni(l) => l.position,
            _ => Point3::<f32>::new(0.0, 0.0, 0.0),
        }
    }

    pub fn direction(&self) -> Vec3<f32> {
        match self {
            Light::Sun(l) => l.direction,
            _ => Vec3::<f32>::new(0.0, 1.0, 0.0),
        }
    }

    pub fn intensity(&self) -> f32 {
        match self {
            Light::Omni(l) => l.intensity,
            Light::Area(l) => l.intensity,
            _ => 1.0,
        }
    }

    pub fn material_id(&self) -> u32 {
        match self {
            Light::Area(l) => l.material_id,
            _ => u32::MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OmniLight {
    pub name: String,
    pub color: Spectrum,
    pub position: Point3<f32>,
    pub intensity: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SunLight {
    pub name: String,
    pub color: Spectrum,
    pub direction: Vec3<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AreaLight {
    pub name: String,
    pub material_id: u32,
    pub intensity: f32,
}
