use crate::Spectrum;
use cgmath::{Point3, Vector3 as Vec3};

#[derive(Debug, Clone, PartialEq)]
pub enum Light {
    Omni(OmniLight),
    Sun(SunLight),
}

/// Used to safely update all the light instances if new lights are added.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum LightType {
    OMNI,
    SUN,
}

impl LightType {
    pub fn all() -> [Self; 2] {
        [Self::OMNI, Self::SUN]
    }

    pub fn name(&self) -> &'static str {
        match self {
            LightType::OMNI => "Omni",
            LightType::SUN => "Sun",
        }
    }
}

impl Light {
    pub fn new_omni(name: String, color: Spectrum, position: Point3<f32>) -> Self {
        Light::Omni(OmniLight {
            name,
            color,
            position,
        })
    }

    pub fn new_sun(name: String, color: Spectrum, direction: Vec3<f32>) -> Self {
        Light::Sun(SunLight {
            name,
            color,
            direction,
        })
    }

    pub fn ltype(&self) -> LightType {
        match self {
            Light::Omni(_) => LightType::OMNI,
            Light::Sun(_) => LightType::SUN,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Light::Omni(l) => &l.name,
            Light::Sun(l) => &l.name,
        }
    }

    pub fn emission(&self) -> Spectrum {
        match self {
            Light::Omni(l) => l.color,
            Light::Sun(l) => l.color,
        }
    }

    pub fn position(&self) -> Point3<f32> {
        match self {
            Light::Omni(l) => l.position,
            Light::Sun(_) => Point3::<f32>::new(0.0, 0.0, 0.0),
        }
    }

    pub fn direction(&self) -> Vec3<f32> {
        match self {
            Light::Omni(_) => Vec3::<f32>::new(0.0, 0.0, 0.0),
            Light::Sun(l) => l.direction,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OmniLight {
    pub name: String,
    pub color: Spectrum,
    pub position: Point3<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SunLight {
    pub name: String,
    pub color: Spectrum,
    pub direction: Vec3<f32>,
}