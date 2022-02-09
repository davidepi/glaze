use crate::Spectrum;
use cgmath::{Point3, Vector3 as Vec3};

#[derive(Debug, Clone, PartialEq)]
pub enum Light {
    Omni(OmniLight),
    Sun(SunLight),
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
