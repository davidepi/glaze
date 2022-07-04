#[cfg(feature = "vulkan")]
use crate::include_shader;
use crate::Spectrum;
use cgmath::{Matrix4, Point3, Vector3 as Vec3};

/// Enumerator wrapping different light sources and their properties.
#[derive(Debug, Clone, PartialEq)]
pub enum Light {
    /// Omnidirectional light.
    Omni(OmniLight),
    /// Infinitely far away light.
    Sun(SunLight),
    /// Light associated with a [Mesh](crate::Mesh).
    Area(AreaLight),
    /// Light emitted by the skydome.
    Sky(SkyLight),
}

#[cfg(feature = "vulkan")]
/// Different types of lights in the SBT.
pub const SBT_LIGHT_TYPES: usize = 4;
#[cfg(feature = "vulkan")]
/// For each type of light in the SBT, how many shaders exists.
pub const SBT_LIGHT_STRIDE: usize = 1;

/// Enumerator listing all different types of lights.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum LightType {
    /// Omnidirectional light.
    OMNI,
    /// Infinitely far away light.
    SUN,
    /// Light associated with a [Mesh](crate::Mesh).
    AREA,
    /// Light emitted by the skydome.
    SKY,
}

impl LightType {
    /// Returns all different types of lights available.
    pub fn all() -> [Self; 4] {
        [Self::OMNI, Self::SUN, Self::AREA, Self::SKY]
    }

    /// Returns the current light type name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            LightType::OMNI => "Omni",
            LightType::SUN => "Sun",
            LightType::AREA => "Area",
            LightType::SKY => "Sky",
        }
    }

    /// Returns true if the current light type is expected to have a position.
    pub fn has_position(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => false,
            LightType::AREA => false,
            LightType::SKY => false,
        }
    }

    /// Returns true if the current light type is expected to have a direction.
    pub fn has_direction(&self) -> bool {
        match self {
            LightType::OMNI => false,
            LightType::SUN => true,
            LightType::AREA => false,
            LightType::SKY => false,
        }
    }

    /// Returns true if the current light type is expected to have an intensity.
    pub fn has_intensity(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => false,
            LightType::AREA => true,
            LightType::SKY => false,
        }
    }

    /// Returns true if the current light type is expected to have an emitted spectrum.
    pub fn has_spectrum(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => true,
            LightType::AREA => false,
            LightType::SKY => false,
        }
    }

    /// Returns true if the current light type is a delta light.
    ///
    /// Delta lights are light infinitely small that cannot be intersected by rays.
    pub fn is_delta(&self) -> bool {
        match self {
            LightType::OMNI => true,
            LightType::SUN => true,
            LightType::AREA => false,
            LightType::SKY => false,
        }
    }

    /// Returns all the callable shader implementations for the lights.
    #[cfg(feature = "vulkan")]
    pub(crate) fn callable_shaders() -> [Vec<u8>; SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE] {
        [
            include_shader!("light_omni_sample_visible.rcall").to_vec(),
            include_shader!("light_sun_sample_visible.rcall").to_vec(),
            include_shader!("light_area_sample_visible.rcall").to_vec(),
            include_shader!("light_sky_sample_visible.rcall").to_vec(),
        ]
    }

    #[cfg(feature = "vulkan")]
    /// Returns the callable shader implementation index for the current light type.
    pub(crate) fn sbt_callable_index(&self) -> u32 {
        let light_index = match self {
            LightType::OMNI => 0,
            LightType::SUN => 1,
            LightType::AREA => 2,
            LightType::SKY => 3,
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
            3 => Ok(LightType::SKY),
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
            LightType::SKY => 3,
        }
    }
}

impl Light {
    /// Creates a new omnidirectional light.
    pub fn new_omni(name: String, color: Spectrum, position: Point3<f32>, intensity: f32) -> Self {
        Light::Omni(OmniLight {
            name,
            color,
            position,
            intensity,
        })
    }

    /// Creates a new infinitely far away light.
    pub fn new_sun(name: String, color: Spectrum, direction: Vec3<f32>) -> Self {
        Light::Sun(SunLight {
            name,
            color,
            direction,
        })
    }

    /// Creates a new area light.
    pub fn new_area(name: String, material_id: u32, intensity: f32) -> Self {
        Light::Area(AreaLight {
            name,
            material_id,
            intensity,
        })
    }

    /// Creates a new skylight.
    pub fn new_sky(tex_id: u16, yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Self {
        Light::Sky(SkyLight {
            yaw_deg,
            tex_id,
            pitch_deg,
            roll_deg,
        })
    }

    /// Returns the light type for the current light.
    pub fn ltype(&self) -> LightType {
        match self {
            Light::Omni(_) => LightType::OMNI,
            Light::Sun(_) => LightType::SUN,
            Light::Area(_) => LightType::AREA,
            Light::Sky(_) => LightType::SKY,
        }
    }

    /// Returns the name of the current light.
    ///
    /// Skylights have no name and will simply return "`Sky`"
    pub fn name(&self) -> &str {
        match self {
            Light::Omni(l) => &l.name,
            Light::Sun(l) => &l.name,
            Light::Area(l) => &l.name,
            Light::Sky(_) => "Sky",
        }
    }

    /// Returns the emitted spectrum for the current light.
    ///
    /// If the light has no spectrum ([LightType::has_spectrum] is `false`), [Spectrum::white] is
    /// returned.
    pub fn emission(&self) -> Spectrum {
        match self {
            Light::Omni(l) => l.color,
            Light::Sun(l) => l.color,
            _ => Spectrum::white(), // the emission is given by the instance material
        }
    }

    /// Returns the position of the current light.
    ///
    /// If the light has no position ([LightType::has_position] is `false`), `[0.0, 0.0, 0.0]` is
    /// returned.
    pub fn position(&self) -> Point3<f32> {
        match self {
            Light::Omni(l) => l.position,
            _ => Point3::<f32>::new(0.0, 0.0, 0.0),
        }
    }

    /// Returns the direction of the current light.
    ///
    /// If the light has no direction ([LightType::has_direction] is `false`), `[0.0, -1.0, 0.0]` is
    /// returned.
    pub fn direction(&self) -> Vec3<f32> {
        match self {
            Light::Sun(l) => l.direction,
            _ => Vec3::<f32>::new(0.0, -1.0, 0.0),
        }
    }

    /// Returns the intensity of the current light.
    ///
    /// If the light has no intensity ([LightType::has_intensity] is `false`), `1.0` is
    /// returned.
    pub fn intensity(&self) -> f32 {
        match self {
            Light::Omni(l) => l.intensity,
            Light::Area(l) => l.intensity,
            _ => 1.0,
        }
    }

    /// Returns the material id associated to the current light.
    ///
    /// Available only on [LightType::AREA] lights, returns [u32::MAX] in all other cases.
    pub fn material_id(&self) -> u32 {
        match self {
            Light::Area(l) => l.material_id,
            _ => u32::MAX,
        }
    }

    /// Coerces the current light as a SkyLight, only if the underlying enum is SkyLight.
    ///
    /// Returns None if the enum does not contain a SkyLight.
    pub fn as_sky(&self) -> Option<SkyLight> {
        match self {
            Light::Sky(l) => Some(*l),
            _ => None,
        }
    }
}

/// Omnidirectional light.
#[derive(Debug, Clone, PartialEq)]
pub struct OmniLight {
    /// Name of the light.
    pub name: String,
    /// Emitted spectrum.
    pub color: Spectrum,
    /// Position of the light.
    pub position: Point3<f32>,
    /// Intensity of the light.
    pub intensity: f32,
}

/// Infinitely far away light.
#[derive(Debug, Clone, PartialEq)]
pub struct SunLight {
    /// Name of the light.
    pub name: String,
    /// Emitted spectrum.
    pub color: Spectrum,
    /// Direction of the light.
    pub direction: Vec3<f32>,
}

/// Light associated with a [Mesh](crate::Mesh).
///
/// An AreaLight is associated with an emitting material and applied to every object with that
/// specific material assigned.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaLight {
    /// Name of the light.
    pub name: String,
    /// Material id of the associated emitting material.
    pub material_id: u32,
    /// Intensity of the light.
    pub intensity: f32,
}

/// Light emitted by a skydome.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SkyLight {
    /// Yaw of the skydome in degrees.
    pub yaw_deg: f32,
    /// Pitch of the skydome in degrees.
    pub pitch_deg: f32,
    /// Roll of the skydome in degrees.
    pub roll_deg: f32,
    /// Texture ID of the skydome.
    pub tex_id: u16,
}

impl SkyLight {
    /// Returns the rotation matrix of the skydome.
    pub fn rotation_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_angle_y(cgmath::Deg(self.yaw_deg))
            * Matrix4::from_angle_z(cgmath::Deg(self.pitch_deg))
            * Matrix4::from_angle_x(cgmath::Deg(self.roll_deg))
    }
}

impl Default for SkyLight {
    fn default() -> Self {
        Self {
            yaw_deg: 0.0,
            pitch_deg: 180.0,
            roll_deg: 0.0,
            tex_id: 0,
        }
    }
}
