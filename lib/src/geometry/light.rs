use crate::Spectrum;
use cgmath::{Matrix4, Point3, Vector3 as Vec3};

#[cfg(feature = "vulkan")]
/// Different types of lights in the SBT.
pub const SBT_LIGHT_TYPES: usize = 4;
#[cfg(feature = "vulkan")]
/// For each type of light in the SBT, how many shaders exists.
pub const SBT_LIGHT_STRIDE: usize = 1;

/// Enumerator listing all different types of lights.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
        true
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

    /// Returns true if the current light requires additional resources such as materials (e.g.
    /// [LightType::AREA]) or textures (e.g. [LightType::SKY]).
    pub fn has_resources(&self) -> bool {
        match self {
            LightType::OMNI => false,
            LightType::SUN => false,
            LightType::AREA => true,
            LightType::SKY => true,
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
    #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
    pub(crate) fn callable_shaders() -> [Vec<u8>; SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE] {
        [
            crate::include_shader!("light_omni_sample_visible.rcall").to_vec(),
            crate::include_shader!("light_sun_sample_visible.rcall").to_vec(),
            crate::include_shader!("light_area_sample_visible.rcall").to_vec(),
            crate::include_shader!("light_sky_sample_visible.rcall").to_vec(),
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

/// A struct representing a light source.
#[derive(Debug, Clone, PartialEq)]
pub struct Light {
    /// Type of light.
    pub ltype: LightType,
    /// Name of the light.
    pub name: String,
    /// Emitted spectrum.
    pub color: Spectrum,
    /// Position of the light.
    pub position: Point3<f32>,
    /// Direction of the light.
    pub direction: Vec3<f32>,
    /// Intensity of the light.
    pub intensity: f32,
    /// - material id of the associated emitting material in case of [LightType::AREA].
    /// - texture id of the associated skydome texture in case of [LightType::SKY].
    pub resource_id: u32,
    /// Yaw of the skydome in degrees.
    pub yaw_deg: f32,
    /// Pitch of the skydome in degrees.
    pub pitch_deg: f32,
    /// Roll of the skydome in degrees.
    pub roll_deg: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            ltype: LightType::OMNI,
            name: Default::default(),
            color: Spectrum::white(),
            position: Point3::<f32>::new(0.0, 0.0, 0.0),
            direction: Vec3::<f32>::new(0.0, -1.0, 0.0),
            intensity: 1.0,
            resource_id: 0,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        }
    }
}

impl Light {
    /// Returns the rotation matrix of the current light. Useful only for [LightType::SKY] lights.
    pub fn rotation_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_angle_y(cgmath::Deg(self.yaw_deg))
            * Matrix4::from_angle_z(cgmath::Deg(self.pitch_deg))
            * Matrix4::from_angle_x(cgmath::Deg(self.roll_deg))
    }
}
