#[cfg(feature = "vulkan")]
use crate::include_shader;
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

/// A struct representing a light source.
#[derive(Debug, Clone, PartialEq)]
pub struct Light {
    /// Type of light.
    pub tp: LightType,
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
            tp: LightType::OMNI,
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
    /// Creates a new omnidirectional light.
    pub fn new_omni(name: String, color: Spectrum, position: Point3<f32>, intensity: f32) -> Self {
        Light {
            tp: LightType::OMNI,
            name,
            color,
            position,
            intensity,
            ..Default::default()
        }
    }

    /// Creates a new infinitely far away light.
    pub fn new_sun(name: String, color: Spectrum, direction: Vec3<f32>) -> Self {
        Light {
            tp: LightType::SUN,
            name,
            color,
            direction,
            ..Default::default()
        }
    }

    /// Creates a new area light.
    pub fn new_area(name: String, material_id: u32, intensity: f32) -> Self {
        Light {
            tp: LightType::AREA,
            name,
            resource_id: material_id,
            intensity,
            ..Default::default()
        }
    }

    /// Creates a new skylight.
    pub fn new_sky(name: String, tex_id: u16, yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Self {
        Light {
            name,
            tp: LightType::SKY,
            yaw_deg,
            resource_id: tex_id as u32,
            pitch_deg,
            roll_deg,
            ..Default::default()
        }
    }

    /// Returns the light type for the current light.
    pub fn ltype(&self) -> LightType {
        self.tp
    }

    /// Returns the name of the current light.
    ///
    /// Skylights have no name and will simply return "`Sky`"
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the emitted spectrum for the current light.
    ///
    /// If the light has no spectrum ([LightType::has_spectrum] is `false`), [Spectrum::white] is
    /// returned.
    pub fn emission(&self) -> Spectrum {
        self.color
    }

    /// Returns the position of the current light.
    ///
    /// If the light has no position ([LightType::has_position] is `false`), `[0.0, 0.0, 0.0]` is
    /// returned.
    pub fn position(&self) -> Point3<f32> {
        self.position
    }

    /// Returns the direction of the current light.
    ///
    /// If the light has no direction ([LightType::has_direction] is `false`), `[0.0, -1.0, 0.0]` is
    /// returned.
    pub fn direction(&self) -> Vec3<f32> {
        self.direction
    }

    /// Returns the intensity of the current light.
    ///
    /// If the light has no intensity ([LightType::has_intensity] is `false`), `1.0` is
    /// returned.
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Returns the resource id associated to the current light.
    ///
    /// This returns:
    /// - the `material_id` associated with the current area light on [LightType::AREA] lights.
    /// - the `texture_id` associated with the skydome texture on [LightType::SKY] lights.
    /// - 0 in all other cases.
    pub fn resource_id(&self) -> u32 {
        self.resource_id
    }

    /// Returns yaw, pitch and rool values in degrees if the current light is [LightType::SKY].
    /// Returns all zeroes otherwise
    pub fn yaw_pitch_roll(&self) -> [f32; 3] {
        [self.yaw_deg, self.pitch_deg, self.roll_deg]
    }

    /// Returns the rotation matrix of the skydome if the current light is [LightType::SKY].
    ///
    /// Returns a zero matrix otherwise.
    pub fn rotation_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_angle_y(cgmath::Deg(self.yaw_deg))
            * Matrix4::from_angle_z(cgmath::Deg(self.pitch_deg))
            * Matrix4::from_angle_x(cgmath::Deg(self.roll_deg))
    }
}
