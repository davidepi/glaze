#[cfg(feature = "vulkan")]
use crate::geometry::{SBT_LIGHT_STRIDE, SBT_LIGHT_TYPES};
use crate::Metal;
use std::error::Error;

#[cfg(feature = "vulkan")]
/// Different types of materials in the SBT.
pub const SBT_MATERIAL_TYPES: usize = 6;
#[cfg(feature = "vulkan")]
/// For each type of material in the SBT, how many shaders exists.
pub const SBT_MATERIAL_STRIDE: usize = 2;

/// Categorize the light interacts with a material.
///
/// Material type shares the same underlying functions but with different parameters (e.g.
/// different reflected color of absorption coefficient)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[allow(non_camel_case_types)]
pub enum MaterialType {
    /// Flat shading.
    /// Light does not affect the material and the diffuse color is shown unaltered.
    /// In raytrace mode, this has the same effect of "Lambert"
    FLAT,
    /// Lambert BRDF for raytracing.
    LAMBERT,
    /// Mirror like material fo raytracing.
    MIRROR,
    /// Perfectly flat transmittent surface.
    GLASS,
    /// Purely metallic material.
    METAL,
    /// Frosted glass-like material.
    FROSTED,
    /// Generic physically based material.
    UBER,
    /// Flat shading.
    /// Internal version, used for two-sided polygons and polygons with opacity maps.
    /// This version is automatically assigned by the engine and **SHOULD NOT** be used.
    INTERNAL_FLAT_2SIDED,
}

impl MaterialType {
    pub const DEFAULT_MAT_TYPE: Self = MaterialType::LAMBERT;

    /// Returns each material type name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            MaterialType::FLAT | MaterialType::INTERNAL_FLAT_2SIDED => "Flat",
            MaterialType::LAMBERT => "Lambert",
            MaterialType::MIRROR => "Mirror",
            MaterialType::GLASS => "Glass",
            MaterialType::METAL => "Metal",
            MaterialType::FROSTED => "Frosted",
            MaterialType::UBER => "Generic (GGX)",
        }
    }

    /// Returns an unique number identifiying the material type.
    /// If the type corresponding to the number does not exist, an error is raised.
    /// Using an internal ID wil
    pub fn from_id(id: u8) -> Result<Self, Box<dyn Error>> {
        match id {
            0 => Ok(MaterialType::FLAT),
            1 => Ok(MaterialType::LAMBERT),
            2 => Ok(MaterialType::MIRROR),
            3 => Ok(MaterialType::GLASS),
            4 => Ok(MaterialType::METAL),
            5 => Ok(MaterialType::FROSTED),
            6 => Ok(MaterialType::UBER),
            _ => Err(format!("Unknown material type: {}", id).into()),
        }
    }

    /// Returns the id corresponding to the material type.
    pub const fn id(&self) -> u8 {
        match self {
            MaterialType::FLAT => 0,
            MaterialType::LAMBERT => 1,
            MaterialType::MIRROR => 2,
            MaterialType::GLASS => 3,
            MaterialType::METAL => 4,
            MaterialType::FROSTED => 5,
            MaterialType::UBER => 6,
            _ => panic!("Internal material types have no ID assigned"),
        }
    }

    /// Iterates all the possible assignable material types.
    /// Types used internally by the engine are skipped.
    pub fn all_values() -> [MaterialType; 7] {
        [
            MaterialType::UBER,
            MaterialType::FLAT,
            MaterialType::LAMBERT,
            MaterialType::MIRROR,
            MaterialType::GLASS,
            MaterialType::METAL,
            MaterialType::FROSTED,
        ]
    }

    /// Returns true if the material type is perfecly specular in any case (mirror or clean glass).
    pub fn is_specular(&self) -> bool {
        match self {
            MaterialType::FLAT => false,
            MaterialType::LAMBERT => false,
            MaterialType::MIRROR => true,
            MaterialType::GLASS => true,
            MaterialType::METAL => false,
            MaterialType::FROSTED => false,
            MaterialType::UBER => false,
            MaterialType::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns true if the material type is expected to have a diffuse map.
    pub fn has_diffuse(&self) -> bool {
        match self {
            MaterialType::FLAT => true,
            MaterialType::LAMBERT => true,
            MaterialType::MIRROR => false,
            MaterialType::GLASS => false,
            MaterialType::METAL => false,
            MaterialType::FROSTED => false,
            MaterialType::UBER => true,
            MaterialType::INTERNAL_FLAT_2SIDED => true,
        }
    }

    /// Returns true if the material type is expected to have a roughness map.
    pub fn has_roughness(&self) -> bool {
        match self {
            MaterialType::FLAT => false,
            MaterialType::LAMBERT => false,
            MaterialType::MIRROR => false,
            MaterialType::GLASS => false,
            MaterialType::METAL => true,
            MaterialType::FROSTED => true,
            MaterialType::UBER => true,
            MaterialType::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns true if the material type is expected to have a metalness map.
    pub fn has_metalness(&self) -> bool {
        match self {
            MaterialType::FLAT => false,
            MaterialType::LAMBERT => false,
            MaterialType::MIRROR => false,
            MaterialType::GLASS => false,
            MaterialType::METAL => false,
            MaterialType::FROSTED => false,
            MaterialType::UBER => true,
            MaterialType::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns true if the material type is expected to have an anisotropy map.
    pub fn has_anisotropy(&self) -> bool {
        match self {
            MaterialType::FLAT => false,
            MaterialType::LAMBERT => false,
            MaterialType::MIRROR => false,
            MaterialType::GLASS => false,
            MaterialType::METAL => true,
            MaterialType::FROSTED => true,
            MaterialType::UBER => true,
            MaterialType::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns true if the material type is expected to have a normal map.
    pub fn has_normal(&self) -> bool {
        true
    }

    /// Returns true if the material type is expected to have an opacity map.
    pub fn has_opacity(&self) -> bool {
        true
    }

    /// Returns true if the material type is expected to behave like a Fresnel conductor.
    pub fn is_fresnel_conductor(&self) -> bool {
        match self {
            MaterialType::FLAT => false,
            MaterialType::LAMBERT => false,
            MaterialType::MIRROR => true,
            MaterialType::GLASS => false,
            MaterialType::METAL => true,
            MaterialType::FROSTED => false,
            MaterialType::UBER => true,
            MaterialType::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns true if the material type is expected to behave like a Fresnel dielectric.
    pub fn is_fresnel_dielectric(&self) -> bool {
        match self {
            MaterialType::FLAT => false,
            MaterialType::LAMBERT => false,
            MaterialType::MIRROR => false,
            MaterialType::GLASS => true,
            MaterialType::METAL => false,
            MaterialType::FROSTED => true,
            MaterialType::UBER => true,
            MaterialType::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns true if the material type is expected to have an emissive color
    pub fn has_emission(&self) -> bool {
        matches!(self, MaterialType::FLAT | MaterialType::LAMBERT)
    }

    /// Consumes a material type and returns its internal version supporting two-sided polygons.
    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn two_sided_viewport(self) -> Self {
        MaterialType::INTERNAL_FLAT_2SIDED
    }

    /// Returns all the callable shaders implementations for every material type.
    #[cfg(all(feature = "vulkan", not(target_os = "macos")))]
    pub(crate) fn callable_shaders() -> [Vec<u8>; SBT_MATERIAL_TYPES * SBT_MATERIAL_STRIDE] {
        [
            crate::include_shader!("mat_lambert_value.rcall").to_vec(),
            crate::include_shader!("mat_lambert_sample_value.rcall").to_vec(),
            crate::include_shader!("mat_mirror_value.rcall").to_vec(),
            crate::include_shader!("mat_mirror_sample_value.rcall").to_vec(),
            crate::include_shader!("mat_glass_value.rcall").to_vec(),
            crate::include_shader!("mat_glass_sample_value.rcall").to_vec(),
            crate::include_shader!("mat_metal_value.rcall").to_vec(),
            crate::include_shader!("mat_metal_sample_value.rcall").to_vec(),
            crate::include_shader!("mat_frosted_value.rcall").to_vec(),
            crate::include_shader!("mat_frosted_sample_value.rcall").to_vec(),
            crate::include_shader!("mat_uber_value.rcall").to_vec(),
            crate::include_shader!("mat_uber_sample_value.rcall").to_vec(),
        ]
    }

    /// Returns the callable shader implementation index for the current material type.
    ///
    /// This is the index in the array returned by [MaterialType::callable_shaders]
    #[cfg(feature = "vulkan")]
    pub(crate) fn sbt_callable_index(&self) -> u32 {
        let base_index = SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE; // lights before mats
        let shader_index = match self {
            MaterialType::FLAT | MaterialType::LAMBERT => 0,
            MaterialType::MIRROR => 1,
            MaterialType::GLASS => 2,
            MaterialType::METAL => 3,
            MaterialType::FROSTED => 4,
            MaterialType::UBER => 5,
            MaterialType::INTERNAL_FLAT_2SIDED => {
                panic!("This shader should not appear in the sbt")
            }
        };
        (base_index + shader_index * SBT_MATERIAL_STRIDE) as u32
    }
}

impl Default for MaterialType {
    fn default() -> Self {
        MaterialType::DEFAULT_MAT_TYPE
    }
}

impl From<u8> for MaterialType {
    fn from(num: u8) -> Self {
        if let Ok(shader) = MaterialType::from_id(num) {
            shader
        } else {
            Self::DEFAULT_MAT_TYPE
        }
    }
}

impl From<MaterialType> for u8 {
    fn from(shader: MaterialType) -> Self {
        shader.id()
    }
}

/// A material used to determine the surface of a [Mesh](crate::Mesh).
#[derive(Debug, PartialEq, Clone)]
pub struct Material {
    /// Name used to refer to a material. The name is not unique and used only to aid the user.
    pub name: String,
    /// Type of material. Determines how the light behaves on the surface.
    pub mtype: MaterialType,
    /// If the material exhibits metallicness, this value contains the metal reference.
    pub metal: Metal,
    /// If the material is a dielectric, this value stores the index of refraction.
    pub ior: f32,
    /// Index of the texture used to determine the diffuse color of the material.
    /// 0 if no diffuse.
    pub diffuse: u16,
    /// Multiplier for the diffuse color.
    pub diffuse_mul: [u8; 3],
    /// Index of the texture used to specify the roughness of the material.
    /// 0 if no roughness map.
    pub roughness: u16,
    /// Multiplier for the roughess of a material, between 0.0 and 1.0.
    pub roughness_mul: f32,
    /// Index of the texture used to specify the metallicness of the material.
    /// 0 if no metallic map.
    pub metalness: u16,
    /// Multiplier for the metallicness of a material, either 0 or 1.
    pub metalness_mul: f32,
    /// Anisotropy of a material, between -1.0 and 1.0.
    /// The negative sign specifies the direction.
    pub anisotropy: f32,
    /// Index of the texture used to determine the opacity mask of the material.
    /// 0 if no opacity.
    pub opacity: u16,
    /// Index of the texture used to determine the normal mapping of the material.
    /// 0 if no normal map.
    pub normal: u16,
    /// Color of the emitted light
    pub emissive_col: Option<[u8; 3]>,
}

impl Default for Material {
    /// Generates a default, flat, white material.
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            mtype: MaterialType::DEFAULT_MAT_TYPE,
            metal: Metal::SILVER,
            ior: 1.46,
            diffuse: 0,
            diffuse_mul: [255, 255, 255],
            opacity: 0,
            normal: 0,
            roughness: 0,
            roughness_mul: 1.0,
            metalness: 0,
            metalness_mul: 0.0,
            anisotropy: 0.0,
            emissive_col: None,
        }
    }
}
