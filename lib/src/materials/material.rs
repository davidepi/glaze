use crate::{Metal, ShaderMat};

/// A material used to determine the surface of a [Mesh](crate::Mesh).
#[derive(Debug, PartialEq, Clone)]
pub struct Material {
    /// Name used to refer to a material. The name is not unique and used only to aid the user.
    pub name: String,
    /// Type of material. Determines how the light behaves on the surface.
    pub shader: ShaderMat,
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
            shader: ShaderMat::DEFAULT_SHADER,
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
