use crate::ShaderMat;

/// A material used to determine the surface of a [Mesh][crate::Mesh].
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Material {
    /// Name used to refer to a material. The name is not unique and used only to aid the user.
    pub name: String,
    /// Type of material. Determines how the light behaves on the surface.
    pub shader: ShaderMat,
    /// Index of the texture used to determine the diffuse color of the material.
    /// 0 if no diffuse.
    pub diffuse: u16,
    /// Multiplier for the diffuse color.
    pub diffuse_mul: [u8; 4],
    /// Index of the texture used to determine the opacity mask of the material.
    /// 0 if no opacity.
    pub opacity: u16,
}

impl Default for Material {
    /// Generates a default, flat, white material.
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            shader: ShaderMat::DEFAULT_SHADER,
            diffuse: 0,
            diffuse_mul: [255, 255, 255, 255],
            opacity: 0,
        }
    }
}
