use crate::ShaderMat;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Material {
    pub name: String,
    pub shader: ShaderMat,
    pub diffuse: Option<u16>,
    pub diffuse_mul: [u8; 3],
    pub opacity: Option<u16>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            shader: ShaderMat::DEFAULT_SHADER,
            diffuse: None,
            diffuse_mul: [255, 255, 255],
            opacity: None,
        }
    }
}
