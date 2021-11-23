use crate::ShaderMat;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Material {
    pub name: String,
    pub shader: ShaderMat,
    pub diffuse: Option<u16>,
}
