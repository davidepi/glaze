#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Material {
    pub shader_id: u8,
    pub diffuse: Option<u16>,
}
