#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Material {
    pub name: String,
    pub shader_id: u8,
    pub diffuse: Option<u16>,
}
