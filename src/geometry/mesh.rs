use cgmath::Matrix4;

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub indices: Vec<u32>,
    pub material: u16,
    pub instances: Vec<Matrix4<f32>>,
}
