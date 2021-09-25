use cgmath::{Vector2 as Vec2, Vector3 as Vec3};

#[repr(C, packed)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex {
    pub vv: Vec3<f32>,
    pub vn: Vec3<f32>,
    pub vt: Vec2<f32>,
}
