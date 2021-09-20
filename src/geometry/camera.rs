use cgmath::Point3;
use cgmath::Vector3 as Vec3;

#[derive(Debug, Clone, PartialEq)]
pub enum Camera {
    Perspective(PerspectiveCam),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerspectiveCam {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vec3<f32>,
    pub fov: f32,
}
