use super::Camera;
use super::Mesh;
use super::Vertex;

pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub meshes: Vec<Mesh>,
    pub cameras: Vec<Camera>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            meshes: Vec::new(),
            cameras: Vec::new(),
        }
    }
}
