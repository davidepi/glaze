use super::Camera;
use super::Mesh;
use super::Vertex;
use crate::materials::Library;
use crate::materials::Texture;

pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub meshes: Vec<Mesh>,
    pub cameras: Vec<Camera>,
    pub textures: Library<Texture>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            meshes: Vec::new(),
            cameras: Vec::new(),
            textures: Library::new(),
        }
    }
}
