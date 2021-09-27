use super::{Camera, Mesh, Vertex};
use crate::materials::{Library, Texture};

/// Struct representing a renderable scene.
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
