use super::{Camera, Mesh, Vertex};
use crate::{Library, Material, Texture};

/// Struct representing a renderable scene.
pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub meshes: Vec<Mesh>,
    pub cameras: Vec<Camera>,
    pub textures: Library<Texture>,
    pub materials: Library<Material>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            meshes: Vec::new(),
            cameras: Vec::new(),
            textures: Library::new(),
            materials: Library::new(),
        }
    }
}
