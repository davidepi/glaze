use cgmath::Matrix4;

/// A collection of triangles in 3D space, having the same material.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// Unique ID for the current mesh.
    pub id: u16,
    /// Indices for the triangles vertices. Each triplet of indices represents a triangle.
    /// The indices are relative to a [Vertex][crate::Vertex] buffer, not contained in the Mesh.
    pub indices: Vec<u32>,
    /// Index of the mesh [Material][crate::Material].
    /// The material is not contained in the Mesh itself.
    pub material: u16,
}

/// A transformation applied to a [Mesh]
///
/// [Mesh]es by default are not placed in the space and requires a transformation, in form of
/// [cgmath::Matrix4] to be applied to them. This class links together the mesh and the
/// transformation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MeshInstance {
    pub mesh_id: u16,
    pub transform_id: u16,
}

/// A 4x4 column-major transformation matrix
pub type Transform = Matrix4<f32>;
