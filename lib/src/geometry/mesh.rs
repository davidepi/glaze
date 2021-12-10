use cgmath::Matrix4;

/// A collection of triangles in 3D space, having the same material.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// Indices for the triangles vertices. Each triplet of indices represents a triangle.
    /// The indices are relative to a [Vertex] buffer, not contained in the Mesh.
    pub indices: Vec<u32>,
    /// Index of the mesh [Material].
    /// The material is not contained in the Mesh itself.
    pub material: u16,
    /// Transformation matrices of the current mesh.
    /// If this vector is empty, a single identity matrix is implied.
    pub instances: Vec<Matrix4<f32>>,
}
