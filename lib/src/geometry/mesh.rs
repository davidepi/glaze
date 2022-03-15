#[cfg(feature = "vulkan")]
use cgmath::Matrix;
use cgmath::{Matrix4, SquareMatrix};

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
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct Transform(Matrix4<f32>);

impl Transform {
    /// Creates an identity transformation.
    pub fn identity() -> Self {
        Self(Matrix4::identity())
    }

    /// Converts the transformation to a sequence of bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let vals: &[f32; 16] = self.0.as_ref();
        let mut retval = [0; 64];
        let mut index = 0;
        for val in vals {
            let bytes = f32::to_le_bytes(*val);
            retval[index] = bytes[0];
            retval[index + 1] = bytes[1];
            retval[index + 2] = bytes[2];
            retval[index + 3] = bytes[3];
            index += 4;
        }
        retval
    }

    /// Reconstructs the transformation from a sequence of bytes.
    pub fn from_bytes(data: [u8; 64]) -> Self {
        let matrix = Matrix4::new(
            f32::from_le_bytes(data[0..4].try_into().unwrap()),
            f32::from_le_bytes(data[4..8].try_into().unwrap()),
            f32::from_le_bytes(data[8..12].try_into().unwrap()),
            f32::from_le_bytes(data[12..16].try_into().unwrap()),
            f32::from_le_bytes(data[16..20].try_into().unwrap()),
            f32::from_le_bytes(data[20..24].try_into().unwrap()),
            f32::from_le_bytes(data[24..28].try_into().unwrap()),
            f32::from_le_bytes(data[28..32].try_into().unwrap()),
            f32::from_le_bytes(data[32..36].try_into().unwrap()),
            f32::from_le_bytes(data[36..40].try_into().unwrap()),
            f32::from_le_bytes(data[40..44].try_into().unwrap()),
            f32::from_le_bytes(data[44..48].try_into().unwrap()),
            f32::from_le_bytes(data[48..52].try_into().unwrap()),
            f32::from_le_bytes(data[52..56].try_into().unwrap()),
            f32::from_le_bytes(data[56..60].try_into().unwrap()),
            f32::from_le_bytes(data[60..64].try_into().unwrap()),
        );
        Self(matrix)
    }

    #[cfg(feature = "vulkan")]
    pub fn to_vulkan_transform(&self) -> ash::vk::TransformMatrixKHR {
        let transpose = self.0.transpose();
        let floats: [f32; 16] = *transpose.as_ref();
        let matrix: [f32; 12] = floats[..12].try_into().unwrap();
        ash::vk::TransformMatrixKHR { matrix }
    }

    pub fn inner(&self) -> &Matrix4<f32> {
        &self.0
    }
}

impl Default for Transform {
    fn default() -> Self {
        Transform::identity()
    }
}

impl From<Matrix4<f32>> for Transform {
    fn from(mat: Matrix4<f32>) -> Self {
        Self(mat)
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use crate::Transform;
    use cgmath::Matrix4;

    #[test]
    fn cgmath_vktransform_memory_layout() {
        let matrix = Matrix4::new(
            0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0,
        );
        let transform = Transform::from(matrix).to_vulkan_transform();
        assert_eq!(
            transform.matrix,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
    }
}
