#[cfg(feature = "vulkan")]
use ash::vk;
use cgmath::{Point2, Point3, Vector3 as Vec3};

/// A three dimensional point in space, used as a vertex for a triangle.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    /// The vertex position in 3D space.
    pub vv: Point3<f32>,
    /// The vertex normal.
    pub vn: Vec3<f32>,
    /// The vertex texture coordinate.
    pub vt: Point2<f32>,
}

impl Vertex {
    #[cfg(feature = "vulkan")]
    /// Returns the vertex bindings descriptions to be used in Vulkan pipelines.
    pub const fn binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    #[cfg(feature = "vulkan")]
    /// Returns the vertex attribute descriptions to be used in Vulkan pipelines.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        use memoffset::offset_of;
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, vv) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, vn) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, vt) as u32,
            },
        ]
    }
}
