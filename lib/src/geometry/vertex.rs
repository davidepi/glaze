#[cfg(feature = "vulkan")]
use ash::vk;
use cgmath::{Vector2 as Vec2, Vector3 as Vec3};

#[repr(C, packed)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex {
    pub vv: Vec3<f32>,
    pub vn: Vec3<f32>,
    pub vt: Vec2<f32>,
}

impl Vertex {
    #[cfg(feature = "vulkan")]
    pub const fn binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    #[cfg(feature = "vulkan")]
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
