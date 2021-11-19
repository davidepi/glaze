#[cfg(feature = "vulkan")]
use crate::vulkan::PipelineBuilder;
use std::error::Error;
#[cfg(test)]
use std::ops::Range;

#[macro_export]
macro_rules! include_shader {
    ($shader_name : expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $shader_name, ".spv"))
    };
}

pub enum ShaderMat {
    Test,
}

impl ShaderMat {
    pub fn from_id(id: u8) -> Result<Self, Box<dyn Error>> {
        match id {
            0 => Ok(ShaderMat::Test),
            _ => Err(format!("Unknown shader id: {}", id).into()),
        }
    }

    pub fn id(&self) -> u8 {
        match self {
            ShaderMat::Test => 0,
        }
    }

    // used in tests to generate random shaders
    #[cfg(test)]
    pub(crate) fn ids_range() -> Range<u8> {
        0..1
    }

    #[cfg(feature = "vulkan")]
    pub fn build_pipeline(&self) -> PipelineBuilder {
        let mut pipeline = PipelineBuilder::default();
        let vertex_shader = include_shader!("test.vert");
        let fragment_shader = include_shader!("test.frag");
        pipeline.push_shader(vertex_shader, "main", ash::vk::ShaderStageFlags::VERTEX);
        pipeline.push_shader(fragment_shader, "main", ash::vk::ShaderStageFlags::FRAGMENT);
        pipeline
    }
}
