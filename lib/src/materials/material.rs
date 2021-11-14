use super::PipelineBuilder;
use ash::vk;
use std::error::Error;
use std::ops::Range;

#[macro_export]
macro_rules! include_shader {
    ($shader_name : expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $shader_name, ".spv"))
    };
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Material {
    pub shader_id: u8,
    pub diffuse: Option<u16>,
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
    pub(crate) fn ids_range() -> Range<u8> {
        0..1
    }

    pub fn build_pipeline(&self) -> PipelineBuilder {
        let mut pipeline = PipelineBuilder::default();
        let vertex_shader = include_shader!("test.vert");
        let fragment_shader = include_shader!("test.frag");
        pipeline.push_shader(vertex_shader, "main", vk::ShaderStageFlags::VERTEX);
        pipeline.push_shader(fragment_shader, "main", vk::ShaderStageFlags::FRAGMENT);
        pipeline
    }
}
