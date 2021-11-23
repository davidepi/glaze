#[cfg(feature = "vulkan")]
use crate::vulkan::PipelineBuilder;
use std::error::Error;

#[macro_export]
macro_rules! include_shader {
    ($shader_name : expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $shader_name, ".spv"))
    };
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ShaderMat {
    Test = 0,
}

impl ShaderMat {
    pub const DEFAULT_SHADER: Self = ShaderMat::Test;

    pub fn name(&self) -> &'static str {
        match self {
            ShaderMat::Test => "Test",
        }
    }

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

    // used to iterate all the possible shaders
    pub fn all_values() -> [ShaderMat; 1] {
        [ShaderMat::Test]
    }

    #[cfg(feature = "vulkan")]
    pub fn build_pipeline(&self) -> PipelineBuilder {
        match self {
            ShaderMat::Test => test_pipeline(),
        }
    }
}

impl From<u8> for ShaderMat {
    fn from(num: u8) -> Self {
        match num {
            0 => ShaderMat::Test,
            _ => Self::DEFAULT_SHADER, // use default
        }
    }
}

impl From<ShaderMat> for u8 {
    fn from(shader: ShaderMat) -> Self {
        match shader {
            ShaderMat::Test => 0,
        }
    }
}

#[cfg(feature = "vulkan")]
fn test_pipeline() -> PipelineBuilder {
    let mut pipeline = PipelineBuilder::default();
    let vertex_shader = include_shader!("test.vert");
    let fragment_shader = include_shader!("test.frag");
    pipeline.push_shader(vertex_shader, "main", ash::vk::ShaderStageFlags::VERTEX);
    pipeline.push_shader(fragment_shader, "main", ash::vk::ShaderStageFlags::FRAGMENT);
    pipeline
}
