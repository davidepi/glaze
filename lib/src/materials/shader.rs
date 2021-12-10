#[cfg(feature = "vulkan")]
use crate::vulkan::PipelineBuilder;
use std::error::Error;

#[macro_export]
macro_rules! include_shader {
    ($shader_name : expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $shader_name, ".spv"))
    };
}

/// A shader determining how the light interacts with the material.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum ShaderMat {
    /// Flat shading.
    /// Light does not affect the material and the diffuse color is shown unaltered.
    Flat = 0,
}

impl ShaderMat {
    pub const DEFAULT_SHADER: Self = ShaderMat::Flat;

    /// Returns each shader's name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            ShaderMat::Flat => "Flat",
        }
    }

    /// Returns an unique number identifiying the shader.
    /// If the shader corresponding to the number does not exist, an error is raised.
    pub fn from_id(id: u8) -> Result<Self, Box<dyn Error>> {
        match id {
            0 => Ok(ShaderMat::Flat),
            _ => Err(format!("Unknown shader id: {}", id).into()),
        }
    }

    /// Returns the id corresponding to the shader.
    pub const fn id(&self) -> u8 {
        match self {
            ShaderMat::Flat => 0,
        }
    }

    /// Iterates all the possible assignable shaders.
    /// Shaders used internally by the engine are skipped.
    pub fn all_values() -> [ShaderMat; 1] {
        [ShaderMat::Flat]
    }

    /// Returns the a builder useful to create the pipeline for the shader.
    #[cfg(feature = "vulkan")]
    pub fn build_pipeline(&self) -> PipelineBuilder {
        match self {
            ShaderMat::Flat => flat_pipeline(),
        }
    }
}

impl Default for ShaderMat {
    fn default() -> Self {
        ShaderMat::DEFAULT_SHADER
    }
}

impl From<u8> for ShaderMat {
    fn from(num: u8) -> Self {
        match num {
            0 => ShaderMat::Flat,
            _ => Self::DEFAULT_SHADER, // use default shader
        }
    }
}

impl From<ShaderMat> for u8 {
    fn from(shader: ShaderMat) -> Self {
        shader.id()
    }
}

#[cfg(feature = "vulkan")]
/// pipeline for the flat shader.
fn flat_pipeline() -> PipelineBuilder {
    let mut pipeline = PipelineBuilder::default();
    let vertex_shader = include_shader!("flat.vert");
    let fragment_shader = include_shader!("flat.frag");
    pipeline.push_shader(vertex_shader, "main", ash::vk::ShaderStageFlags::VERTEX);
    pipeline.push_shader(fragment_shader, "main", ash::vk::ShaderStageFlags::FRAGMENT);
    pipeline
}
