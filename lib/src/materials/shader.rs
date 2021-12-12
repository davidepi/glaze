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
#[allow(non_camel_case_types)]
pub enum ShaderMat {
    /// Flat shading.
    /// Light does not affect the material and the diffuse color is shown unaltered.
    FLAT = 0,
    /// Flat shading.
    /// Internal version, used for two-sided polygons and polygons with opacity maps.
    /// This version is automatically assigned by the engine and **SHOULD NOT** be used.
    INTERNAL_FLAT_2SIDED,
}

impl ShaderMat {
    pub const DEFAULT_SHADER: Self = ShaderMat::FLAT;

    /// Returns each shader's name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            ShaderMat::FLAT | ShaderMat::INTERNAL_FLAT_2SIDED => "Flat",
        }
    }

    /// Returns an unique number identifiying the shader.
    /// If the shader corresponding to the number does not exist, an error is raised.
    /// Using an internal ID wil
    pub fn from_id(id: u8) -> Result<Self, Box<dyn Error>> {
        match id {
            0 => Ok(ShaderMat::FLAT),
            _ => Err(format!("Unknown shader id: {}", id).into()),
        }
    }

    /// Returns the id corresponding to the shader.
    pub const fn id(&self) -> u8 {
        match self {
            ShaderMat::FLAT => 0,
            _ => panic!("Internal shaders have no ID assigned"),
        }
    }

    /// Iterates all the possible assignable shaders.
    /// Shaders used internally by the engine are skipped.
    pub fn all_values() -> [ShaderMat; 1] {
        [ShaderMat::FLAT]
    }

    /// Returns the a builder useful to create the pipeline for the shader.
    #[cfg(feature = "vulkan")]
    pub fn build_pipeline(&self) -> PipelineBuilder {
        match self {
            ShaderMat::FLAT => pipelines::flat_pipeline(),
            ShaderMat::INTERNAL_FLAT_2SIDED => pipelines::flat_2s_pipeline(),
        }
    }

    /// Consumes a shader and returns the its internal version supporting two-sided polygons.
    pub fn two_sided(self) -> Self {
        match self {
            ShaderMat::FLAT => ShaderMat::INTERNAL_FLAT_2SIDED,
            ShaderMat::INTERNAL_FLAT_2SIDED => ShaderMat::INTERNAL_FLAT_2SIDED,
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
            0 => ShaderMat::FLAT,
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
mod pipelines {
    use crate::PipelineBuilder;
    use ash::vk;

    /// pipeline for the flat shader.
    pub(super) fn flat_pipeline() -> PipelineBuilder {
        let mut pipeline = PipelineBuilder::default();
        let vertex_shader = include_shader!("flat.vert");
        let fragment_shader = include_shader!("flat.frag");
        pipeline.push_shader(vertex_shader, "main", ash::vk::ShaderStageFlags::VERTEX);
        pipeline.push_shader(fragment_shader, "main", ash::vk::ShaderStageFlags::FRAGMENT);
        pipeline
    }

    /// pipeline for the flat shader, used for two-sided polygons.
    pub(super) fn flat_2s_pipeline() -> PipelineBuilder {
        let mut pipeline = PipelineBuilder::default();
        let vertex_shader = include_shader!("flat.vert");
        let fragment_shader = include_shader!("flat_twosided.frag");
        pipeline.push_shader(vertex_shader, "main", ash::vk::ShaderStageFlags::VERTEX);
        pipeline.push_shader(fragment_shader, "main", ash::vk::ShaderStageFlags::FRAGMENT);
        pipeline.rasterizer.cull_mode = vk::CullModeFlags::NONE;
        pipeline
    }
}
