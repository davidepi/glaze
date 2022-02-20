#[cfg(feature = "vulkan")]
use crate::geometry::{SBT_LIGHT_STRIDE, SBT_LIGHT_TYPES};
#[cfg(feature = "vulkan")]
use crate::materials::{SBT_MATERIAL_STRIDE, SBT_MATERIAL_TYPES};
#[cfg(feature = "vulkan")]
use crate::vulkan::PipelineBuilder;
use std::error::Error;

/// Macro used to include the shader contained inside the /shader directory as a `[u8; _]`.
///
/// Probably will not work outside this crate.
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
    /// In raytrace mode, this has the same effect of "Lambert"
    FLAT = 0,
    /// Lambert BRDF for raytracing.
    LAMBERT = 1,
    /// Mirror like material fo raytracing.
    MIRROR = 2,
    /// Flat shading.
    /// Internal version, used for two-sided polygons and polygons with opacity maps.
    /// This version is automatically assigned by the engine and **SHOULD NOT** be used.
    INTERNAL_FLAT_2SIDED,
}

impl ShaderMat {
    pub const DEFAULT_SHADER: Self = ShaderMat::LAMBERT;

    /// Returns each shader's name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            ShaderMat::FLAT | ShaderMat::INTERNAL_FLAT_2SIDED => "Flat",
            ShaderMat::LAMBERT => "Lambert",
            ShaderMat::MIRROR => "Mirror",
        }
    }

    /// Returns an unique number identifiying the shader.
    /// If the shader corresponding to the number does not exist, an error is raised.
    /// Using an internal ID wil
    pub fn from_id(id: u8) -> Result<Self, Box<dyn Error>> {
        match id {
            0 => Ok(ShaderMat::FLAT),
            1 => Ok(ShaderMat::LAMBERT),
            2 => Ok(ShaderMat::MIRROR),
            _ => Err(format!("Unknown shader id: {}", id).into()),
        }
    }

    /// Returns the id corresponding to the shader.
    pub const fn id(&self) -> u8 {
        match self {
            ShaderMat::FLAT => 0,
            ShaderMat::LAMBERT => 1,
            ShaderMat::MIRROR => 2,
            _ => panic!("Internal shaders have no ID assigned"),
        }
    }

    /// Iterates all the possible assignable shaders.
    /// Shaders used internally by the engine are skipped.
    pub fn all_values() -> [ShaderMat; 3] {
        [ShaderMat::FLAT, ShaderMat::LAMBERT, ShaderMat::MIRROR]
    }

    /// Returns true if the shader exhibit metallicness.
    pub fn is_metallic(&self) -> bool {
        match self {
            ShaderMat::FLAT => false,
            ShaderMat::LAMBERT => false,
            ShaderMat::MIRROR => true,
            ShaderMat::INTERNAL_FLAT_2SIDED => false,
        }
    }

    /// Returns a builder useful to create the pipeline for the shader.
    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn build_viewport_pipeline(&self) -> PipelineBuilder {
        match self {
            ShaderMat::INTERNAL_FLAT_2SIDED => pipelines::flat_2s_pipeline(),
            _ => pipelines::flat_pipeline(),
        }
    }

    /// Consumes a shader and returns its internal version supporting two-sided polygons.
    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn two_sided_viewport(self) -> Self {
        ShaderMat::INTERNAL_FLAT_2SIDED
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn callable_shaders() -> [Vec<u8>; SBT_MATERIAL_TYPES * SBT_MATERIAL_STRIDE] {
        [
            include_shader!("mat_lambert_value.rcall").to_vec(),
            include_shader!("mat_lambert_sample_value.rcall").to_vec(),
            include_shader!("mat_lambert_pdf.rcall").to_vec(),
            include_shader!("mat_mirror_value.rcall").to_vec(),
            include_shader!("mat_mirror_sample_value.rcall").to_vec(),
            include_shader!("mat_mirror_pdf.rcall").to_vec(),
        ]
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn sbt_callable_index(&self) -> u32 {
        let base_index = SBT_LIGHT_TYPES * SBT_LIGHT_STRIDE; // lights before mats
        let shader_index = match self {
            ShaderMat::FLAT | ShaderMat::LAMBERT => 0,
            ShaderMat::MIRROR => 1,
            ShaderMat::INTERNAL_FLAT_2SIDED => panic!("This shader should not appear in the sbt"),
        };
        (base_index + shader_index * SBT_MATERIAL_STRIDE) as u32
    }
}

impl Default for ShaderMat {
    fn default() -> Self {
        ShaderMat::DEFAULT_SHADER
    }
}

impl From<u8> for ShaderMat {
    fn from(num: u8) -> Self {
        if let Ok(shader) = ShaderMat::from_id(num) {
            shader
        } else {
            Self::DEFAULT_SHADER
        }
    }
}

impl From<ShaderMat> for u8 {
    fn from(shader: ShaderMat) -> Self {
        shader.id()
    }
}

#[cfg(feature = "vulkan-interactive")]
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
