use crate::Spectrum;
use cgmath::Matrix4;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RTFrameData {
    pub seed: u32,
    pub lights_no: u32,
    pub pixel_offset: [f32; 2],
    pub scene_radius: f32,
    pub exposure: f32,
    pub scene_size: [f32; 2],
    pub scene_centre: [f32; 4],
    pub camera_persp: bool,
}

impl Default for RTFrameData {
    fn default() -> Self {
        Self {
            seed: 0,
            lights_no: 0,
            scene_radius: 0.0,
            exposure: 1.0,
            pixel_offset: [0.0; 2],
            scene_size: [0.0; 2],
            scene_centre: [0.0; 4],
            camera_persp: true,
        }
    }
}

/// This is the struct passed to the GPU that represents a mesh instance.
/// Used to retrieve the mesh attributes in a raytracing context.
/// Unlike MeshInstance and VulkanMesh it MUST be indexed by its position in the array.
/// The index must correspond to the MeshInstance index passed to the acceleration structure.
#[repr(C, align(4))]
pub struct RTInstance {
    pub index_offset: u32,
    pub index_count: u32,
    pub material_id: u32, // due to how std430 works
    pub transform_id: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Default, Copy, Clone)]
pub struct RTMaterial {
    pub diffuse_mul: [f32; 4],
    pub emissive_col: [f32; 4],
    pub metal_ior: Spectrum,
    pub metal_fresnel: Spectrum,
    pub diffuse: u32,
    pub roughness: u32,
    pub metalness: u32,
    pub opacity: u32,
    pub normal: u32,
    // callable shader index for the bsdf_value.
    pub bsdf_index: u32,
    pub roughness_mul: f32,
    pub metalness_mul: f32,
    pub anisotropy: f32,
    pub ior_dielectric: f32,
    pub is_specular: u32,
    pub is_emissive: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
pub struct RTLight {
    pub color: Spectrum,
    pub pos: [f32; 4],
    pub dir: [f32; 4],
    pub shader: u32,
    pub instance_id: u32,
    pub intensity: f32,
    pub delta: bool,
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
pub struct RTSky {
    pub obj2world: Matrix4<f32>,
    pub world2obj: Matrix4<f32>,
    pub tex_id: u32,
}

pub const PT_STEPS: usize = 6;

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
pub struct PTLastVertex {
    pub importance: Spectrum,
    pub wi: [f32; 4],  // wi.z indicates if the last bounce was specular
    pub hit: [f32; 4], // hit.z indicates the bounce number
}
