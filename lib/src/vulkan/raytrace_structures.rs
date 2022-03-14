use cgmath::Vector2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RTFrameData {
    pub seed: u32,
    pub lights_no: u32,
    pub pixel_offset: Vector2<f32>,
    pub scene_radius: f32,
    pub exposure: f32,
}

impl Default for RTFrameData {
    fn default() -> Self {
        Self {
            seed: 0,
            lights_no: 0,
            scene_radius: 0.0,
            pixel_offset: Vector2::new(0.0, 0.0),
            exposure: 1.0,
        }
    }
}

/// This is the struct passed to the GPU that represents a mesh instance.
/// Used to retrieve the mesh attributes in a raytracing context.
/// Unlike MeshInstance and VulkanMesh it MUST be indexed by its position in the array.
/// The index must correspond to the MeshInstance index passed to the acceleration structure.
#[repr(C)]
pub struct RTInstance {
    pub index_offset: u32,
    pub index_count: u32,
    pub material_id: u32, // due to how std430 works
}

#[repr(C, align(16))]
#[derive(Debug, Default, Copy, Clone)]
pub struct RTMaterial {
    pub diffuse_mul: [f32; 4],
    pub ior0: [f32; 4],
    pub ior1: [f32; 4],
    pub ior2: [f32; 4],
    pub ior3: [f32; 4],
    pub metal_fresnel0: [f32; 4],
    pub metal_fresnel1: [f32; 4],
    pub metal_fresnel2: [f32; 4],
    pub metal_fresnel3: [f32; 4],
    pub diffuse: u32,
    pub roughness: u32,
    pub metalness: u32,
    pub opacity: u32,
    pub normal: u32,
    // callable shader index for the bsdf_value
    pub bsdf_index: u32,
    pub roughness_mul: f32,
    pub metalness_mul: f32,
    pub anisotropy: f32,
    pub ior_dielectric: f32,
    pub is_specular: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
pub struct RTLight {
    pub color0: [f32; 4],
    pub color1: [f32; 4],
    pub color2: [f32; 4],
    pub color3: [f32; 4],
    pub pos: [f32; 4],
    pub dir: [f32; 4],
    pub shader: u32,
}
