use super::acceleration::{SceneAS, SceneASBuilder};
use super::cmd::CommandManager;
#[cfg(feature = "vulkan-interactive")]
use super::descriptor::{Descriptor, DescriptorSetManager};
use super::device::Device;
use super::instance::Instance;
use super::memory::{AllocatedBuffer, MemoryManager};
use super::pipeline::build_compute_pipeline;
#[cfg(feature = "vulkan-interactive")]
use super::pipeline::Pipeline;
use super::raytrace_structures::{RTInstance, RTLight, RTMaterial};
use super::UnfinishedExecutions;
use crate::geometry::SkyLight;
#[cfg(feature = "vulkan-interactive")]
use crate::materials::{TextureFormat, TextureLoaded};
use crate::{
    include_shader, Camera, Light, LightType, Material, Mesh, MeshInstance, Meta, Metal,
    ParsedScene, PipelineBuilder, RayTraceInstance, Spectrum, Vertex,
};
#[cfg(feature = "vulkan-interactive")]
use crate::{PresentInstance, ShaderMat, Texture, Transform};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk;
use cgmath::{InnerSpace, Vector3 as Vec3};
#[cfg(feature = "vulkan-interactive")]
use fnv::{FnvBuildHasher, FnvHashMap};
use gpu_allocator::MemoryLocation;
#[cfg(feature = "vulkan-interactive")]
use std::collections::hash_map::Entry;
use std::f32::consts::PI;
#[cfg(feature = "vulkan-interactive")]
use std::ffi::c_void;
#[cfg(feature = "vulkan-interactive")]
use std::ptr;
use std::sync::{Arc, RwLock};

/// A scene optimized to be rendered using this crates vulkan implementation.
#[cfg(feature = "vulkan-interactive")]
pub struct VulkanScene {
    /// The scene on disk.
    file: Box<dyn ParsedScene + Send>,
    /// Not used in this scene, but required in case of updates.
    pub(super) meta: Meta,
    /// The camera for the current scene.
    pub current_cam: Camera,
    /// The buffer containing all vertices for the current scene.
    pub(super) vertex_buffer: Arc<AllocatedBuffer>,
    /// The buffer containing all indices for the current scene.
    pub(super) index_buffer: Arc<AllocatedBuffer>,
    /// Buffer used during a single material update, as a transfer buffer to the GPU
    update_buffer: AllocatedBuffer,
    /// GPU buffer containing all parameters for all materials in the scene
    params_buffer: AllocatedBuffer,
    /// All the meshes in the scene. Not guaranteed to be ordered by ID.
    pub(super) meshes: Vec<VulkanMesh>,
    /// Generic sampler used for all textures
    sampler: vk::Sampler,
    /// Manages descriptors in the current scene.
    dm: DescriptorSetManager,
    /// All materials descriptors in the scene.
    pub(super) materials_desc: Vec<(ShaderMat, Descriptor)>,
    /// All the materials in the scene.
    materials: Vec<Material>,
    /// Map of all shaders in the scene with their pipeline.
    pub(super) pipelines: FnvHashMap<ShaderMat, Pipeline>,
    /// All textures in the scene.
    pub(super) textures: Arc<RwLock<Vec<TextureLoaded>>>,
    /// Raw textures.
    raw_textures: Vec<Texture>,
    /// True if textures has been changed (so they need to be saved again, this takes time)
    edited_textures: bool,
    /// All the transforms in the scene.
    transforms: Arc<AllocatedBuffer>,
    /// All the instances in the scene in form (Mesh ID, Vec<Transform ID>).
    pub(super) instances: FnvHashMap<u16, Vec<u16>>,
    /// Scene level descriptor set
    pub(super) scene_desc: Descriptor,
    /// skylighy used for the skydome.
    skylight: Option<(SkyLight, Option<(Descriptor, Pipeline)>)>,
    /// All the lights in the scene.
    lights: Vec<Light>,
    /// Instance storing this scene.
    instance: Arc<PresentInstance>,
}

/// A mesh optimized to be rendered using this crates renderer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VulkanMesh {
    /// Original ID of the [Mesh]
    pub mesh_id: u16,
    /// Offset of the mesh in the scene index buffer.
    pub index_offset: u32,
    /// Number of indices of this mesh in the index buffer.
    pub index_count: u32,
    /// Maximum index value that appears for this mesh.
    pub max_index: u32,
    /// Material id of this mesh.
    pub material: u16,
}

#[cfg(feature = "vulkan-interactive")]
impl VulkanScene {
    /// Converts a parsed scene into a vulkan scene.
    ///
    /// `wchan` is used to send feedbacks about the current loading status.
    pub fn new(instance: Arc<PresentInstance>, parsed: Box<dyn ParsedScene + Send>) -> Self {
        let device = instance.device();
        let mm = instance.allocator();
        let with_raytrace = instance.supports_raytrace();
        let mut unf = UnfinishedExecutions::new(device);
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 5);
        let mut gcmdm = CommandManager::new(device.logical_clone(), device.graphic_queue().idx, 5);
        let avg_desc = [
            (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
            (vk::DescriptorType::STORAGE_BUFFER, 1.0),
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1.5),
        ];
        let mut dm = DescriptorSetManager::new(
            device.logical_clone(),
            &avg_desc,
            instance.desc_layout_cache(),
        );
        let vertex_buffer = load_vertices_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed.vertices().unwrap_or_default(),
            with_raytrace,
        );
        let transforms = Arc::new(load_transforms_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed
                .transforms()
                .unwrap_or_else(|_| vec![Transform::default()]),
        ));
        let (mut meshes, index_buffer) = load_indices_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed.meshes().unwrap_or_default(),
            with_raytrace,
        );
        let instances = instances_to_map(&parsed.instances().unwrap_or_default());
        let sampler = create_sampler(device);
        let raw_textures = parsed
            .textures()
            .unwrap_or_else(|_| vec![Texture::default()]);
        let textures = raw_textures
            .iter()
            .map(|tex| load_texture_to_gpu(Arc::clone(&instance), mm, &mut gcmdm, &mut unf, tex))
            .collect::<Vec<_>>();
        let materials = parsed
            .materials()
            .unwrap_or_else(|_| vec![Material::default()]);
        let mut lights = parsed.lights().unwrap_or_default();
        let params_buffer = load_materials_parameters(device, &materials, mm, &mut tcmdm, &mut unf);
        unf.wait_completion();
        let materials_desc = materials
            .iter()
            .enumerate()
            .map(|(id, mat)| {
                let (shader, desc) = build_mat_desc_set(
                    device,
                    &textures,
                    &params_buffer,
                    sampler,
                    id as u16,
                    mat,
                    &mut dm,
                );
                (shader, desc)
            })
            .collect::<Vec<_>>();
        let current_cam = parsed
            .cameras()
            .unwrap_or_default()
            .pop()
            .unwrap_or_default();
        let pipelines = FnvHashMap::default();
        let update_buffer = mm.create_buffer(
            "Material update transfer buffer",
            std::mem::size_of::<MaterialParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        sort_meshes(&mut meshes, &materials_desc);
        let skylight = lights.iter().find_map(|l| match l {
            Light::Sky(s) => Some((*s, None)),
            _ => None,
        });
        lights = lights
            .into_iter()
            .filter(|l| l.ltype() != LightType::SKY)
            .collect();
        let meta = parsed.meta().unwrap_or_default();
        let scene_desc = build_realtime_descriptor(&mut dm, &transforms);
        VulkanScene {
            file: parsed,
            meta,
            current_cam,
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            update_buffer,
            params_buffer,
            meshes,
            sampler,
            dm,
            materials_desc,
            materials,
            pipelines,
            textures: Arc::new(RwLock::new(textures)),
            raw_textures,
            edited_textures: false,
            transforms,
            instances,
            scene_desc,
            lights,
            instance,
            skylight,
        }
    }

    /// Updates (changes) a single material in the scene.
    pub(super) fn update_material(
        &mut self,
        mat_id: u16,
        new: Material,
        tcmdm: &mut CommandManager, /* graphic because the realtime renderer does not have a tranfer */
        unf: &mut UnfinishedExecutions,
        rpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
        render_size: vk::Extent2D,
    ) {
        let device = self.instance.device();
        // setup the new descriptor
        let params = MaterialParams::from(&new);
        let mapped = self
            .update_buffer
            .allocation()
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(&params, mapped, 1);
        }
        let align = device
            .physical()
            .properties
            .limits
            .min_uniform_buffer_offset_alignment;
        let padding = padding(PARAMS_SIZE, align);
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: (PARAMS_SIZE + padding) * mat_id as u64,
            size: PARAMS_SIZE,
        };
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_copy_buffer(
                    cmd,
                    self.update_buffer.buffer,
                    self.params_buffer.buffer,
                    &[copy_region],
                );
            }
        };
        let cmd = tcmdm.get_cmd_buffer();
        let queue = device.transfer_queue();
        let fence = device.immediate_execute(cmd, queue, command);
        unf.add_fence(fence); // the buffer is always stored in self
        let (new_shader, new_desc) = build_mat_desc_set(
            device,
            &self.textures.read().unwrap(),
            &self.params_buffer,
            self.sampler,
            mat_id,
            &new,
            &mut self.dm,
        );
        // build the new shader if not existing
        self.pipelines.entry(new_shader).or_insert_with(|| {
            new.shader.build_viewport_pipeline().build(
                device.logical_clone(),
                rpass,
                render_size,
                &[frame_desc_layout, new_desc.layout, self.scene_desc.layout],
            )
        });
        // keep the list of lights aligned
        let old = &self.materials[mat_id as usize];
        if new.emissive_col.is_some() || old.emissive_col.is_some() {
            if new.emissive_col.is_some() && !old.emissive_col.is_some() {
                // lights should be added
                self.lights
                    .push(Light::new_area(new.name.clone(), mat_id as u32, 1.0));
            } else if !new.emissive_col.is_some() && old.emissive_col.is_some() {
                // light should be removed
                self.lights
                    .retain(|x| x.ltype() != LightType::AREA || x.material_id() != mat_id as u32);
            }
            // no need to update lights buffer because FOR NOW lights are not
            // rendered in the realtime preview
        }
        // replace the old material
        self.materials[mat_id as usize] = new;
        self.materials_desc[mat_id as usize] = (new_shader, new_desc);
        // sort the meshes to minimize bindings
        sort_meshes(&mut self.meshes, &self.materials_desc);
    }

    /// Initializes the scene's pipelines.
    pub(super) fn init_pipelines(
        &mut self,
        render_size: vk::Extent2D,
        renderpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
    ) {
        self.pipelines = FnvHashMap::default();
        for (shader, desc) in &self.materials_desc {
            let device = self.instance.device().logical_clone();
            self.pipelines.entry(*shader).or_insert_with(|| {
                shader.build_viewport_pipeline().build(
                    device,
                    renderpass,
                    render_size,
                    &[
                        frame_desc_layout,
                        desc.layout,
                        self.scene_desc.layout, // pick the first per-object transform
                    ],
                )
            });
        }
        if let Some((sky, data)) = &mut self.skylight {
            let device = self.instance.device().logical_clone();
            *data = Some(build_skydome(
                device,
                render_size,
                &mut self.dm,
                *sky,
                &self.textures.read().unwrap(),
                self.sampler,
                renderpass,
            ));
        }
    }

    /// Destroys the scene's pipelines.
    pub(super) fn deinit_pipelines(&mut self) {
        self.pipelines.clear();
        if let Some((_, data)) = &mut self.skylight {
            *data = None;
        }
    }

    /// Returns a material in the scene, given its ID.
    /// Returns None if the material does not exist.
    pub fn single_material(&self, id: u16) -> Option<&Material> {
        self.materials.get(id as usize)
    }

    /// Returns all the materials in the scene.
    /// The index of the material correspond to its ID.
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    /// Returns all the textures in the scene.
    /// The position in the array is the texture ID.
    pub fn textures(&self) -> &[Texture] {
        &self.raw_textures
    }

    /// Returns a texture in the scene, given its ID.
    /// Returns None if the texture does not exist.
    pub fn single_texture(&self, id: u16) -> Option<&Texture> {
        self.raw_textures.get(id as usize)
    }

    /// Returns the light used as skydome.
    pub fn skydome(&self) -> Option<SkyLight> {
        if let Some((sky, _)) = self.skylight {
            Some(sky)
        } else {
            None
        }
    }

    pub(super) fn skydome_data(&self) -> Option<(SkyLight, Descriptor, &Pipeline)> {
        if let Some((sky, Some((desc, pipeline)))) = &self.skylight {
            Some((*sky, *desc, pipeline))
        } else {
            None
        }
    }

    pub(super) fn set_skydome(
        &mut self,
        skydome: Option<SkyLight>,
        render_size: vk::Extent2D,
        renderpass: vk::RenderPass,
    ) {
        self.skylight = if let Some(sky) = skydome {
            let render_data = build_skydome(
                self.instance.device().logical_clone(),
                render_size,
                &mut self.dm,
                sky,
                &self.textures.read().unwrap(),
                self.sampler,
                renderpass,
            );
            Some((sky, Some(render_data)))
        } else {
            None
        };
    }

    /// Adds a single texture to the scene.
    pub fn add_texture(&mut self, texture: Texture) {
        self.edited_textures = true;
        let device = self.instance.device();
        let mm = self.instance.allocator();
        let mut unf = UnfinishedExecutions::new(device);
        let mut gcmdm = CommandManager::new(device.logical_clone(), device.graphic_queue().idx, 5);
        let loaded = load_texture_to_gpu(
            Arc::clone(&self.instance),
            mm,
            &mut gcmdm,
            &mut unf,
            &texture,
        );
        self.raw_textures.push(texture);
        self.textures.write().unwrap().push(loaded);
        unf.wait_completion()
    }

    /// Removes from the scene the texture with the given ID.
    pub fn remove_texture(&mut self, id: u16) {
        self.raw_textures.remove(id as usize);
        self.textures.write().unwrap().remove(id as usize);
    }

    /// Returns all the lights in the scene.
    pub fn lights(&self) -> &[Light] {
        &self.lights
    }

    pub fn update_lights(&mut self, lights: &[Light]) {
        self.lights = lights.to_vec();
    }

    pub fn save(&mut self) -> Result<(), std::io::Error> {
        let cameras = [self.current_cam];
        let mut lights = self.lights().to_vec();
        if let Some((sky, _)) = self.skylight {
            lights.push(Light::Sky(sky));
        }
        let materials = self.materials().to_vec();
        let meta = &self.meta;
        let textures = if self.edited_textures {
            Some(&self.raw_textures[..])
        } else {
            None
        };
        self.file.update(
            Some(&cameras),
            Some(&materials),
            Some(&lights),
            textures,
            Some(meta),
        )
    }
}

#[cfg(feature = "vulkan-interactive")]
impl Drop for VulkanScene {
    fn drop(&mut self) {
        self.deinit_pipelines();
        unsafe {
            self.instance
                .device()
                .logical()
                .destroy_sampler(self.sampler, None)
        };
    }
}

fn build_realtime_descriptor(
    dm: &mut DescriptorSetManager,
    transforms_buffer: &AllocatedBuffer,
) -> Descriptor {
    dm.new_set()
        .bind_buffer(
            transforms_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::VERTEX,
        )
        .build()
}

fn gen_sphere(lat: u32, lon: u32, ccw: bool) -> (Vec<f32>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(((lat * lon + 2) * 5) as usize);
    let mut indices = Vec::with_capacity(((lon * 2 + ((lat - 1) * 2 * lon)) * 3) as usize);
    for i in 1..(lat + 1) {
        for j in 0..lon {
            let v = i as f32 / lat as f32;
            let u = j as f32 / (lon - 1) as f32;
            let x = f32::sin(PI * v) * f32::cos(2.0 * PI * u);
            let y = f32::sin(PI * v) * f32::sin(2.0 * PI * u);
            let z = f32::cos(PI * v);
            vertices.extend([x, y, z, u, v]);
        }
    }
    let pole_top = lat * lon;
    let pole_bot = pole_top + 1;
    // The u value at poles can be anything between 0.0 and 1.0
    vertices.extend([0.0, 0.0, 1.0, 0.5, 0.0]);
    vertices.extend([0.0, 0.0, -1.0, 0.5, 1.0]);
    // generate the sphere caps
    for j in 0..lon {
        let i = (lat - 1) * lon;
        let next_j = if j + 1 == lon { 0 } else { j + 1 };
        if ccw {
            indices.extend([pole_top, j, next_j]);
            indices.extend([pole_bot, i + next_j, i + j]);
        } else {
            indices.extend([j, pole_top, next_j]);
            indices.extend([i + j, i + next_j, pole_bot]);
        }
    }
    // generate the rest of the sphere
    for i in 0..lat - 1 {
        for j in 0..lon {
            let next_j = if j + 1 == lon { 0 } else { j + 1 };
            let p0 = i * lon + j;
            let p0n = i * lon + next_j;
            let p1 = (i + 1) * lat + j;
            let p1n = (i + 1) * lon + next_j;
            if ccw {
                indices.extend([p0, p1, p1n]);
                indices.extend([p0, p1n, p0n]);
            } else {
                indices.extend([p0, p1n, p1]);
                indices.extend([p0, p0n, p1n]);
            }
        }
    }
    (vertices, indices)
}

fn build_skydome(
    device: Arc<ash::Device>,
    extent: vk::Extent2D,
    dm: &mut DescriptorSetManager,
    sky: SkyLight,
    textures: &[TextureLoaded],
    sampler: vk::Sampler,
    rp: vk::RenderPass,
) -> (Descriptor, Pipeline) {
    let bound_sky = &textures[sky.tex_id as usize].image;
    let descriptor = dm
        .new_set()
        .bind_image(
            bound_sky,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        )
        .build();
    let mut builder = PipelineBuilder::default();
    let vs = include_shader!("skydome.vert");
    let fs = include_shader!("texture.frag");
    builder.push_shader(vs, "main", vk::ShaderStageFlags::VERTEX);
    builder.push_shader(fs, "main", vk::ShaderStageFlags::FRAGMENT);
    builder.push_constants(24, vk::ShaderStageFlags::VERTEX);
    // vertices will be hard-coded (fullscreen)
    builder.binding_descriptions = Vec::with_capacity(0);
    builder.attribute_descriptions = Vec::with_capacity(0);
    // deactivate depth stencil
    builder.no_depth();
    builder.rasterizer.cull_mode = vk::CullModeFlags::NONE;
    let pipeline = builder.build(device, rp, extent, &[descriptor.layout]);
    (descriptor, pipeline)
}

/// Creates the default sampler for this scene.
/// Uses anisotropic filtering with the max anisotropy supported by the GPU.
fn create_sampler(device: &Device) -> vk::Sampler {
    let max_anisotropy = device.physical().properties.limits.max_sampler_anisotropy;
    let ci = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SamplerCreateFlags::empty(),
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        mip_lod_bias: 0.0,
        anisotropy_enable: vk::TRUE,
        max_anisotropy,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 1000.0, // use maximum possible lod
        border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        unnormalized_coordinates: vk::FALSE,
    };
    unsafe {
        device
            .logical()
            .create_sampler(&ci, None)
            .expect("Failed to create sampler")
    }
}

/// Loads all vertices to GPU.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
fn load_vertices_to_gpu(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unf: &mut UnfinishedExecutions,
    vertices: &[Vertex],
    with_raytrace: bool,
) -> AllocatedBuffer {
    let flags = if with_raytrace {
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    } else {
        vk::BufferUsageFlags::VERTEX_BUFFER
    };
    upload_buffer(device, mm, tcmdm, flags, unf, vertices)
}

/// Loads all transforms to GPU.
/// Likely this will become a per-object binding.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
#[cfg(feature = "vulkan-interactive")]
fn load_transforms_to_gpu(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unf: &mut UnfinishedExecutions,
    transforms: &[Transform],
) -> AllocatedBuffer {
    upload_buffer(
        device,
        mm,
        tcmdm,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        unf,
        transforms,
    )
}

/// Loads all indices to GPU.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
/// Returns the list of meshes and the index buffer.
fn load_indices_to_gpu(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    meshes: &[Mesh],
    with_raytrace: bool,
) -> (Vec<VulkanMesh>, AllocatedBuffer) {
    let size = u64::max(
        1,
        (std::mem::size_of::<u32>() * meshes.iter().map(|m| m.indices.len()).sum::<usize>()) as u64,
    );
    let mut converted_meshes = Vec::with_capacity(meshes.len());
    let raytrace_flags = if with_raytrace {
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    } else {
        vk::BufferUsageFlags::empty()
    };
    let gpu_buffer = mm.create_buffer(
        "indices_dedicated",
        size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | raytrace_flags,
        MemoryLocation::GpuOnly,
    );
    if !meshes.is_empty() {
        let cpu_buffer = mm.create_buffer(
            "indices_local",
            size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        let mut mapped = cpu_buffer
            .allocation()
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        let mut offset = 0;
        for mesh in meshes {
            unsafe {
                std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), mapped, mesh.indices.len());
                mapped = mapped.add(mesh.indices.len());
            };
            let max_index = mesh.indices.iter().max().copied().unwrap_or(0);
            let converted = VulkanMesh {
                mesh_id: mesh.id,
                index_offset: offset,
                index_count: mesh.indices.len() as u32,
                max_index,
                material: mesh.material,
            };
            converted_meshes.push(converted);
            offset += mesh.indices.len() as u32;
        }
        let copy_region = vk::BufferCopy {
            // these are not the allocation offset, but the buffer offset!
            src_offset: 0,
            dst_offset: 0,
            size: cpu_buffer.size,
        };
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_copy_buffer(cmd, cpu_buffer.buffer, gpu_buffer.buffer, &[copy_region]);
            }
        };
        let cmd = tcmdm.get_cmd_buffer();
        let transfer_queue = device.transfer_queue();
        let fence = device.immediate_execute(cmd, transfer_queue, command);
        unfinished.add(fence, cpu_buffer);
    }
    (converted_meshes, gpu_buffer)
}

/// Converts a slice of MeshInstances into a map.
/// The slice is expected to contain One-to-Many relationships.
#[cfg(feature = "vulkan-interactive")]
fn instances_to_map(instances: &[MeshInstance]) -> FnvHashMap<u16, Vec<u16>> {
    // first iteration, record the amount of tranformations for each mesh
    let mut map = FnvHashMap::with_capacity_and_hasher(instances.len(), FnvBuildHasher::default());
    for instance in instances {
        *map.entry(instance.mesh_id).or_insert(0) += 1;
    }
    // second iteration build the map with the correct sizes
    let mut retval =
        FnvHashMap::<u16, Vec<u16>>::with_capacity_and_hasher(map.len(), FnvBuildHasher::default());
    for instance in instances {
        match retval.entry(instance.mesh_id) {
            Entry::Occupied(e) => e.into_mut().push(instance.transform_id),
            Entry::Vacant(e) => {
                let mut val = Vec::with_capacity(*map.get(&instance.mesh_id).unwrap());
                val.push(instance.transform_id);
                e.insert(val);
            }
        }
    }
    retval
}

/// Builds a single material descriptor set.
#[cfg(feature = "vulkan-interactive")]
fn build_mat_desc_set(
    device: &Device,
    textures: &[TextureLoaded],
    params: &AllocatedBuffer,
    sampler: vk::Sampler,
    id: u16,
    material: &Material,
    dm: &mut DescriptorSetManager,
) -> (ShaderMat, Descriptor) {
    use crate::materials::DEFAULT_TEXTURE_ID;

    let mut shader = material.shader;
    let dflt_tex = &textures[DEFAULT_TEXTURE_ID as usize];
    let diffuse = textures.get(material.diffuse as usize).unwrap_or(dflt_tex);
    let align = device
        .physical()
        .properties
        .limits
        .min_uniform_buffer_offset_alignment;
    let padding = padding(PARAMS_SIZE, align);
    let buf_info = vk::DescriptorBufferInfo {
        buffer: params.buffer,
        offset: (PARAMS_SIZE + padding) * id as u64,
        range: PARAMS_SIZE,
    };
    // TODO: merge materials with the same textures to avoid extra bindings
    let mut descriptor = dm
        .new_set()
        .bind_buffer_with_info(
            buf_info,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::FRAGMENT,
        )
        .bind_image(
            &diffuse.image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        );
    if material.opacity != 0 {
        shader = material.shader.two_sided_viewport(); // use a two-sided shader
        let opacity = textures.get(material.opacity as usize).unwrap_or(dflt_tex);
        descriptor = descriptor.bind_image(
            &opacity.image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        );
    }
    (shader, descriptor.build())
}

/// Loads all materials parameters to GPU.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
#[cfg(feature = "vulkan-interactive")]
fn load_materials_parameters(
    device: &Device,
    materials: &[Material],
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
) -> AllocatedBuffer {
    let align = device
        .physical()
        .properties
        .limits
        .min_uniform_buffer_offset_alignment;
    let padding = padding(PARAMS_SIZE, align);
    let size = (PARAMS_SIZE + padding) * materials.len() as u64;
    let cpu_buffer = mm.create_buffer(
        "Materials Parameters CPU",
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let gpu_buffer = mm.create_buffer(
        "Materials Parameters GPU",
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
    );
    let mut mapped = cpu_buffer
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    for mat in materials {
        let params = MaterialParams::from(mat);
        unsafe {
            std::ptr::copy_nonoverlapping(&params, mapped, 1);
            let mapped_void = mapped as *mut c_void;
            mapped = mapped_void.add((PARAMS_SIZE + padding) as usize).cast();
        }
    }
    let copy_region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    };
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_copy_buffer(cmd, cpu_buffer.buffer, gpu_buffer.buffer, &[copy_region]);
        }
    };
    let cmd = tcmdm.get_cmd_buffer();
    let transfer_queue = device.transfer_queue();
    let fence = device.immediate_execute(cmd, transfer_queue, command);
    unfinished.add(fence, cpu_buffer);
    gpu_buffer
}

/// Loads a single texture to the GPU with optimal layout.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
#[cfg(feature = "vulkan-interactive")]
fn load_texture_to_gpu<T: Instance + Send + Sync + 'static>(
    instance: Arc<T>,
    mm: &MemoryManager,
    gcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    texture: &Texture,
) -> TextureLoaded {
    let (width, height) = texture.dimensions(0);
    let mip_levels = texture.max_mipmap_levels();
    let full_size = (0..mip_levels)
        .map(|mip_level| texture.size_bytes(mip_level))
        .sum::<usize>();
    let extent = vk::Extent2D {
        width: width as u32,
        height: height as u32,
    };
    let vkformat = match texture.format() {
        TextureFormat::Gray => vk::Format::R8_UNORM,
        TextureFormat::RgbaSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::RgbaNorm => vk::Format::R8G8B8A8_UNORM,
    };
    let cpu_buf = mm.create_buffer(
        "Texture Buffer",
        full_size as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let image = mm.create_image_gpu(
        "Texture Image",
        vkformat,
        extent,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        mip_levels as u32,
    );
    let mut mapped = cpu_buf
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    // if the texture has all mipmaps, just copy them to gpu.
    // otherwise copy the first level and generates the other ones.
    let mipmaps_to_copy = if texture.has_mipmaps() {
        texture.mipmap_levels()
    } else {
        1
    };
    for mip_level in 0..mipmaps_to_copy {
        let size = texture.size_bytes(mip_level);
        unsafe {
            std::ptr::copy_nonoverlapping(texture.ptr(mip_level), mapped, size);
            mapped = mapped.add(size);
        }
    }
    let subresource_range_mipmaps_to_copy = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mipmaps_to_copy as u32,
        base_array_layer: 0,
        layer_count: 1,
    };
    let subresource_range_mipmaps_to_use = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mip_levels as u32,
        base_array_layer: 0,
        layer_count: 1,
    };
    let barrier_transfer = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: image.image,
        subresource_range: subresource_range_mipmaps_to_copy,
    };
    let mut regions = Vec::with_capacity(mipmaps_to_copy);
    let mut buffer_offset = 0;
    for level in 0..mipmaps_to_copy {
        let (mip_w, mip_h) = texture.dimensions(level);
        let image_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: level as u32,
            base_array_layer: 0,
            layer_count: 1,
        };
        let copy_region = vk::BufferImageCopy {
            buffer_offset,
            buffer_row_length: 0,   // 0 = same as image width
            buffer_image_height: 0, // 0 = same as image height
            image_subresource,
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: mip_w as u32,
                height: mip_h as u32,
                depth: 1,
            },
        };
        regions.push(copy_region);
        buffer_offset += texture.size_bytes(level) as u64;
    }
    let command = unsafe {
        |device: &ash::Device, cmd: vk::CommandBuffer| {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_transfer],
            );
            device.cmd_copy_buffer_to_image(
                cmd,
                cpu_buf.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            // generates mipmaps on the fly if they are not loaded from file
            let barrier_use = if !texture.has_mipmaps() {
                // ensures level 0 is finished
                let mip_0_subrange = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let barrier_0_finished = vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: image.image,
                    subresource_range: mip_0_subrange,
                };
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_0_finished],
                );
                // generates all other levels
                for mip_level in 1..mip_levels {
                    let mip_subrange = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: mip_level as u32,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let barrier_prepare_for_write = vk::ImageMemoryBarrier {
                        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                        p_next: ptr::null(),
                        src_access_mask: vk::AccessFlags::NONE_KHR,
                        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: image.image,
                        subresource_range: mip_subrange,
                    };
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier_prepare_for_write],
                    );
                    let src_subresource = vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: mip_level as u32 - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let dst_subresource = vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: mip_level as u32,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let (prev_w, prev_h) = texture.dimensions(mip_level - 1);
                    let (req_w, req_h) = texture.dimensions(mip_level);
                    let blit_region = vk::ImageBlit {
                        src_subresource,
                        dst_subresource,
                        src_offsets: [
                            vk::Offset3D::default(),
                            vk::Offset3D {
                                x: prev_w as i32,
                                y: prev_h as i32,
                                z: 1,
                            },
                        ],
                        dst_offsets: [
                            vk::Offset3D::default(),
                            vk::Offset3D {
                                x: req_w as i32,
                                y: req_h as i32,
                                z: 1,
                            },
                        ],
                    };
                    device.cmd_blit_image(
                        cmd,
                        image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit_region],
                        vk::Filter::LINEAR,
                    );
                    let barrier_for_next = vk::ImageMemoryBarrier {
                        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                        p_next: ptr::null(),
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: image.image,
                        subresource_range: mip_subrange,
                    };
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier_for_next],
                    );
                }
                vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: image.image,
                    subresource_range: subresource_range_mipmaps_to_use,
                }
            } else {
                vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: image.image,
                    subresource_range: subresource_range_mipmaps_to_use,
                }
            };
            // prepare the texture for shader reading
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_use],
            );
        }
    };
    let cmd = gcmdm.get_cmd_buffer();
    let device = instance.device();
    let transfer_queue = device.transfer_queue();
    let fence = device.immediate_execute(cmd, transfer_queue, command);
    unfinished.add(fence, cpu_buf);
    TextureLoaded {
        format: texture.info().format,
        image,
    }
}

// sort mehses by shader id (first) and then material id (second) to minimize binding changes
#[cfg(feature = "vulkan-interactive")]
fn sort_meshes(meshes: &mut [VulkanMesh], mats: &[(ShaderMat, Descriptor)]) {
    meshes.sort_unstable_by(|a, b| {
        let (_, desc_a) = &mats[a.material as usize];
        let (_, desc_b) = &mats[b.material as usize];
        match desc_a.cmp(desc_b) {
            std::cmp::Ordering::Less => std::cmp::Ordering::Less,
            std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
            std::cmp::Ordering::Equal => a.material.cmp(&b.material),
        }
    });
}

/// Material parameters representation used by the shaders.
#[repr(C)]
struct MaterialParams {
    /// Multiplier for the diffuse color.
    diffuse_mul: Vec3<f32>,
}

/// Size of the material parameters struct.
#[cfg(feature = "vulkan-interactive")]
const PARAMS_SIZE: u64 = std::mem::size_of::<MaterialParams>() as u64;

impl From<&Material> for MaterialParams {
    fn from(material: &Material) -> Self {
        MaterialParams {
            diffuse_mul: Vec3::new(
                material.diffuse_mul[0] as f32 / 255.0,
                material.diffuse_mul[1] as f32 / 255.0,
                material.diffuse_mul[2] as f32 / 255.0,
            ),
        }
    }
}

/// How many bytes are required for `n` to be aligned to `align` boundary
pub fn padding<T: Into<u64>>(n: T, align: T) -> u64 {
    ((!n.into()).wrapping_add(1)) & (align.into().wrapping_sub(1))
}

pub struct RayTraceScene<T: Instance + Send + Sync> {
    pub(crate) camera: Camera,
    pub(crate) descriptor: Descriptor,
    sampler: vk::Sampler,
    vertex_buffer: Arc<AllocatedBuffer>,
    index_buffer: Arc<AllocatedBuffer>,
    instance_buffer: AllocatedBuffer,
    material_buffer: AllocatedBuffer,
    light_buffer: AllocatedBuffer,
    derivative_buffer: AllocatedBuffer,
    transforms_buffer: Arc<AllocatedBuffer>,
    // (material_id, transform_ids)
    material_instance_ids: FnvHashMap<u16, Vec<u16>>,
    // does not account for skylights
    pub(crate) lights_no: u32,
    pub(crate) meta: Meta,
    textures: Arc<RwLock<Vec<TextureLoaded>>>,
    acc: SceneAS,
    dm: DescriptorSetManager,
    instance: Arc<T>,
}

impl<T: Instance + Send + Sync> RayTraceScene<T> {
    const AVG_DESC: [(vk::DescriptorType, f32); 2] = [
        (vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1.0),
        (vk::DescriptorType::STORAGE_BUFFER, 4.0),
    ];

    pub fn new(
        instance: Arc<RayTraceInstance>,
        parsed: Box<dyn ParsedScene>,
    ) -> RayTraceScene<RayTraceInstance> {
        let mm = instance.allocator();
        let device = instance.device();
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let mut unf = UnfinishedExecutions::new(instance.device());
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 1);
        let mut ccmdm = CommandManager::new(device.logical_clone(), device.compute_queue().idx, 1);
        let instances = parsed.instances().unwrap_or_default();
        let transforms = parsed
            .transforms()
            .unwrap_or_else(|_| vec![Transform::default()]);
        let materials = parsed
            .materials()
            .unwrap_or_else(|_| vec![Material::default()]);
        let mut lights = parsed.lights().unwrap_or_default();
        let mut dm = DescriptorSetManager::new(
            instance.device().logical_clone(),
            &Self::AVG_DESC,
            instance.desc_layout_cache(),
        );
        let vertex_buffer = load_vertices_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed.vertices().unwrap_or_default(),
            true,
        );
        let (meshes, index_buffer) = load_indices_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed.meshes().unwrap_or_default(),
            true,
        );
        let derivative_buffer = calculate_geometric_derivatives(
            instance.as_ref(),
            &mut dm,
            &mut ccmdm,
            &meshes,
            &index_buffer,
            &vertex_buffer,
            &mut unf,
        );
        let transforms_buffer =
            load_transforms_to_gpu(device, mm, &mut tcmdm, &mut unf, &transforms);
        let textures = Arc::new(RwLock::new(
            parsed
                .textures()
                .unwrap_or_else(|_| vec![Texture::default()])
                .iter()
                .map(|tex| load_texture_to_gpu(instance.clone(), mm, &mut tcmdm, &mut unf, tex))
                .collect::<Vec<_>>(),
        ));
        let material_instance_ids = map_materials_to_instances(&meshes, &instances);
        let builder = SceneASBuilder::new(
            device,
            loader,
            mm,
            &mut ccmdm,
            &vertex_buffer,
            &index_buffer,
        )
        .with_meshes(&meshes, &instances, &transforms, &materials);
        let acc = builder.build();
        let camera = parsed
            .cameras()
            .unwrap_or_default()
            .pop()
            .unwrap_or_default();
        let instance_buffer =
            load_raytrace_instances_to_gpu(device, mm, &mut tcmdm, &mut unf, &meshes, &instances);
        let material_buffer =
            load_raytrace_materials_to_gpu(device, mm, &mut tcmdm, &mut unf, &materials);
        let light_buffer = load_raytrace_lights_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &lights,
            &material_instance_ids,
        );
        let sampler = create_sampler(device);
        let descriptor = build_raytrace_descriptor(
            &mut dm,
            &acc,
            &vertex_buffer,
            &index_buffer,
            &instance_buffer,
            &material_buffer,
            &light_buffer,
            &derivative_buffer,
            &transforms_buffer,
            &textures.read().unwrap(),
            sampler,
        );
        let skylight = lights
            .iter()
            .find_map(|l| match l {
                Light::Sky(s) => Some(s),
                _ => None,
            })
            .copied();
        lights = lights
            .into_iter()
            .filter(|l| l.ltype() != LightType::SKY)
            .collect();
        let meta = parsed.meta().unwrap_or_default();
        unf.wait_completion();
        RayTraceScene {
            camera,
            descriptor,
            sampler,
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            instance_buffer,
            material_buffer,
            light_buffer,
            derivative_buffer,
            transforms_buffer: Arc::new(transforms_buffer),
            material_instance_ids,
            lights_no: lights.len() as u32,
            meta,
            textures,
            acc,
            dm,
            instance,
        }
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn refresh_descriptors(&mut self) {
        self.descriptor = build_raytrace_descriptor(
            &mut self.dm,
            &self.acc,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.instance_buffer,
            &self.material_buffer,
            &self.light_buffer,
            &self.derivative_buffer,
            &self.transforms_buffer,
            &self.textures.read().unwrap(),
            self.sampler,
        );
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(crate) fn update_materials_and_lights(
        &mut self,
        materials: &[Material],
        lights: &[Light],
        tcmdm: &mut CommandManager,
        unf: &mut UnfinishedExecutions,
    ) {
        let mm = self.instance.allocator();
        let mut mat_buffer =
            load_raytrace_materials_to_gpu(self.instance.device(), mm, tcmdm, unf, materials);
        let mut light_buffer = load_raytrace_lights_to_gpu(
            self.instance.device(),
            mm,
            tcmdm,
            unf,
            lights,
            &self.material_instance_ids,
        );
        // cannot drop yet, the loading is not finished yet
        std::mem::swap(&mut self.material_buffer, &mut mat_buffer);
        std::mem::swap(&mut self.light_buffer, &mut light_buffer);
        unf.add_buffer(mat_buffer);
        unf.add_buffer(light_buffer);
        self.lights_no = lights.len() as u32;
        self.refresh_descriptors();
    }
}

impl<T: Instance + Send + Sync> Drop for RayTraceScene<T> {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .device()
                .logical()
                .destroy_sampler(self.sampler, None)
        }
    }
}

#[cfg(feature = "vulkan-interactive")]
impl From<&VulkanScene> for RayTraceScene<PresentInstance> {
    fn from(scene: &VulkanScene) -> RayTraceScene<PresentInstance> {
        let instance = Arc::clone(&scene.instance);
        let device = instance.device();
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let mm = instance.allocator();
        let mut unf = UnfinishedExecutions::new(instance.device());
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 1);
        let mut ccmdm = CommandManager::new(device.logical_clone(), device.compute_queue().idx, 11);
        let mut dm = DescriptorSetManager::new(
            instance.device().logical_clone(),
            &Self::AVG_DESC,
            instance.desc_layout_cache(),
        );
        let instances = scene.file.instances().unwrap_or_default();
        let transforms = scene
            .file
            .transforms()
            .unwrap_or_else(|_| vec![Transform::default()]);
        let materials = scene.materials().to_vec();
        let vertex_buffer = Arc::clone(&scene.vertex_buffer);
        let index_buffer = Arc::clone(&scene.index_buffer);
        let transforms_buffer = Arc::clone(&scene.transforms);
        let textures = Arc::clone(&scene.textures);
        let derivative_buffer = calculate_geometric_derivatives(
            instance.as_ref(),
            &mut dm,
            &mut ccmdm,
            &scene.meshes,
            &index_buffer,
            &vertex_buffer,
            &mut unf,
        );
        let material_instance_ids = map_materials_to_instances(&scene.meshes, &instances);
        let builder = SceneASBuilder::new(
            device,
            loader,
            mm,
            &mut ccmdm,
            &vertex_buffer,
            &index_buffer,
        )
        .with_meshes(&scene.meshes, &instances, &transforms, &materials);
        let acc = builder.build();
        let instance_buffer = load_raytrace_instances_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &scene.meshes,
            &instances,
        );
        let material_buffer =
            load_raytrace_materials_to_gpu(device, mm, &mut tcmdm, &mut unf, &materials);
        let light_buffer = load_raytrace_lights_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &scene.lights,
            &material_instance_ids,
        );
        let sampler = create_sampler(device);
        let descriptor = build_raytrace_descriptor(
            &mut dm,
            &acc,
            &vertex_buffer,
            &index_buffer,
            &instance_buffer,
            &material_buffer,
            &light_buffer,
            &derivative_buffer,
            &transforms_buffer,
            &textures.read().unwrap(),
            sampler,
        );
        let meta = scene.meta;
        unf.wait_completion();
        RayTraceScene {
            camera: scene.current_cam,
            descriptor,
            sampler,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            material_buffer,
            light_buffer,
            derivative_buffer,
            transforms_buffer,
            material_instance_ids,
            lights_no: scene.lights.len() as u32,
            meta,
            textures,
            acc,
            dm,
            instance,
        }
    }
}

fn map_materials_to_instances(
    meshes: &[VulkanMesh],
    instances: &[MeshInstance],
) -> FnvHashMap<u16, Vec<u16>> {
    let materials_by_mesh_id = meshes
        .iter()
        .map(|m| (m.mesh_id, m.material))
        .collect::<FnvHashMap<_, _>>();
    let mut retval = FnvHashMap::default();
    for (instance_id, instance) in instances.iter().enumerate() {
        let material_id = *materials_by_mesh_id.get(&instance.mesh_id).unwrap();
        retval
            .entry(material_id)
            .or_insert(Vec::new())
            .push(instance_id as u16);
    }
    retval
}

fn load_raytrace_instances_to_gpu(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unf: &mut UnfinishedExecutions,
    meshes: &[VulkanMesh],
    instances: &[MeshInstance],
) -> AllocatedBuffer {
    let mut rt_instances = Vec::with_capacity(instances.len());
    let meshes_indexed = meshes
        .iter()
        .map(|m| (m.mesh_id, m))
        .collect::<FnvHashMap<_, _>>();
    for instance in instances {
        if let Some(original_mesh) = meshes_indexed.get(&instance.mesh_id) {
            rt_instances.push(RTInstance {
                index_offset: original_mesh.index_offset,
                index_count: original_mesh.index_count,
                material_id: original_mesh.material as u32,
                transform_id: instance.transform_id as u32,
            });
        } else {
            log::warn!(
                "Found an instance reference that points to no mesh (MeshID: {})",
                instance.mesh_id
            );
        }
    }
    upload_buffer(
        device,
        mm,
        tcmdm,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        unf,
        &rt_instances,
    )
}

fn load_raytrace_materials_to_gpu(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unf: &mut UnfinishedExecutions,
    materials: &[Material],
) -> AllocatedBuffer {
    let data = materials
        .iter()
        .map(|mat| {
            let metal: Metal = mat.metal;
            let ior = metal.index_of_refraction();
            let k = metal.absorption();
            let fresnel = (ior * ior) + (k * k);
            RTMaterial {
                diffuse_mul: col_int_to_f32(mat.diffuse_mul),
                metal_ior: ior,
                metal_fresnel: fresnel,
                diffuse: mat.diffuse as u32,
                roughness: mat.roughness as u32,
                metalness: mat.metalness as u32,
                opacity: mat.opacity as u32,
                normal: mat.normal as u32,
                bsdf_index: mat.shader.sbt_callable_index(),
                roughness_mul: mat.roughness_mul,
                metalness_mul: mat.metalness_mul as f32,
                anisotropy: mat.anisotropy,
                ior_dielectric: mat.ior,
                is_specular: mat.shader.is_specular() as u32,
                is_emissive: mat.emissive_col.is_some() as u32,
                emissive_col: col_int_to_f32(mat.emissive_col.unwrap_or([0, 0, 0])),
            }
        })
        .collect::<Vec<_>>();
    upload_buffer(
        device,
        mm,
        tcmdm,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        unf,
        &data,
    )
}

fn load_raytrace_lights_to_gpu(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unf: &mut UnfinishedExecutions,
    lights: &[Light],
    transform_ids: &FnvHashMap<u16, Vec<u16>>,
) -> AllocatedBuffer {
    let mut data = Vec::new();
    for l in lights {
        let pos = l.position();
        let mut dir = l.direction();
        if dir.x == 0.0 && dir.y == 0.0 && dir.z == 0.0 {
            log::warn!("zero length vector was changed to (1.0, 0.0, 0.0)");
            dir.y = 1.0;
        }
        dir.normalize();
        let mut addlight = RTLight {
            color: l.emission(),
            pos: [pos.x, pos.y, pos.z, 0.0],
            dir: [dir.x, dir.y, dir.z, 0.0],
            shader: l.ltype().sbt_callable_index(),
            instance_id: u32::MAX,
            intensity: l.intensity(),
            delta: l.ltype().is_delta(),
        };
        if l.ltype() == LightType::AREA {
            // add all the isntances of the material (material_id to isntance_id conversion)
            let material_id = l.material_id() as u16;
            let dflt = vec![0];
            let instances = transform_ids.get(&material_id).unwrap_or(&dflt);
            for instance in instances {
                addlight.instance_id = *instance as u32;
                data.push(addlight);
            }
        } else {
            data.push(addlight);
        }
    }
    if data.is_empty() {
        // push an empty light to avoid having a zero sized buffer
        // since there is no "default" light
        data.push(RTLight {
            color: Spectrum::black(),
            pos: [0.0, 0.0, 0.0, 0.0],
            dir: [0.0, 0.0, 0.0, 0.0],
            shader: 0,
            instance_id: u32::MAX,
            intensity: 1.0,
            delta: true,
        })
    }
    upload_buffer(
        device,
        mm,
        tcmdm,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        unf,
        &data,
    )
}

fn col_int_to_f32(col: [u8; 3]) -> [f32; 4] {
    [
        col[0] as f32 / 255.0,
        col[1] as f32 / 255.0,
        col[2] as f32 / 255.0,
        1.0,
    ]
}

// 'static in T avoids references that would fuck up the size_of value
fn upload_buffer<T: Sized + 'static>(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    flags: vk::BufferUsageFlags,
    unf: &mut UnfinishedExecutions,
    data: &[T],
) -> AllocatedBuffer {
    let size = (std::mem::size_of::<T>() * data.len()) as u64;
    let gpu_buffer;
    if !data.is_empty() {
        let cpu_buffer = mm.create_buffer(
            "scene_buffer_local",
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        gpu_buffer = mm.create_buffer(
            "scene_buffer_dedicated",
            size,
            flags | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        );
        let mapped = cpu_buffer
            .allocation()
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len()) };
        let copy_region = vk::BufferCopy {
            // these are not the allocation offset, but the buffer offset!
            src_offset: 0,
            dst_offset: 0,
            size: cpu_buffer.size,
        };
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_copy_buffer(cmd, cpu_buffer.buffer, gpu_buffer.buffer, &[copy_region]);
            }
        };
        let cmd = tcmdm.get_cmd_buffer();
        let fence = device.immediate_execute(cmd, device.transfer_queue(), command);
        unf.add(fence, cpu_buffer);
    } else {
        gpu_buffer = mm.create_buffer("empty_buffer", 1, flags, MemoryLocation::GpuOnly);
    }
    gpu_buffer
}

fn calculate_geometric_derivatives<T: Instance>(
    instance: &T,
    dm: &mut DescriptorSetManager,
    ccmdm: &mut CommandManager,
    meshes: &[VulkanMesh],
    index_buffer: &AllocatedBuffer,
    vertex_buffer: &AllocatedBuffer,
    unf: &mut UnfinishedExecutions,
) -> AllocatedBuffer {
    let mm = instance.allocator();
    let triangles = meshes
        .iter()
        .map(|m| m.index_offset + m.index_count)
        .max()
        .unwrap_or(0)
        / 3;
    let device = instance.device();
    let size = u64::max(1, triangles as u64 * 48); // 3*vec4(normal, dpdu and dpdv)
    let buffer = mm.create_buffer(
        "Derivatives",
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
    );
    if triangles > 0 {
        let desc = dm
            .new_set()
            .bind_buffer(
                vertex_buffer,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
            )
            .bind_buffer(
                index_buffer,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
            )
            .bind_buffer(
                &buffer,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
            )
            .build();
        let pipeline = build_compute_pipeline(
            device.logical_clone(),
            include_shader!("generate_derivatives.comp").to_vec(),
            4,
            &[desc.layout],
        );
        let cmd = ccmdm.get_cmd_buffer();
        let group_count = (triangles / 256) + 1;
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[desc.set],
                    &[],
                );
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                device.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &u32::to_ne_bytes(triangles),
                );
                device.cmd_dispatch(cmd, group_count, 1, 1);
            }
        };
        let fence = device.immediate_execute(cmd, device.compute_queue(), command);
        unf.add_pipeline_execution(fence, pipeline);
    }
    buffer
}

fn build_raytrace_descriptor(
    dm: &mut DescriptorSetManager,
    acc: &SceneAS,
    vertex_buffer: &AllocatedBuffer,
    index_buffer: &AllocatedBuffer,
    instance_buffer: &AllocatedBuffer,
    material_buffer: &AllocatedBuffer,
    light_buffer: &AllocatedBuffer,
    derivative_buffer: &AllocatedBuffer,
    transforms_buffer: &AllocatedBuffer,
    textures: &[TextureLoaded],
    sampler: vk::Sampler,
) -> Descriptor {
    let textures_memory = textures
        .iter()
        .map(|t| t.image.image_view)
        .collect::<Vec<_>>();
    dm.new_set()
        .bind_acceleration_structure(&acc.tlas.accel, vk::ShaderStageFlags::RAYGEN_KHR)
        .bind_buffer(
            vertex_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::ANY_HIT_KHR
                | vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_buffer(
            index_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::ANY_HIT_KHR
                | vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_buffer(
            instance_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::ANY_HIT_KHR
                | vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_buffer(
            material_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::CALLABLE_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::ANY_HIT_KHR,
        )
        .bind_buffer(
            light_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_image_array(
            &textures_memory,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::ANY_HIT_KHR
                | vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_buffer(
            derivative_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            transforms_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .build()
}
