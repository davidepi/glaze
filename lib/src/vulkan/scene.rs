use super::acceleration::{SceneAS, SceneASBuilder};
use super::cmd::CommandManager;
use super::descriptor::{Descriptor, DescriptorSetManager};
use super::device::Device;
use super::instance::Instance;
use super::memory::{AllocatedBuffer, MemoryManager};
use super::pipeline::build_compute_pipeline;
use super::raytrace_structures::{RTInstance, RTLight, RTMaterial, RTSky};
use super::{as_u8_slice, AllocatedImage, UnfinishedExecutions};
use crate::geometry::{Distribution1D, Distribution2D};
use crate::materials::TextureLoaded;
use crate::{
    include_shader, Camera, ColorRGB, Light, LightType, Material, Mesh, MeshInstance, Meta, Metal,
    ParsedScene, RayTraceInstance, Spectrum, Texture, TextureFormat, Transform, Vertex,
};
#[cfg(feature = "vulkan-interactive")]
use crate::{MaterialType, Pipeline, PipelineBuilder, PresentInstance, Serializer};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk;
use cgmath::{InnerSpace, SquareMatrix, Vector3 as Vec3};
use fnv::FnvHashMap;
use gpu_allocator::MemoryLocation;
use std::f32::consts::PI;
use std::ptr;
use std::sync::{Arc, RwLock};

/// A scene optimized to be rendered using this crates vulkan implementation.
#[cfg(feature = "vulkan-interactive")]
pub struct RealtimeScene {
    /// The scene on disk.
    file: Box<dyn ParsedScene + Send>,
    /// Not used in this scene, but required in case of updates.
    pub(super) meta: Meta,
    /// The camera for the current scene.
    pub current_cam: Camera,
    /// The buffer containing all vertices for the current scene.
    /// Might be shared with the RayTraceScene.
    pub(super) vertex_buffer: Arc<AllocatedBuffer>,
    /// The buffer containing all indices for the current scene.
    /// Might be shared with the RayTraceScene.
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
    pub(super) materials_desc: Vec<(MaterialType, Descriptor)>,
    /// All the materials in the scene.
    materials: Vec<Material>,
    /// Map of all shaders in the scene with their pipeline.
    pub(super) pipelines: FnvHashMap<MaterialType, Pipeline>,
    /// All textures in the scene.
    /// Might be shared with RayTraceScene.
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
    /// skylight draw data.
    pub(super) skydome_data: Option<SkydomeDrawData>,
    /// All the lights in the scene, including skylight. Skylight is ALWAYS the last member.
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

/// Data required to draw the skydome in the non-raytraced renderer.
#[cfg(feature = "vulkan-interactive")]
pub struct SkydomeDrawData {
    /// Number of vertices in the skydome
    pub index_count: u32,
    /// Contains the vertices of the sphere dome, allocated in the GPU.
    pub vertices: AllocatedBuffer,
    /// Cibstaubs the indices of the sphere dome, allocated in the GPU.
    pub indices: AllocatedBuffer,
    /// Descriptor for the texture
    pub descriptor: Descriptor,
    /// Pipeline for drawing the skydome
    pub pipeline: Pipeline,
}

#[cfg(feature = "vulkan-interactive")]
impl RealtimeScene {
    /// Converts a parsed scene into a vulkan scene.
    ///
    /// `wchan` is used to send feedbacks about the current loading status.
    pub fn new(instance: Arc<PresentInstance>, parsed: Box<dyn ParsedScene + Send>) -> Self {
        let device = instance.device();
        let mm = instance.allocator();
        let with_raytrace = instance.supports_raytrace();
        let mut unf = UnfinishedExecutions::new(device);
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue(), 5);
        let mut gcmdm = CommandManager::new(device.logical_clone(), device.graphic_queue(), 5);
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
        let sampler = create_sampler(device, false);
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
        let lights = parsed.lights().unwrap_or_default();
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
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        sort_meshes(&mut meshes, &materials_desc);
        // ensures a single skylight and as last member. This guarantee is not satisfied by the
        // parser but by the scene.
        let lights = reorder_lights(lights);
        let meta = parsed.meta().unwrap_or_default();
        let scene_desc = build_realtime_descriptor(&mut dm, &transforms);
        RealtimeScene {
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
            skydome_data: None,
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
        let fence = device.submit_immediate(tcmdm, command);
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
            build_realtime_pipeline(new.mtype).build(
                device.logical_clone(),
                rpass,
                render_size,
                &[frame_desc_layout, new_desc.layout, self.scene_desc.layout],
            )
        });
        // keep the list of lights aligned
        let old = &self.materials[mat_id as usize];
        if new.emissive_col.is_some() || old.emissive_col.is_some() {
            if new.emissive_col.is_some() && old.emissive_col.is_none() {
                // lights should be added
                self.lights
                    .push(Light::new_area(new.name.clone(), mat_id as u32, 1.0));
            } else if new.emissive_col.is_none() && old.emissive_col.is_some() {
                // light should be removed
                self.lights
                    .retain(|x| x.ltype() != LightType::AREA || x.resource_id() != mat_id as u32);
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
        for (mtype, desc) in &self.materials_desc {
            let device = self.instance.device().logical_clone();
            self.pipelines.entry(*mtype).or_insert_with(|| {
                build_realtime_pipeline(*mtype).build(
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
        if let Some(light) = self
            .lights
            .last()
            .filter(|x| x.ltype() == LightType::SKY)
            .cloned()
        {
            let device = self.instance.device();
            let mm = self.instance.allocator();
            let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue(), 1);
            self.skydome_data = Some(build_skydome_realtime(
                device,
                mm,
                render_size,
                &mut tcmdm,
                &mut self.dm,
                light,
                &self.textures.read().unwrap(),
                self.sampler,
                renderpass,
            ));
        }
    }

    /// Destroys the scene's pipelines.
    pub(super) fn deinit_pipelines(&mut self) {
        self.pipelines.clear();
        self.skydome_data = None;
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
    pub fn skydome(&self) -> Option<Light> {
        self.lights
            .last()
            .filter(|x| x.ltype() == LightType::SKY)
            .cloned()
    }

    /// Adds a single texture to the scene.
    pub fn add_texture(&mut self, texture: Texture) {
        self.edited_textures = true;
        let device = self.instance.device();
        let mm = self.instance.allocator();
        let mut unf = UnfinishedExecutions::new(device);
        let mut gcmdm = CommandManager::new(device.logical_clone(), device.graphic_queue(), 5);
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
    /// The sky is ALWAYS the last light in this slice.
    ///
    /// If the last element is not a [LightType::SKY] no such type can be found in the slice.
    pub fn lights(&self) -> &[Light] {
        &self.lights
    }

    /// Changes the lights in the scene.
    /// If there is a skylight in the scene it is expected to be the last element of the input
    /// slice.
    pub(super) fn update_lights(
        &mut self,
        lights: &[Light],
        render_size: vk::Extent2D,
        renderpass: vk::RenderPass,
    ) {
        self.lights = reorder_lights(lights.to_vec());
        if let Some(sky) = self
            .lights
            .last()
            .filter(|x| x.ltype() == LightType::SKY)
            .cloned()
        {
            // update skydome
            let device = self.instance.device();
            let mm = self.instance.allocator();
            let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue(), 1);
            self.skydome_data = Some(build_skydome_realtime(
                device,
                mm,
                render_size,
                &mut tcmdm,
                &mut self.dm,
                sky,
                &self.textures.read().unwrap(),
                self.sampler,
                renderpass,
            ));
        } else {
            self.skydome_data = None;
        }
    }

    /// Saves the scene on the disk.
    pub fn save(&mut self) -> Result<(), std::io::Error> {
        let cameras = [self.current_cam];
        let lights = self.lights().to_vec();
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

    /// Saves the scene as a new file.
    ///
    /// The file is saved using [ParserVersion::V1]
    pub fn save_as(&self, path: &str) -> Result<(), std::io::Error> {
        Serializer::new(path, crate::ParserVersion::V1)
            .with_vertices(&self.file.vertices()?)
            .with_meshes(&self.file.meshes()?)
            .with_instances(&self.file.instances()?)
            .with_transforms(&self.file.transforms()?)
            .with_cameras(&[self.current_cam])
            .with_lights(self.lights())
            .with_materials(self.materials())
            .with_metadata(&self.meta)
            .with_textures(&self.raw_textures[..])
            .serialize()
    }
}

#[cfg(feature = "vulkan-interactive")]
impl Drop for RealtimeScene {
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

/// Build the scene-bound descriptor for the realtime renderer.
#[cfg(feature = "vulkan-interactive")]
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

/// Build the realtime pipeline for each material type
#[cfg(feature = "vulkan-interactive")]
fn build_realtime_pipeline(mtype: MaterialType) -> PipelineBuilder {
    let mut pipeline = PipelineBuilder::default();
    let vertex_shader = include_shader!("flat.vert");
    let fragment_shader = match mtype {
        MaterialType::INTERNAL_FLAT_2SIDED => include_shader!("flat_twosided.frag").to_vec(),
        _ => include_shader!("flat.frag").to_vec(),
    };
    pipeline.push_shader(vertex_shader, "main", ash::vk::ShaderStageFlags::VERTEX);
    pipeline.push_shader(
        &fragment_shader,
        "main",
        ash::vk::ShaderStageFlags::FRAGMENT,
    );
    pipeline.push_constants(4, vk::ShaderStageFlags::VERTEX);
    if mtype == MaterialType::INTERNAL_FLAT_2SIDED {
        pipeline.rasterizer.cull_mode = vk::CullModeFlags::NONE;
    }
    pipeline
}

/// Generates a icosphere.
#[cfg(feature = "vulkan-interactive")]
fn gen_icosphere(subdivisions: u8) -> (Vec<f32>, Vec<u32>) {
    // move a vertex over the unit sphere
    fn unit_sphere(x: f32, y: f32, z: f32) -> [f32; 3] {
        let len = f32::sqrt(x * x + y * y + z * z);
        [x / len, y / len, z / len]
    }
    // calculates midpoint between two vertices and caches it.
    // closely related to gen_icosphere, so is inside here.
    fn icosphere_midpoint(
        mut a: u32,
        mut b: u32,
        vertices: &mut Vec<f32>,
        cache: &mut FnvHashMap<u64, u32>,
    ) -> u32 {
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        let key = (a as u64) << 32 | b as u64;
        *cache.entry(key).or_insert({
            let tri_a = &vertices[a as usize * 3..a as usize * 3 + 3];
            let tri_b = &vertices[b as usize * 3..b as usize * 3 + 3];
            let mid_p = unit_sphere(
                (tri_a[0] + tri_b[0]) / 2.0,
                (tri_a[1] + tri_b[1]) / 2.0,
                (tri_a[2] + tri_b[2]) / 2.0,
            );
            let index = vertices.len() / 3;
            vertices.extend(mid_p);
            index as u32
        })
    }

    // subdv 0
    let mut cache = FnvHashMap::default();
    let mid = 0.8506508;
    let one = 0.5257311;
    let mut vertices = vec![
        -one, mid, 0.0, one, mid, 0.0, -one, -mid, 0.0, one, -mid, 0.0, 0.0, -one, mid, 0.0, one,
        mid, 0.0, -one, -mid, 0.0, one, -mid, mid, 0.0, -one, mid, 0.0, one, -mid, 0.0, -one, -mid,
        0.0, one,
    ];
    let mut indices = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];
    for _ in 0..subdivisions {
        indices = indices
            .chunks_exact(3)
            .flat_map(|f| {
                let a = icosphere_midpoint(f[0], f[1], &mut vertices, &mut cache);
                let b = icosphere_midpoint(f[1], f[2], &mut vertices, &mut cache);
                let c = icosphere_midpoint(f[2], f[0], &mut vertices, &mut cache);
                [f[0], a, c, f[1], b, a, f[2], c, b, a, b, c]
            })
            .collect();
    }
    (vertices, indices)
}

/// reorder the lights, so there is only a single LightType::SKY and is the last element of the
/// array.
fn reorder_lights(mut lights: Vec<Light>) -> Vec<Light> {
    let sky = lights.iter().find(|l| l.ltype() == LightType::SKY).cloned();
    lights.retain(|l| l.ltype() != LightType::SKY);
    if let Some(sky) = sky {
        lights.push(sky);
    }
    lights
}

/// Build the skydome for the realtime renderer (including its pipelines).
#[cfg(feature = "vulkan-interactive")]
fn build_skydome_realtime(
    device: &Device,
    mm: &MemoryManager,
    extent: vk::Extent2D,
    tcmdm: &mut CommandManager,
    dm: &mut DescriptorSetManager,
    sky: Light,
    textures: &[TextureLoaded],
    sampler: vk::Sampler,
    rp: vk::RenderPass,
) -> SkydomeDrawData {
    // geometry
    let mut unf = UnfinishedExecutions::new(device);
    let (v, i) = gen_icosphere(3);
    let si = i.into_iter().map(|i| i as u16).collect::<Vec<_>>();
    let vertices = upload_buffer(
        device,
        mm,
        tcmdm,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &mut unf,
        &v,
    );
    let indices = upload_buffer(
        device,
        mm,
        tcmdm,
        vk::BufferUsageFlags::INDEX_BUFFER,
        &mut unf,
        &si,
    );
    // descriptor
    let bound_sky = &textures[sky.resource_id() as usize].image;
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
    // pipeline
    let mut builder = PipelineBuilder::default();
    let vs = include_shader!("skydome.vert");
    let fs = include_shader!("skydome.frag");
    builder.push_shader(vs, "main", vk::ShaderStageFlags::VERTEX);
    builder.push_shader(fs, "main", vk::ShaderStageFlags::FRAGMENT);
    builder.push_constants(64, vk::ShaderStageFlags::VERTEX);
    builder.binding_descriptions = vec![vk::VertexInputBindingDescription {
        binding: 0,
        stride: (std::mem::size_of::<f32>() * 3) as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    builder.attribute_descriptions = vec![vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    }];
    builder.no_depth();
    builder.rasterizer.cull_mode = vk::CullModeFlags::NONE;
    let pipeline = builder.build(device.logical_clone(), rp, extent, &[descriptor.layout]);
    unf.wait_completion();
    SkydomeDrawData {
        vertices,
        indices,
        descriptor,
        pipeline,
        index_count: si.len() as u32,
    }
}

/// Creates the default sampler for this scene.
/// Uses anisotropic filtering with the max anisotropy supported by the GPU.
/// pass true to create a nearest neighbour filter.
fn create_sampler(device: &Device, nn: bool) -> vk::Sampler {
    let max_anisotropy = device.physical().properties.limits.max_sampler_anisotropy;
    let filter = if nn {
        vk::Filter::NEAREST
    } else {
        vk::Filter::LINEAR
    };
    let ci = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SamplerCreateFlags::empty(),
        mag_filter: filter,
        min_filter: filter,
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
            vk::BufferUsageFlags::TRANSFER_SRC,
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
        let fence = device.submit_immediate(tcmdm, command);
        unfinished.add(fence, cpu_buffer);
    }
    (converted_meshes, gpu_buffer)
}

/// Converts a slice of MeshInstances into a map.
/// The slice is expected to contain One-to-Many relationships.
#[cfg(feature = "vulkan-interactive")]
fn instances_to_map(instances: &[MeshInstance]) -> FnvHashMap<u16, Vec<u16>> {
    use fnv::FnvBuildHasher;
    use std::collections::hash_map::Entry;

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
) -> (MaterialType, Descriptor) {
    use crate::materials::DEFAULT_TEXTURE_ID;

    let mut shader = material.mtype;
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
        shader = material.mtype.two_sided_viewport(); // use a two-sided shader
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
            let mapped_void = mapped as *mut std::ffi::c_void;
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
    let fence = device.submit_immediate(tcmdm, command);
    unfinished.add(fence, cpu_buffer);
    gpu_buffer
}

/// Loads a single texture to the GPU with optimal layout.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
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
    let fence = instance.device().submit_immediate(gcmdm, command);
    unfinished.add(fence, cpu_buf);
    TextureLoaded {
        format: texture.info().format,
        image,
    }
}

// sort mehses by shader id (first) and then material id (second) to minimize binding changes
#[cfg(feature = "vulkan-interactive")]
fn sort_meshes(meshes: &mut [VulkanMesh], mats: &[(MaterialType, Descriptor)]) {
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

/// Material parameters representation used by the shaders in realtime.
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

/// A scene suitable to be rendered by the [RayTraceRenderer](crate::RayTraceRenderer)
pub struct RayTraceScene<T: Instance + Send + Sync> {
    /// camera used in the scene.
    pub(crate) camera: Camera,
    /// scene-bound descriptor.
    pub(crate) descriptor: Descriptor,
    /// linear sampler.
    linear_sampler: vk::Sampler,
    /// nearest neighbour sampler.
    nn_sampler: vk::Sampler,
    /// The buffer containing all vertices for the current scene.
    /// Might be shared with the realtime scene.
    vertex_buffer: Arc<AllocatedBuffer>,
    /// The buffer containing all indices for the current scene. May be shared with realtime scene.
    /// Might be shared with the realtime scene.
    index_buffer: Arc<AllocatedBuffer>,
    /// Buffer containing instances (defined in the RTInstance struct)
    instance_buffer: AllocatedBuffer,
    /// The buffer containing all materials for the current scene. These materials are different
    /// from the one used by the realtime renderer.
    material_buffer: AllocatedBuffer,
    /// The buffer containing all lights for the current scene.
    light_buffer: AllocatedBuffer,
    /// Buffer containing the dpdu and dpdv derivatives for the triangles.
    derivative_buffer: AllocatedBuffer,
    /// Buffer containing the various transformation matrices for each object.
    transforms_buffer: Arc<AllocatedBuffer>,
    /// maps a material to the various instances using it.
    material_instance_ids: FnvHashMap<u16, Vec<u16>>,
    /// Skylight used in this scene.
    sky: Option<Light>,
    /// Buffer containing skylight data. This field is Option::Some even when RayTraceScene::sky is
    /// Option::None. I just need to use std::mem::take.
    sky_buffer: Option<SkyBuffers>,
    /// Number of lights in the scene.
    pub(crate) lights_no: u32,
    /// Additional parameters for the scene.
    pub(crate) meta: Meta,
    /// All textures in the scene.
    /// Might be shared with RayTraceScene.
    textures: Arc<RwLock<Vec<TextureLoaded>>>,
    /// Acceleration structure for the scene.
    acc: SceneAS,
    /// Descriptor manager for the scene.
    dm: DescriptorSetManager,
    /// Underlying vulkan instance.
    instance: Arc<T>,
}

impl<T: Instance + Send + Sync> RayTraceScene<T> {
    const AVG_DESC: [(vk::DescriptorType, f32); 4] = [
        (vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1.0),
        (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
        (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1.0),
        (vk::DescriptorType::STORAGE_BUFFER, 7.0),
    ];

    /// Creates a new scene from a parsed file.
    ///
    /// The scene creates in this way is **NOT** suitable for a realtime raytracer. To obtain a
    /// scene compatible with a realtime raytracer call the method [From] with the original
    /// [RealtimeScene] as parameter.
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
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue(), 1);
        let mut ccmdm = CommandManager::new(device.logical_clone(), device.compute_queue(), 1);
        let instances = parsed.instances().unwrap_or_default();
        let transforms = parsed
            .transforms()
            .unwrap_or_else(|_| vec![Transform::default()]);
        let materials = parsed
            .materials()
            .unwrap_or_else(|_| vec![Material::default()]);
        let lights = reorder_lights(parsed.lights().unwrap_or_default());
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
        let raw_textures = parsed
            .textures()
            .unwrap_or_else(|_| vec![Texture::default()]);
        let textures = Arc::new(RwLock::new(
            raw_textures
                .iter()
                .map(|tex| load_texture_to_gpu(instance.clone(), mm, &mut tcmdm, &mut unf, tex))
                .collect::<Vec<_>>(),
        ));
        let material_instance_ids = map_materials_to_instances(&meshes, &instances);
        let builder = SceneASBuilder::new(
            instance.as_ref(),
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
        let linear_sampler = create_sampler(device, false);
        let nn_sampler = create_sampler(device, true);
        let sky = lights
            .last()
            .filter(|x| x.ltype() == LightType::SKY)
            .cloned();
        let sky_buffer = build_sky_raytrace_buffers(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            sky.clone(),
            &raw_textures,
            None,
            None,
        );
        let light_buffer = load_raytrace_lights_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &lights,
            &material_instance_ids,
        );
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
            &sky_buffer,
            &textures.read().unwrap(),
            linear_sampler,
            nn_sampler,
        );
        let meta = parsed.meta().unwrap_or_default();
        unf.wait_completion();
        RayTraceScene {
            camera,
            descriptor,
            linear_sampler,
            nn_sampler,
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            instance_buffer,
            material_buffer,
            light_buffer,
            derivative_buffer,
            transforms_buffer: Arc::new(transforms_buffer),
            material_instance_ids,
            sky,
            sky_buffer: Some(sky_buffer),
            lights_no: lights.len() as u32,
            meta,
            textures,
            acc,
            dm,
            instance,
        }
    }

    /// rebuild the scene descriptor set.
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
            self.sky_buffer.as_ref().unwrap(),
            &self.textures.read().unwrap(),
            self.linear_sampler,
            self.nn_sampler,
        );
    }

    /// update the lights, materials and textures.
    pub(crate) fn update_materials_and_lights(
        &mut self,
        materials: &[Material],
        lights: &[Light],
        textures: &[Texture],
        tcmdm: &mut CommandManager,
        unf: &mut UnfinishedExecutions,
    ) {
        let device = self.instance.device();
        let mm = self.instance.allocator();
        let mut mat_buffer =
            load_raytrace_materials_to_gpu(self.instance.device(), mm, tcmdm, unf, materials);
        let mut light_buffer = load_raytrace_lights_to_gpu(
            device,
            mm,
            tcmdm,
            unf,
            lights,
            &self.material_instance_ids,
        );
        let sky = lights
            .last()
            .filter(|l| l.ltype() == LightType::SKY)
            .cloned();
        // call the update sky function only if it is effectively changed
        // this is a quite expensive function.
        if sky != self.sky {
            let old_sky = std::mem::take(&mut self.sky);
            let old_buffers = std::mem::take(&mut self.sky_buffer);
            let sky_buffer = build_sky_raytrace_buffers(
                device,
                mm,
                tcmdm,
                unf,
                sky.clone(),
                textures,
                old_sky,
                old_buffers,
            );
            self.sky = sky;
            self.sky_buffer = Some(sky_buffer);
        }
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
                .destroy_sampler(self.linear_sampler, None);
            self.instance
                .device()
                .logical()
                .destroy_sampler(self.nn_sampler, None);
        }
    }
}

#[cfg(feature = "vulkan-interactive")]
impl From<&RealtimeScene> for RayTraceScene<PresentInstance> {
    fn from(scene: &RealtimeScene) -> RayTraceScene<PresentInstance> {
        let instance = Arc::clone(&scene.instance);
        let device = instance.device();
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let mm = instance.allocator();
        let mut unf = UnfinishedExecutions::new(instance.device());
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue(), 1);
        let mut ccmdm = CommandManager::new(device.logical_clone(), device.compute_queue(), 11);
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
            instance.as_ref(),
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
        let linear_sampler = create_sampler(device, false);
        let nn_sampler = create_sampler(device, true);
        let sky = scene.skydome();
        let sky_buffer = build_sky_raytrace_buffers(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            sky.clone(),
            &scene.raw_textures,
            None,
            None,
        );
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
            &sky_buffer,
            &textures.read().unwrap(),
            linear_sampler,
            nn_sampler,
        );
        let meta = scene.meta;
        unf.wait_completion();
        RayTraceScene {
            camera: scene.current_cam,
            descriptor,
            linear_sampler,
            nn_sampler,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            material_buffer,
            light_buffer,
            derivative_buffer,
            transforms_buffer,
            material_instance_ids,
            sky,
            sky_buffer: Some(sky_buffer),
            lights_no: scene.lights.len() as u32,
            meta,
            textures,
            acc,
            dm,
            instance,
        }
    }
}

/// generates the map (material, vec<instance id>)
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

/// load the various mesh instances to the gpu.
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

/// load the various raytrace materials to the gpu.
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
                bsdf_index: mat.mtype.sbt_callable_index(),
                roughness_mul: mat.roughness_mul,
                metalness_mul: mat.metalness_mul as f32,
                anisotropy: mat.anisotropy,
                ior_dielectric: mat.ior,
                is_specular: mat.mtype.is_specular() as u32,
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

/// load the various lights to the gpu
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
            log::warn!("zero length vector was changed to (0.0, -1.0, 0.0)");
            dir.y = -1.0;
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
            let material_id = l.resource_id() as u16;
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

/// converts a u8 color to a f32 color
fn col_int_to_f32(col: [u8; 3]) -> [f32; 4] {
    [
        col[0] as f32 / 255.0,
        col[1] as f32 / 255.0,
        col[2] as f32 / 255.0,
        1.0,
    ]
}

/// uploads a buffer, given a slice of objects.
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
        let fence = device.submit_immediate(tcmdm, command);
        unf.add(fence, cpu_buffer);
    } else {
        gpu_buffer = mm.create_buffer("empty_buffer", 1, flags, MemoryLocation::GpuOnly);
    }
    gpu_buffer
}

/// uploads a 2D buffer (as image) given a slice of objects.
fn upload_2d_buffer<T: Sized + 'static>(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    dimensions: (usize, usize),
    format: vk::Format,
    data: &[T],
) -> AllocatedImage {
    let size = (std::mem::size_of::<T>() * data.len()) as u64;
    let extent = vk::Extent2D {
        width: dimensions.0 as u32,
        height: dimensions.1 as u32,
    };
    let cpu_buf = mm.create_buffer(
        "2D Buffer (CPU)",
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let gpu_buf = mm.create_image_gpu(
        "2D Buffer (GPU)",
        format,
        extent,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let mapped = cpu_buf
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len()) };
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
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
        image: gpu_buf.image,
        subresource_range,
    };
    let barrier_use = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: gpu_buf.image,
        subresource_range,
    };
    let image_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let copy_region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,   // 0 = same as image width
        buffer_image_height: 0, // 0 = same as image height
        image_subresource,
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: dimensions.0 as u32,
            height: dimensions.1 as u32,
            depth: 1,
        },
    };
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
                gpu_buf.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
            // prepare the texture for shader reading
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_use],
            );
        }
    };
    let fence = device.submit_immediate(tcmdm, command);
    unfinished.add(fence, cpu_buf);
    gpu_buf
}

/// calculate the derivatives using the GPU and a comp shader.
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
        let fence = device.submit_immediate(ccmdm, command);
        unf.add_pipeline_execution(fence, pipeline);
    }
    buffer
}

/// calculate the distributions used for efficient skymap sampling.
fn calculate_skymap_distributions(map: &Texture) -> Distribution2D<f32> {
    let raw = map.raw(0);
    let pixel_size = map.bytes_per_pixel();
    assert!(pixel_size >= 3);
    let (width, height) = map.dimensions(0);
    let values = raw
        .chunks_exact(pixel_size * width as usize)
        .enumerate()
        .flat_map(|(y, row)| {
            let sint = f32::sin(PI * (y as f32 + 0.5) / height as f32);
            //TODO: according to PBRT I should filter the pixel lookup
            // however I have no filtering implemented on the CPU side yet and no time to make one
            // so currently is Nearest neighbour
            row.chunks_exact(pixel_size).map(move |pixel| {
                let pixel_rgb: [u8; 3] = pixel[0..3].try_into().unwrap();
                Spectrum::from_rgb(ColorRGB::from(pixel_rgb), true).luminance() * sint
            })
        });
    Distribution2D::new(values, width as usize)
}

/// Buffers required to efficiently draw the skymap in the raytracer.
struct SkyBuffers {
    /// SSBO containing:
    /// - RTSky structure
    /// - amount of marginal cumulative distribution function values
    /// - offset (in elements) of the conditional integrals (the number of integrals is the same of
    /// the amount of marginal values = same as the number of rows per image).
    /// - amount of conditional cumulative distribution function values (the same for each
    /// conditional distribution)
    /// - values of the marginal integral
    /// - array containing all the previous values
    ssbo: AllocatedBuffer,
    /// the conditional values (having chosen an image row of pixels)
    conditional_values: AllocatedImage,
    /// the conditional cumulative distribution function (having chosen an image row of pixels)
    conditional_cdf: AllocatedImage,
}

/// build the buffers required to draw the skylight in the raytrace renderer.
#[allow(clippy::unnecessary_unwrap)] // false positive in this function
fn build_sky_raytrace_buffers(
    device: &Device,
    mm: &MemoryManager,
    tcmdm: &mut CommandManager,
    unf: &mut UnfinishedExecutions,
    sky: Option<Light>,
    textures: &[Texture],
    old_sky: Option<Light>,
    old_buffer: Option<SkyBuffers>,
) -> SkyBuffers {
    // default does not return skylight, but everything else should succeed.
    let skylight = sky.unwrap_or_default();
    let rtlight = RTSky {
        obj2world: skylight.rotation_matrix(),
        world2obj: skylight.rotation_matrix().invert().unwrap(),
        tex_id: skylight.resource_id(),
    };
    if old_buffer.is_none()
        || old_sky.is_none()
        || old_sky.unwrap().resource_id() != skylight.resource_id()
    {
        // recalculate the entire distribution of the texture map
        let distribution = calculate_skymap_distributions(&textures[rtlight.tex_id as usize]);
        let values = distribution
            .conditional()
            .iter()
            .flat_map(Distribution1D::values)
            .copied()
            .collect::<Vec<_>>();
        let cdfs = distribution
            .conditional()
            .iter()
            .flat_map(Distribution1D::cdf)
            .copied()
            .collect::<Vec<_>>();
        let conditional_values = upload_2d_buffer(
            device,
            mm,
            tcmdm,
            unf,
            distribution.dimensions_values(),
            vk::Format::R32_SFLOAT,
            &values,
        );
        let conditional_cdf = upload_2d_buffer(
            device,
            mm,
            tcmdm,
            unf,
            distribution.dimensions_cdf(),
            vk::Format::R32_SFLOAT,
            &cdfs,
        );
        let mut ssbo_data = unsafe { as_u8_slice(&rtlight) }.to_vec();
        let marginal_values = distribution
            .marginal()
            .values()
            .iter()
            .flat_map(|x| x.to_le_bytes());
        let marginal_cdf = distribution
            .marginal()
            .cdf()
            .iter()
            .flat_map(|x| x.to_le_bytes());
        let marginal_integral = distribution.marginal().integral();
        let conditional_integrals = distribution
            .conditional()
            .iter()
            .map(Distribution1D::integral)
            .flat_map(|x| x.to_le_bytes());
        let marginal_cdf_count = distribution.marginal().cdf().len() as u32;
        let conditional_integrals_offset =
            (distribution.marginal().values().len() + distribution.marginal().cdf().len()) as u32;
        let conditional_cdf_count = distribution.conditional().first().unwrap().cdf().len() as u32;
        ssbo_data.extend_from_slice(&marginal_cdf_count.to_le_bytes());
        ssbo_data.extend_from_slice(&conditional_integrals_offset.to_le_bytes());
        ssbo_data.extend_from_slice(&conditional_cdf_count.to_le_bytes());
        ssbo_data.extend_from_slice(&marginal_integral.to_le_bytes());
        ssbo_data.extend(marginal_cdf);
        ssbo_data.extend(marginal_values);
        ssbo_data.extend(conditional_integrals);
        let cpu_buffer = mm.create_buffer(
            "Sky SSBO CPU",
            ssbo_data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        let gpu_buffer = mm.create_buffer(
            "Sky SSBO GPU",
            ssbo_data.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        );
        let mapped = cpu_buffer
            .allocation()
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        unsafe { std::ptr::copy_nonoverlapping(ssbo_data.as_ptr(), mapped, ssbo_data.len()) };
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
        let fence = device.submit_immediate(tcmdm, command);
        unf.add(fence, cpu_buffer);
        SkyBuffers {
            ssbo: gpu_buffer,
            conditional_values,
            conditional_cdf,
        }
    } else {
        // just reuse the old buffer and update just the parameters.
        let ssbo_data = unsafe { as_u8_slice(&rtlight) }.to_vec();
        let retval = old_buffer.unwrap();
        let cpu_buffer = mm.create_buffer(
            "Sky SSBO update CPU",
            ssbo_data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        let mapped = cpu_buffer
            .allocation()
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: ssbo_data.len() as u64,
        };
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_copy_buffer(cmd, cpu_buffer.buffer, retval.ssbo.buffer, &[copy_region]);
            }
        };
        let fence = device.submit_immediate(tcmdm, command);
        unf.add(fence, cpu_buffer);
        unsafe { std::ptr::copy_nonoverlapping(ssbo_data.as_ptr(), mapped, ssbo_data.len()) };
        retval
    }
}

/// build the scene-level descriptor.
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
    sky_buffers: &SkyBuffers,
    textures: &[TextureLoaded],
    linear_sampler: vk::Sampler,
    nn_sampler: vk::Sampler,
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
            linear_sampler,
            vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR
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
        .bind_buffer(
            &sky_buffers.ssbo,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_image(
            &sky_buffers.conditional_values,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            nn_sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .bind_image(
            &sky_buffers.conditional_cdf,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            nn_sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::CALLABLE_KHR,
        )
        .build()
}
