use super::acceleration::{SceneAS, SceneASBuilder};
use super::cmd::CommandManager;
#[cfg(feature = "vulkan-interactive")]
use super::descriptor::{Descriptor, DescriptorSetManager};
use super::device::Device;
use super::instance::Instance;
use super::memory::{AllocatedBuffer, MemoryManager};
#[cfg(feature = "vulkan-interactive")]
use super::pipeline::Pipeline;
use super::UnfinishedExecutions;
#[cfg(feature = "vulkan-interactive")]
use crate::materials::{TextureFormat, TextureLoaded};
use crate::{
    Camera, Light, LightType, Material, Mesh, MeshInstance, ParsedScene, RayTraceInstance, Vertex,
};
#[cfg(feature = "vulkan-interactive")]
use crate::{PresentInstance, ShaderMat, Texture, Transform};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk;
use cgmath::Vector3 as Vec3;
#[cfg(feature = "vulkan-interactive")]
use fnv::{FnvBuildHasher, FnvHashMap};
use gpu_allocator::MemoryLocation;
#[cfg(feature = "vulkan-interactive")]
use std::collections::hash_map::Entry;
#[cfg(feature = "vulkan-interactive")]
use std::ffi::c_void;
#[cfg(feature = "vulkan-interactive")]
use std::ptr;
#[cfg(feature = "vulkan-interactive")]
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// A scene optimized to be rendered using this crates vulkan implementation.
#[cfg(feature = "vulkan-interactive")]
pub struct VulkanScene {
    /// The scene on disk.
    file: Box<dyn ParsedScene + Send>,
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
    pub(super) textures: Arc<Vec<TextureLoaded>>,
    /// All the transforms in the scene.
    pub(super) transforms: Vec<(AllocatedBuffer, Descriptor)>,
    /// All the instances in the scene in form (Mesh ID, Vec<Transform ID>).
    pub(super) instances: FnvHashMap<u16, Vec<u16>>,
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
    pub fn load(
        instance: Arc<PresentInstance>,
        mut parsed: Box<dyn ParsedScene + Send>,
        wchan: Sender<String>,
    ) -> Result<Self, std::io::Error> {
        let device = instance.device();
        let mm = instance.allocator();
        let with_raytrace = instance.supports_raytrace();
        let mut unf = UnfinishedExecutions::new(device);
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 5);
        let avg_desc = [
            (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1.5),
        ];
        let mut dm = DescriptorSetManager::new(
            device.logical_clone(),
            &avg_desc,
            instance.desc_layout_cache(),
        );
        wchan.send("[1/4] Loading vertices...".to_string()).ok();
        let vertex_buffer = load_vertices_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed.vertices()?,
            with_raytrace,
        );
        wchan.send("[2/4] Loading meshes...".to_string()).ok();
        let transforms = load_transforms_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut dm,
            &mut unf,
            &parsed.transforms()?,
        );
        let (mut meshes, index_buffer) = load_indices_to_gpu(
            device,
            mm,
            &mut tcmdm,
            &mut unf,
            &parsed.meshes()?,
            with_raytrace,
        );
        let instances = instances_to_map(&parsed.instances()?);
        wchan.send("[3/4] Loading textures...".to_string()).ok();
        let sampler = create_sampler(device);
        let scene_textures = parsed.textures()?;
        let textures = Arc::new(
            scene_textures
                .into_iter()
                .map(|tex| load_texture_to_gpu(instance.clone(), mm, &mut tcmdm, &mut unf, tex))
                .collect::<Vec<_>>(),
        );
        let materials = parsed.materials()?;
        let params_buffer = load_materials_parameters(device, &materials, mm, &mut tcmdm, &mut unf);
        unf.wait_completion();
        wchan.send("[4/4] Loading materials...".to_string()).ok();
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
                    &mat,
                    &mut dm,
                );
                (shader, desc)
            })
            .collect::<Vec<_>>();
        let current_cam = parsed.cameras()?[0].clone(); // parser automatically adds a default cam
        let lights = parsed.lights()?;
        let pipelines = FnvHashMap::default();
        let update_buffer = mm.create_buffer(
            "Material update transfer buffer",
            std::mem::size_of::<MaterialParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        sort_meshes(&mut meshes, &materials_desc);
        Ok(VulkanScene {
            file: parsed,
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
            textures,
            transforms,
            instances,
            lights,
            instance,
        })
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
            &self.textures,
            &self.params_buffer,
            self.sampler,
            mat_id,
            &new,
            &mut self.dm,
        );
        // build the new shader if not existing
        self.pipelines.entry(new_shader).or_insert_with(|| {
            new.shader.build_pipeline().build(
                device.logical_clone(),
                rpass,
                render_size,
                &[frame_desc_layout, new_desc.layout],
            )
        });
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
        device: Arc<ash::Device>,
        renderpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
    ) {
        self.pipelines = FnvHashMap::default();
        for (shader, desc) in &self.materials_desc {
            let device = device.clone();
            self.pipelines.entry(*shader).or_insert_with(|| {
                shader.build_pipeline().build(
                    device,
                    renderpass,
                    render_size,
                    &[
                        frame_desc_layout,
                        desc.layout,
                        self.transforms[0].1.layout, // pick the first per-object transform
                    ],
                )
            });
        }
    }

    /// Destroys the scene's pipelines.
    pub(super) fn deinit_pipelines(&mut self) {
        self.pipelines.clear();
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

    //// Returns a texture in the scene, given its ID.
    /// Returns None if the texture does not exists.
    pub fn single_texture(&self, id: u16) -> Option<&TextureLoaded> {
        self.textures.get(id as usize)
    }

    /// Returns all the textures in the scene.
    /// The position in the array is the texture ID.
    pub fn textures(&self) -> &[TextureLoaded] {
        &self.textures
    }

    /// Returns all the lights in the scene.
    pub fn lights(&self) -> &[Light] {
        &self.lights
    }

    /// Adds a light to the scene.
    pub(super) fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    /// Replaces an existing light with the one passed as input.
    /// Does nothing if the index does not exist.
    pub(super) fn update_light(&mut self, index: usize, light: Light) {
        if index < self.lights().len() {
            self.lights[index] = light;
        }
    }

    /// Removes a light at a specific index.
    /// This has complexity O(n) in the number of lights.
    ///
    /// Does nothing if the light does not exist.
    pub fn remove_light(&mut self, index: usize) {
        if index < self.lights.len() {
            self.lights.remove(index);
        }
    }

    pub fn save(&mut self) -> Result<(), std::io::Error> {
        let cameras = [self.current_cam.clone()];
        let lights = self.lights().to_vec();
        let materials = self.materials().to_vec();
        self.file
            .update(Some(&cameras), Some(&materials), Some(&lights))
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
    dm: &mut DescriptorSetManager,
    unf: &mut UnfinishedExecutions,
    transforms: &[Transform],
) -> Vec<(AllocatedBuffer, Descriptor)> {
    transforms
        .iter()
        .map(|transform| {
            //TODO: maybe a single buffer would be better, considering that I will probably never edit
            // transforms. I should also consider removing the FnvHashMap and use a Vec given that I am
            // the one assigning indices to the transforms so I know for sure they are contiguous.
            let gpu_buffer = upload_buffer(
                device,
                mm,
                tcmdm,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                unf,
                &[transform.clone()],
            );
            let desc = dm.new_set().bind_buffer(
                &gpu_buffer,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            );
            (gpu_buffer, desc.build())
        })
        .collect()
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
    let size =
        (std::mem::size_of::<u32>() * meshes.iter().map(|m| m.indices.len()).sum::<usize>()) as u64;
    let cpu_buffer = mm.create_buffer(
        "indices_local",
        size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
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
    let mut converted_meshes = Vec::with_capacity(meshes.len());
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
        shader = material.shader.two_sided(); // use a two-sided shader
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
    tcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    texture: Texture,
) -> TextureLoaded {
    let (width, height) = texture.dimensions(0);
    let mip_levels = texture.mipmap_levels();
    let full_size = (0..mip_levels)
        .map(|x| texture.size_bytes(x))
        .sum::<usize>();
    let extent = vk::Extent2D {
        width: width as u32,
        height: height as u32,
    };
    let vkformat = match texture.format() {
        TextureFormat::Gray => vk::Format::R8_UNORM,
        TextureFormat::Rgba => vk::Format::R8G8B8A8_SRGB,
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
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
        mip_levels as u32,
    );
    let mut mapped = cpu_buf
        .allocation()
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    for level in 0..mip_levels {
        let size = texture.size_bytes(level);
        unsafe {
            std::ptr::copy_nonoverlapping(texture.ptr(level), mapped, size);
            mapped = mapped.add(size);
        }
    }
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
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
        image: image.image,
        subresource_range,
    };
    let mut regions = Vec::with_capacity(mip_levels);
    let mut buffer_offset = 0;
    for level in 0..mip_levels {
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
    let cmd = tcmdm.get_cmd_buffer();
    let device = instance.device();
    let transfer_queue = device.transfer_queue();
    let fence = device.immediate_execute(cmd, transfer_queue, command);
    unfinished.add(fence, cpu_buf);
    TextureLoaded {
        info: texture.to_info(),
        image,
        instance: instance.clone(),
    }
}

// sort mehses by shader id (first) and then material id (second) to minimize binding changes
#[cfg(feature = "vulkan-interactive")]
fn sort_meshes(meshes: &mut Vec<VulkanMesh>, mats: &[(ShaderMat, Descriptor)]) {
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

/// This is the struct passed to the GPU that represents a mesh instance.
/// Used to retrieve the mesh attributes in a raytracing context.
/// Unlike MeshInstance and VulkanMesh it MUST be indexed by its position in the array.
/// The index must correspond to the MeshInstance index passed to the acceleration structure.
#[repr(C)]
struct RTInstance {
    index_offset: u32,
    index_count: u32,
    material_id: u32, // due to how std430 works
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct RTMaterial {
    diffuse_mul: [f32; 4],
    diffuse: u32,
    opacity: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct RTLight {
    // w is the type, xyz is either position or direction
    typeposdir: [f32; 4],
    color0: [f32; 4],
    color1: [f32; 4],
    color2: [f32; 4],
    color3: [f32; 4],
}

pub struct RayTraceScene<T: Instance + Send + Sync> {
    pub camera: Camera,
    pub descriptor: Descriptor,
    sampler: vk::Sampler,
    vertex_buffer: Arc<AllocatedBuffer>,
    index_buffer: Arc<AllocatedBuffer>,
    instance_buffer: AllocatedBuffer,
    material_buffer: AllocatedBuffer,
    light_buffer: AllocatedBuffer,
    textures: Arc<Vec<TextureLoaded>>,
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
        loader: Arc<AccelerationLoader>,
        mut scene: Box<dyn ParsedScene>,
        ccmdm: &mut CommandManager,
    ) -> Result<RayTraceScene<RayTraceInstance>, std::io::Error> {
        let mm = instance.allocator();
        let device = instance.device();
        let mut unf = UnfinishedExecutions::new(instance.device());
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 1);

        let instances = scene.instances()?;
        let transforms = scene.transforms()?;
        let materials = scene.materials()?;
        let lights = scene.lights()?;
        let vertex_buffer =
            load_vertices_to_gpu(device, mm, &mut tcmdm, &mut unf, &scene.vertices()?, true);
        let (meshes, index_buffer) =
            load_indices_to_gpu(device, mm, &mut tcmdm, &mut unf, &scene.meshes()?, true);
        let textures = Arc::new(
            scene
                .textures()?
                .into_iter()
                .map(|tex| load_texture_to_gpu(instance.clone(), mm, &mut tcmdm, &mut unf, tex))
                .collect::<Vec<_>>(),
        );
        let builder = SceneASBuilder::new(device, loader, mm, ccmdm, &vertex_buffer, &index_buffer)
            .with_meshes(&meshes, &instances, &transforms);
        let acc = builder.build();
        let camera = scene.cameras()?[0].clone();
        let instance_buffer =
            load_raytrace_instances_to_gpu(device, mm, &mut tcmdm, &mut unf, &meshes, &instances);
        let material_buffer =
            load_raytrace_materials_to_gpu(device, mm, &mut tcmdm, &mut unf, &materials);
        let mut dm = DescriptorSetManager::new(
            instance.device().logical_clone(),
            &Self::AVG_DESC,
            instance.desc_layout_cache(),
        );
        let light_buffer = load_raytrace_lights_to_gpu(device, mm, &mut tcmdm, &mut unf, &lights);
        let sampler = create_sampler(device);
        let descriptor = build_raytrace_descriptor(
            &mut dm,
            &acc,
            &vertex_buffer,
            &index_buffer,
            &instance_buffer,
            &material_buffer,
            &light_buffer,
            &textures,
            sampler,
        );
        unf.wait_completion();
        Ok(RayTraceScene {
            camera,
            descriptor,
            sampler,
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            instance_buffer,
            material_buffer,
            light_buffer,
            textures,
            acc,
            dm,
            instance,
        })
    }

    #[cfg(feature = "vulkan-interactive")]
    pub fn from(
        loader: Arc<AccelerationLoader>,
        scene: &mut VulkanScene,
        ccmdm: &mut CommandManager,
    ) -> Result<RayTraceScene<PresentInstance>, std::io::Error> {
        let instance = Arc::clone(&scene.instance);
        let mm = instance.allocator();
        let device = instance.device();
        let mut unf = UnfinishedExecutions::new(instance.device());
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 1);

        let instances = scene.file.instances()?;
        let transforms = scene.file.transforms()?;
        let materials = scene.materials().into_iter().cloned().collect::<Vec<_>>();
        let vertex_buffer = Arc::clone(&scene.vertex_buffer);
        let index_buffer = Arc::clone(&scene.index_buffer);
        let textures = Arc::clone(&scene.textures);
        let builder = SceneASBuilder::new(device, loader, mm, ccmdm, &vertex_buffer, &index_buffer)
            .with_meshes(&scene.meshes, &instances, &transforms);
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
        let light_buffer =
            load_raytrace_lights_to_gpu(device, mm, &mut tcmdm, &mut unf, &scene.lights);
        let mut dm = DescriptorSetManager::new(
            instance.device().logical_clone(),
            &Self::AVG_DESC,
            instance.desc_layout_cache(),
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
            &textures,
            sampler,
        );
        unf.wait_completion();
        Ok(RayTraceScene {
            camera: scene.current_cam.clone(),
            descriptor,
            sampler,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            material_buffer,
            light_buffer,
            textures,
            acc,
            dm,
            instance,
        })
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(super) fn update_materials(
        &mut self,
        materials: &[Material],
        tcmdm: &mut CommandManager,
        unf: &mut UnfinishedExecutions,
    ) {
        let mm = self.instance.allocator();
        let mut new_buffer =
            load_raytrace_materials_to_gpu(self.instance.device(), mm, tcmdm, unf, materials);
        // cannot drop yet, the loading is not finished yet
        std::mem::swap(&mut self.material_buffer, &mut new_buffer);
        unf.add_buffer(new_buffer);
        self.descriptor = build_raytrace_descriptor(
            &mut self.dm,
            &self.acc,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.instance_buffer,
            &self.material_buffer,
            &self.light_buffer,
            &self.textures,
            self.sampler,
        );
    }

    #[cfg(feature = "vulkan-interactive")]
    pub(super) fn update_lights(
        &mut self,
        lights: &[Light],
        tcmdm: &mut CommandManager,
        unf: &mut UnfinishedExecutions,
    ) {
        let mm = self.instance.allocator();
        let mut new_buffer =
            load_raytrace_lights_to_gpu(self.instance.device(), mm, tcmdm, unf, lights);
        // cannot drop yet, the loading is not finished yet
        std::mem::swap(&mut self.light_buffer, &mut new_buffer);
        unf.add_buffer(new_buffer);
        self.descriptor = build_raytrace_descriptor(
            &mut self.dm,
            &self.acc,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.instance_buffer,
            &self.material_buffer,
            &self.light_buffer,
            &self.textures,
            self.sampler,
        );
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
        .map(|mat| RTMaterial {
            diffuse_mul: col_int_to_f32(mat.diffuse_mul),
            diffuse: mat.diffuse as u32,
            opacity: mat.opacity as u32,
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
) -> AllocatedBuffer {
    let mut data = lights
        .iter()
        .map(|l| {
            let w = l.emission().wavelength;
            RTLight {
                typeposdir: match l.ltype() {
                    LightType::OMNI => {
                        let pos = l.position();
                        [pos.x, pos.y, pos.z, 0.0]
                    }
                    LightType::SUN => {
                        let dir = l.direction();
                        [dir.x, dir.y, dir.z, 1.0]
                    }
                },
                color0: [w[0], w[1], w[2], w[3]],
                color1: [w[4], w[5], w[6], w[7]],
                color2: [w[8], w[9], w[10], w[11]],
                color3: [w[12], w[13], w[14], w[15]],
            }
        })
        .collect::<Vec<_>>();
    if data.is_empty() {
        // push an empty light to avoid having a zero sized buffer
        // since there is no "default" light
        data.push(RTLight {
            typeposdir: [0.0; 4],
            color0: [0.0; 4],
            color1: [0.0; 4],
            color2: [0.0; 4],
            color3: [0.0; 4],
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

fn col_int_to_f32(col: [u8; 4]) -> [f32; 4] {
    [
        col[0] as f32 / 255.0,
        col[1] as f32 / 255.0,
        col[2] as f32 / 255.0,
        col[3] as f32 / 255.0,
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
    let cpu_buffer = mm.create_buffer(
        "scene_buffer_local",
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let gpu_buffer = mm.create_buffer(
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
    gpu_buffer
}

fn build_raytrace_descriptor(
    dm: &mut DescriptorSetManager,
    acc: &SceneAS,
    vertex_buffer: &AllocatedBuffer,
    index_buffer: &AllocatedBuffer,
    instance_buffer: &AllocatedBuffer,
    material_buffer: &AllocatedBuffer,
    light_buffer: &AllocatedBuffer,
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
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            index_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            instance_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            material_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_buffer(
            light_buffer,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .bind_image_array(
            &textures_memory,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        )
        .build()
}
