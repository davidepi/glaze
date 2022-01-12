use super::acceleration::{SceneAS, SceneASBuilder};
use super::cmd::CommandManager;
use super::descriptor::{DLayoutCache, Descriptor, DescriptorSetManager};
use super::device::Device;
use super::memory::{AllocatedBuffer, MemoryManager};
use super::pipeline::Pipeline;
use crate::materials::{TextureFormat, TextureLoaded};
use crate::{Camera, Material, Mesh, ParsedScene, ShaderMat, Texture, Vertex};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk;
use cgmath::Vector3 as Vec3;
use fnv::FnvHashMap;
use gpu_allocator::MemoryLocation;
use std::ffi::c_void;
use std::ptr;
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// A scene optimized to be rendered using this crates vulkan implementation.
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
    /// All the meshes in the scene
    pub(super) meshes: Vec<VulkanMesh>,
    /// Generic sampler used for all textures
    sampler: vk::Sampler,
    /// Default texture used when a texture is required but missing. This is a 1x1 white texture.
    dflt_tex: TextureLoaded,
    /// Manages descriptors in the current scene.
    dm: DescriptorSetManager,
    /// Map of all materials in the scene.
    pub(super) materials: FnvHashMap<u16, (Material, ShaderMat, Descriptor)>,
    /// Map of all shaders in the scene with their pipeline.
    pub(super) pipelines: FnvHashMap<ShaderMat, Pipeline>,
    /// Map of all textures in the scene.
    pub(super) textures: FnvHashMap<u16, TextureLoaded>,
    /// If the materials changed after loading the scene
    mat_changed: bool,
}

/// A mesh optimized to be rendered using this crates renderer.
pub struct VulkanMesh {
    /// Offset of the mesh in the scene index buffer.
    pub index_offset: u32,
    /// Number of indices of this mesh in the index buffer.
    pub index_count: u32,
    /// Maximum index value that appears for this mesh.
    pub max_index: u32,
    /// Material id of this mesh.
    pub material: u16,
}

/// Contains the fences for commands run at load time. All these fences are waited on at the end
/// of the scene loading.
struct UnfinishedExecutions {
    /// Fences to be waited on.
    fences: Vec<vk::Fence>,
    /// Buffers that are to be freed after waiting on the fences.
    buffers_to_free: Vec<AllocatedBuffer>,
}

impl UnfinishedExecutions {
    fn wait_completion(self, device: &Device, mm: &mut MemoryManager) {
        device.wait_completion(&self.fences);
        self.buffers_to_free
            .into_iter()
            .for_each(|b| mm.free_buffer(b));
    }
}

impl VulkanScene {
    /// Converts a parsed scene into a vulkan scene.
    /// wchan is used to send feedbacks about the current loading status.
    pub(super) fn load(
        device: &Device,
        mut parsed: Box<dyn ParsedScene + Send>,
        mm: &mut MemoryManager,
        desc_cache: DLayoutCache,
        wchan: Sender<String>,
    ) -> Result<Self, std::io::Error> {
        let mut unf = UnfinishedExecutions {
            fences: Vec::new(),
            buffers_to_free: Vec::new(),
        };
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 5);
        wchan.send("[1/4] Loading vertices...".to_string()).ok();
        let vertex_buffer =
            load_vertices_to_gpu(device, mm, &mut tcmdm, &mut unf, &parsed.vertices()?);
        wchan.send("[2/4] Loading meshes...".to_string()).ok();
        let (mut meshes, index_buffer) =
            load_indices_to_gpu(device, mm, &mut tcmdm, &mut unf, &parsed.meshes()?);
        let sampler = create_sampler(device);
        wchan.send("[3/4] Loading textures...".to_string()).ok();
        let scene_textures = parsed.textures()?;
        let textures_no = scene_textures.len();
        let textures = scene_textures
            .into_iter()
            .enumerate()
            .map(|(idx, (id, tex))| {
                wchan
                    .send(format!(
                        "[3/4] Loading textures... ({}/{})",
                        idx + 1,
                        textures_no
                    ))
                    .ok();
                (
                    id,
                    load_texture_to_gpu(device, mm, &mut tcmdm, &mut unf, tex),
                )
            })
            .collect();
        let dflt_tex = load_texture_to_gpu(device, mm, &mut tcmdm, &mut unf, Texture::default());
        let parsed_mats = parsed.materials()?;
        let params_buffer =
            load_materials_parameters(device, &parsed_mats, mm, &mut tcmdm, &mut unf);
        unf.wait_completion(device, mm);
        wchan.send("[4/4] Loading materials...".to_string()).ok();
        let avg_desc = [
            (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
            (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1.5),
        ];
        let mut dm =
            DescriptorSetManager::with_cache(device.logical_clone(), &avg_desc, desc_cache);
        let materials = parsed_mats
            .into_iter()
            .map(|(id, mat)| {
                let (shader, desc) = build_mat_desc_set(
                    device,
                    (&textures, &dflt_tex),
                    &params_buffer,
                    sampler,
                    id,
                    &mat,
                    &mut dm,
                );
                (id, (mat, shader, desc))
            })
            .collect();
        let current_cam = parsed.cameras()?[0].clone(); // parser automatically adds a default cam
        let pipelines = FnvHashMap::default();
        let update_buffer = mm.create_buffer(
            "Material update transfer buffer",
            std::mem::size_of::<MaterialParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        sort_meshes(&mut meshes, &materials);
        Ok(VulkanScene {
            file: parsed,
            current_cam,
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            update_buffer,
            params_buffer,
            meshes,
            sampler,
            dflt_tex,
            dm,
            materials,
            pipelines,
            textures,
            mat_changed: false,
        })
    }

    /// Updates (changes) a single material in the scene.
    pub(super) fn update_material(
        &mut self,
        device: &Device,
        mat_id: u16,
        new: Material,
        gcmdm: &mut CommandManager,
        rpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
        render_size: vk::Extent2D,
    ) {
        // setup the new descriptor
        let params = MaterialParams::from(&new);
        let mapped = self
            .update_buffer
            .allocation
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
        let cmd = gcmdm.get_cmd_buffer();
        let queue = device.graphic_queue();
        let fence = device.immediate_execute(cmd, queue, command);
        device.wait_completion(&[fence]);
        let (new_shader, new_desc) = build_mat_desc_set(
            device,
            (&self.textures, &self.dflt_tex),
            &self.params_buffer,
            self.sampler,
            mat_id,
            &new,
            &mut self.dm,
        );
        // build the new shader if not existing
        self.pipelines.entry(new_shader).or_insert_with(|| {
            new.shader.build_pipeline().build(
                device.logical(),
                rpass,
                render_size,
                &[frame_desc_layout, new_desc.layout],
            )
        });
        // insert the new material
        self.materials.insert(mat_id, (new, new_shader, new_desc));
        // sort the meshes to minimize bindings
        sort_meshes(&mut self.meshes, &self.materials);
        self.mat_changed = true;
    }

    /// Initializes the scene's pipelines.
    pub(super) fn init_pipelines(
        &mut self,
        render_size: vk::Extent2D,
        device: &ash::Device,
        renderpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
    ) {
        self.pipelines = FnvHashMap::default();
        for (_, shader, desc) in self.materials.values() {
            self.pipelines.entry(*shader).or_insert_with(|| {
                shader.build_pipeline().build(
                    device,
                    renderpass,
                    render_size,
                    &[frame_desc_layout, desc.layout],
                )
            });
        }
    }

    /// Destroys the scene's pipelines.
    pub(super) fn deinit_pipelines(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.pipelines.drain() {
            pipeline.destroy(device);
        }
    }

    /// Unloads the scene from the GPU memory
    pub(super) fn unload(self, device: &Device, mm: &mut MemoryManager) {
        self.textures
            .into_iter()
            .chain([(u16::MAX, self.dflt_tex)])
            .for_each(|(_, tex)| mm.free_image(tex.image));
        unsafe { device.logical().destroy_sampler(self.sampler, None) };
        if let Ok(last_buffer) = Arc::try_unwrap(self.vertex_buffer) {
            mm.free_buffer(last_buffer);
        }
        if let Ok(last_buffer) = Arc::try_unwrap(self.index_buffer) {
            mm.free_buffer(last_buffer)
        }
        mm.free_buffer(self.params_buffer);
        mm.free_buffer(self.update_buffer);
    }

    /// Returns a material in the scene, given its ID.
    /// Returns None if the material does not exist.
    pub fn single_material(&self, id: u16) -> Option<&Material> {
        self.materials.get(&id).map(|(mat, _, _)| mat)
    }

    /// Returns all the materials in the scene.
    /// Each returned material is a tuple containing the material ID and the material itself.
    /// The order of the materials is not guaranteed.
    pub fn materials(&self) -> Vec<(u16, &Material)> {
        self.materials
            .iter()
            .map(|(id, (mat, _, _))| (*id, mat))
            .collect()
    }

    //// Returns a texture in the scene, given its ID.
    /// Returns None if the texture does not exists.
    pub fn single_texture(&self, id: u16) -> Option<&TextureLoaded> {
        self.textures.get(&id)
    }

    /// Returns all the textures in the scene.
    /// Each returned texture is a tuple containing the texture ID and the texture itself.
    /// The order of the textures is not guaranteed.
    pub fn textures(&self) -> Vec<(u16, &TextureLoaded)> {
        self.textures.iter().map(|(id, tex)| (*id, tex)).collect()
    }

    pub fn save(&mut self) -> Result<(), std::io::Error> {
        let cameras = [self.current_cam.clone()];
        if self.mat_changed {
            let materials = self
                .materials
                .iter()
                .map(|(id, (mat, _, _))| (*id, mat.clone()))
                .collect::<Vec<_>>();
            self.file.update(Some(&cameras), Some(&materials))
        } else {
            self.file.update(Some(&cameras), None)
        }
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
    mm: &mut MemoryManager,
    tcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    vertices: &[Vertex],
) -> AllocatedBuffer {
    let size = (std::mem::size_of::<Vertex>() * vertices.len()) as u64;
    let cpu_buffer = mm.create_buffer(
        "vertices_local",
        size,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let gpu_buffer = mm.create_buffer(
        "vertices_dedicated",
        size,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
    );
    let mapped = cpu_buffer
        .allocation
        .mapped_ptr()
        .expect("Faield to map memory")
        .cast()
        .as_ptr();
    unsafe { std::ptr::copy_nonoverlapping(vertices.as_ptr(), mapped, vertices.len()) };
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
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(cpu_buffer);
    gpu_buffer
}

/// Loads all indices to GPU.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
/// Returns the list of meshes and the index buffer.
fn load_indices_to_gpu(
    device: &Device,
    mm: &mut MemoryManager,
    tcmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    meshes: &[Mesh],
) -> (Vec<VulkanMesh>, AllocatedBuffer) {
    let size =
        (std::mem::size_of::<u32>() * meshes.iter().map(|m| m.indices.len()).sum::<usize>()) as u64;
    let cpu_buffer = mm.create_buffer(
        "indices_local",
        size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let gpu_buffer = mm.create_buffer(
        "indices_dedicated",
        size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
    );
    let mut converted_meshes = Vec::with_capacity(meshes.len());
    let mut mapped = cpu_buffer
        .allocation
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
        converted_meshes.push(VulkanMesh {
            index_offset: offset,
            index_count: mesh.indices.len() as u32,
            max_index,
            material: mesh.material,
        });
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
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(cpu_buffer);
    (converted_meshes, gpu_buffer)
}

/// Builds a single material descriptor set.
fn build_mat_desc_set(
    device: &Device,
    (textures, dflt_tex): (&FnvHashMap<u16, TextureLoaded>, &TextureLoaded),
    params: &AllocatedBuffer,
    sampler: vk::Sampler,
    id: u16,
    material: &Material,
    dm: &mut DescriptorSetManager,
) -> (ShaderMat, Descriptor) {
    let mut shader = material.shader;
    let diffuse = if let Some(diff_id) = material.diffuse {
        textures.get(&diff_id).unwrap_or(dflt_tex)
    } else {
        dflt_tex
    };
    let diffuse_image_info = vk::DescriptorImageInfo {
        sampler,
        image_view: diffuse.image.image_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
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
        .bind_buffer(
            buf_info,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::FRAGMENT,
        )
        .bind_image(
            diffuse_image_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        );
    if let Some(op_id) = material.opacity {
        shader = material.shader.two_sided(); // use a two-sided shader
        let opacity = textures.get(&op_id).unwrap_or(dflt_tex);
        let opacity_image_info = vk::DescriptorImageInfo {
            sampler,
            image_view: opacity.image.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        descriptor = descriptor.bind_image(
            opacity_image_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        );
    }
    (shader, descriptor.build())
}

/// Loads all materials parameters to GPU.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
fn load_materials_parameters(
    device: &Device,
    materials: &[(u16, Material)],
    mm: &mut MemoryManager,
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
        .allocation
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    for (_, mat) in materials {
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
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(cpu_buffer);
    gpu_buffer
}

/// Loads all textures to the GPU with optimal layout.
/// Updates the UnfinishedExecutions with the buffers to free and fences to wait on.
fn load_texture_to_gpu(
    device: &Device,
    mm: &mut MemoryManager,
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
    let buffer = mm.create_buffer(
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
    let mut mapped = buffer
        .allocation
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
                buffer.buffer,
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
    let transfer_queue = device.transfer_queue();
    let fence = device.immediate_execute(cmd, transfer_queue, command);
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(buffer);
    TextureLoaded {
        info: texture.to_info(),
        image,
    }
}

// sort mehses by shader id (first) and then material id (second) to minimize binding changes
fn sort_meshes(
    meshes: &mut Vec<VulkanMesh>,
    mats: &FnvHashMap<u16, (Material, ShaderMat, Descriptor)>,
) {
    meshes.sort_unstable_by(|a, b| {
        let (_, _, desc_a) = mats.get(&a.material).unwrap();
        let (_, _, desc_b) = mats.get(&b.material).unwrap();
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
fn padding<T: Into<u64>>(n: T, align: T) -> u64 {
    ((!n.into()).wrapping_add(1)) & (align.into().wrapping_sub(1))
}

pub struct RayTraceScene {
    vertex_buffer: Arc<AllocatedBuffer>,
    index_buffer: Arc<AllocatedBuffer>,
    acc: SceneAS,
}

impl RayTraceScene {
    pub fn new(
        device: &Device,
        loader: &AccelerationLoader,
        mut scene: Box<dyn ParsedScene>,
        mm: &mut MemoryManager,
        ccmdm: &mut CommandManager,
    ) -> Result<Self, std::io::Error> {
        let mut unf = UnfinishedExecutions {
            fences: Vec::new(),
            buffers_to_free: Vec::new(),
        };
        let mut tcmdm = CommandManager::new(device.logical_clone(), device.transfer_queue().idx, 1);
        let vertex_buffer =
            load_vertices_to_gpu(device, mm, &mut tcmdm, &mut unf, &scene.vertices()?);
        let (meshes, index_buffer) =
            load_indices_to_gpu(device, mm, &mut tcmdm, &mut unf, &scene.meshes()?);
        unf.wait_completion(device, mm);
        let mut builder =
            SceneASBuilder::new(device, mm, ccmdm, &vertex_buffer, &index_buffer, loader);
        meshes.iter().for_each(|m| builder.add_mesh(m));
        let acc = builder.build();
        Ok(Self {
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            acc,
        })
    }

    pub fn from(
        device: &Device,
        loader: &AccelerationLoader,
        scene: &VulkanScene,
        mm: &mut MemoryManager,
        ccmdm: &mut CommandManager,
    ) -> Self {
        let vertex_buffer = scene.vertex_buffer.clone();
        let index_buffer = scene.index_buffer.clone();
        let mut builder =
            SceneASBuilder::new(device, mm, ccmdm, &vertex_buffer, &index_buffer, loader);
        scene.meshes.iter().for_each(|m| builder.add_mesh(m));
        let acc = builder.build();
        Self {
            vertex_buffer,
            index_buffer,
            acc,
        }
    }

    pub fn destroy(self, loader: &AccelerationLoader, mm: &mut MemoryManager) {
        if let Ok(last_buffer) = Arc::try_unwrap(self.vertex_buffer) {
            mm.free_buffer(last_buffer);
        }
        if let Ok(last_buffer) = Arc::try_unwrap(self.index_buffer) {
            mm.free_buffer(last_buffer);
        }
        self.acc.destroy(loader, mm);
    }
}
