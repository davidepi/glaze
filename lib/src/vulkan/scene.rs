use std::ptr;

use super::cmd::CommandManager;
use super::descriptor::{Descriptor, DescriptorSetCreator};
use super::device::Device;
use super::memory::{AllocatedBuffer, MemoryManager};
use super::pipeline::Pipeline;
use crate::materials::{TextureFormat, TextureLoaded};
use crate::{Camera, Material, Mesh, ReadParsed, ShaderMat, Texture, Vertex};
use ash::vk;
use cgmath::Vector3 as Vec3;
use fnv::FnvHashMap;
use gpu_allocator::MemoryLocation;

pub struct VulkanScene {
    pub current_cam: Camera,
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
    // buffer used during a material update, as a transfer buffer to the GPU
    update_buffer: AllocatedBuffer,
    pub params_buffer: AllocatedBuffer,
    pub meshes: Vec<VulkanMesh>,
    sampler: vk::Sampler,
    pub materials: FnvHashMap<u16, (Material, Descriptor)>,
    pub pipelines: FnvHashMap<ShaderMat, Pipeline>,
    pub textures: FnvHashMap<u16, TextureLoaded>,
}

pub struct VulkanMesh {
    pub index_offset: u32,
    pub index_count: u32,
    pub material: u16,
}

struct UnfinishedExecutions {
    fences: Vec<vk::Fence>,
    buffers_to_free: Vec<AllocatedBuffer>,
}

impl VulkanScene {
    pub fn load<T: Device>(
        device: &mut T,
        mut scene: Box<dyn ReadParsed>,
        mm: &mut MemoryManager,
        cmdm: &mut CommandManager,
        dm: &mut DescriptorSetCreator,
    ) -> Result<Self, std::io::Error> {
        let mut unf = UnfinishedExecutions {
            fences: Vec::new(),
            buffers_to_free: Vec::new(),
        };
        let vertex_buffer = load_vertices_to_gpu(device, mm, cmdm, &mut unf, &scene.vertices()?);
        let (meshes, index_buffer) =
            load_indices_to_gpu(device, mm, cmdm, &mut unf, &scene.meshes()?);
        let sampler = create_sampler(device);
        let textures = scene
            .textures()?
            .into_iter()
            .map(|(id, tex)| (id, load_texture_to_gpu(device, mm, cmdm, &mut unf, tex)))
            .collect();
        let parsed_mats = scene.materials()?;
        let params_buffer = load_materials_parameters(device, &parsed_mats, mm, cmdm, &mut unf);
        device.wait_completion(&unf.fences);
        unf.buffers_to_free
            .into_iter()
            .for_each(|b| mm.free_buffer(b));
        let materials = parsed_mats
            .into_iter()
            .map(|(id, mat)| {
                let descriptor =
                    build_mat_desc_set(&textures, &params_buffer, sampler, id, &mat, dm);
                (id, (mat, descriptor))
            })
            .collect();
        let current_cam = scene.cameras()?[0].clone(); // parser automatically adds a default cam
        let pipelines = FnvHashMap::default();
        let update_buffer = mm.create_buffer(
            "Material update transfer buffer",
            std::mem::size_of::<MaterialParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        Ok(VulkanScene {
            current_cam,
            vertex_buffer,
            index_buffer,
            update_buffer,
            params_buffer,
            meshes,
            sampler,
            materials,
            pipelines,
            textures,
        })
    }

    pub(crate) fn update_material<T: Device>(
        &mut self,
        device: &mut T,
        old: u16,
        new: Material,
        cmdm: &mut CommandManager,
        dm: &mut DescriptorSetCreator,
    ) {
        let (mat, desc) = self.materials.get_mut(&old).unwrap();
        *mat = new;
        let params = MaterialParams::from(&*mat);
        let mapped = self
            .update_buffer
            .allocation
            .mapped_ptr()
            .expect("Faield to map memory")
            .cast()
            .as_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(&params, mapped, 1);
        }
        let copy_region = vk::BufferCopy {
            src_offset: PARAMS_SIZE * old as u64,
            dst_offset: PARAMS_SIZE * old as u64,
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
        let cmd = cmdm.get_cmd_buffer();
        let fence = device.immediate_execute(cmd, command);
        *desc = build_mat_desc_set(
            &self.textures,
            &self.update_buffer,
            self.sampler,
            old,
            mat,
            dm,
        );
        device.wait_completion(&[fence]);
    }

    pub fn init_pipelines(
        &mut self,
        render_size: vk::Extent2D,
        device: &ash::Device,
        renderpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
    ) {
        self.pipelines = FnvHashMap::default();
        for (mat, desc) in self.materials.values() {
            let shader = mat.shader;
            self.pipelines.entry(shader).or_insert_with(|| {
                shader.build_pipeline().build(
                    device,
                    renderpass,
                    render_size,
                    &[frame_desc_layout, desc.layout],
                )
            });
        }
    }

    pub fn deinit_pipelines(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.pipelines.drain() {
            pipeline.destroy(device);
        }
    }

    pub fn unload<T: Device>(self, device: &T, mm: &mut MemoryManager) {
        self.textures
            .into_iter()
            .for_each(|(_, tex)| mm.free_image(tex.image));
        unsafe { device.logical().destroy_sampler(self.sampler, None) };
        mm.free_buffer(self.vertex_buffer);
        mm.free_buffer(self.index_buffer);
        mm.free_buffer(self.params_buffer);
        mm.free_buffer(self.update_buffer);
    }
}

fn create_sampler<T: Device>(device: &T) -> vk::Sampler {
    let max_anisotropy = device.physical().properties.limits.max_sampler_anisotropy;
    let ci = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SamplerCreateFlags::empty(),
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        mip_lod_bias: 0.0,
        anisotropy_enable: vk::FALSE,
        max_anisotropy,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 0.0,
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

fn load_vertices_to_gpu<T: Device>(
    device: &mut T,
    mm: &mut MemoryManager,
    cmdm: &mut CommandManager,
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
    let cmd = cmdm.get_cmd_buffer();
    let fence = device.immediate_execute(cmd, command);
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(cpu_buffer);
    gpu_buffer
}

fn load_indices_to_gpu<T: Device>(
    device: &mut T,
    mm: &mut MemoryManager,
    cmdm: &mut CommandManager,
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
    let mapped = cpu_buffer
        .allocation
        .mapped_ptr()
        .expect("Faield to map memory")
        .cast()
        .as_ptr();
    let mut offset = 0;
    for mesh in meshes {
        unsafe { std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), mapped, mesh.indices.len()) };
        converted_meshes.push(VulkanMesh {
            index_offset: offset,
            index_count: mesh.indices.len() as u32,
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
    let cmd = cmdm.get_cmd_buffer();
    let fence = device.immediate_execute(cmd, command);
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(cpu_buffer);
    (converted_meshes, gpu_buffer)
}

fn build_mat_desc_set(
    textures: &FnvHashMap<u16, TextureLoaded>,
    params: &AllocatedBuffer,
    sampler: vk::Sampler,
    id: u16,
    material: &Material,
    dm: &mut DescriptorSetCreator,
) -> Descriptor {
    let diffuse_id = material.diffuse.unwrap_or(0); // TODO: set default texture
    let diffuse = textures.get(&diffuse_id).expect("Failed to find texture"); //TODO: add default texture
    let diffuse_image_info = vk::DescriptorImageInfo {
        sampler,
        image_view: diffuse.image.image_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let buf_info = vk::DescriptorBufferInfo {
        buffer: params.buffer,
        offset: PARAMS_SIZE * id as u64,
        range: PARAMS_SIZE,
    };
    let descriptor = dm
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
        )
        .build();
    descriptor
}

fn load_materials_parameters<T: Device>(
    device: &mut T,
    materials: &[(u16, Material)],
    mm: &mut MemoryManager,
    cmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
) -> AllocatedBuffer {
    let size = (std::mem::size_of::<MaterialParams>() * materials.len()) as u64;
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
        .expect("Faield to map memory")
        .cast()
        .as_ptr();
    for (_, mat) in materials {
        let params = MaterialParams::from(mat);
        unsafe {
            std::ptr::copy_nonoverlapping(&params, mapped, 1);
            mapped = mapped.offset(std::mem::size_of::<MaterialParams>() as isize);
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
    let cmd = cmdm.get_cmd_buffer();
    let fence = device.immediate_execute(cmd, command);
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(cpu_buffer);
    gpu_buffer
}

fn load_texture_to_gpu<T: Device>(
    device: &mut T,
    mm: &mut MemoryManager,
    cmdm: &mut CommandManager,
    unfinished: &mut UnfinishedExecutions,
    texture: Texture,
) -> TextureLoaded {
    let (width, height) = texture.dimensions();
    let size = (width as u32 * height as u32 * 4) as u64;
    let extent = vk::Extent2D {
        width: width as u32,
        height: height as u32,
    };
    let vkformat = match texture.format() {
        TextureFormat::Gray => vk::Format::R8_UNORM,
        TextureFormat::Rgb => vk::Format::R8G8B8_SRGB,
        TextureFormat::Rgba => vk::Format::R8G8B8A8_SRGB,
    };
    let buffer = mm.create_buffer(
        "Texture Buffer",
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let image = mm.create_image_gpu(
        "Texture Image",
        vkformat,
        extent,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
    );
    let mapped = buffer
        .allocation
        .mapped_ptr()
        .expect("Failed to map memory")
        .cast()
        .as_ptr();
    unsafe {
        std::ptr::copy_nonoverlapping(texture.ptr(), mapped, size as usize);
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
            width: width as u32,
            height: height as u32,
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
                buffer.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
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
    let cmd = cmdm.get_cmd_buffer();
    let fence = device.immediate_execute(cmd, command);
    unfinished.fences.push(fence);
    unfinished.buffers_to_free.push(buffer);
    TextureLoaded {
        info: texture.to_info(),
        image,
    }
}

#[repr(C)]
struct MaterialParams {
    diffuse_mul: Vec3<f32>,
}

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
