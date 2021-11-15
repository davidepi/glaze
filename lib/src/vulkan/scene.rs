use std::ptr;

use super::descriptor::{Descriptor, DescriptorSetCreator};
use super::device::Device;
use super::memory::{AllocatedBuffer, AllocatedImage, MemoryManager};
use crate::materials::Pipeline;
use crate::{Camera, Material, Mesh, Scene, ShaderMat, Texture, Vertex};
use ash::vk::{self, DescriptorType};
use fnv::{FnvHashMap, FnvHashSet};
use gpu_allocator::MemoryLocation;

pub struct VulkanScene {
    pub current_cam: Camera,
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
    pub meshes: Vec<VulkanMesh>,
    pub sampler: vk::Sampler,
    pub materials: FnvHashMap<u16, (Material, Descriptor)>,
    pub pipelines: FnvHashMap<u8, Pipeline>,
    pub textures: FnvHashMap<u16, AllocatedImage>,
}

pub struct VulkanMesh {
    pub index_offset: u32,
    pub index_count: u32,
    pub material: u16,
}

impl VulkanScene {
    pub fn load<T: Device>(
        device: &T,
        mm: &mut MemoryManager,
        scene: Scene,
        descriptor_creator: &mut DescriptorSetCreator,
    ) -> Self {
        let vertex_buffer = load_vertices_to_gpu(device, mm, &scene.vertices[..]);
        let (meshes, index_buffer) = load_indices_to_gpu(device, mm, &scene.meshes[..]);
        let sampler = create_sampler(device);
        let textures = scene
            .textures
            .iter()
            .map(|(id, _, tex)| (*id, load_single_texture(device, mm, tex)))
            .collect();
        let materials = load_materials_to_gpu(
            device,
            &textures,
            sampler,
            &scene.materials.iter().as_slice(),
            descriptor_creator,
        );
        let current_cam = scene.cameras[0].clone(); // parser automatically adds a default cam
        let pipelines = FnvHashMap::default();
        VulkanScene {
            current_cam,
            vertex_buffer,
            index_buffer,
            meshes,
            sampler,
            materials,
            pipelines,
            textures,
        }
    }

    pub fn init_pipelines(
        &mut self,
        width: u32,
        height: u32,
        device: &ash::Device,
        renderpass: vk::RenderPass,
        frame_desc_layout: vk::DescriptorSetLayout,
    ) {
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        }];
        self.pipelines = FnvHashMap::default();
        for (_, (mat, desc)) in &self.materials {
            let shader_id = mat.shader_id;
            if !self.pipelines.contains_key(&shader_id) {
                // material layout == shader layout
                let pipeline = ShaderMat::from_id(shader_id)
                    .expect("Unexpected shader ID")
                    .build_pipeline()
                    .build(
                        device,
                        renderpass,
                        &viewports,
                        &scissors,
                        &[frame_desc_layout, desc.layout],
                    );
                self.pipelines.insert(shader_id, pipeline);
            }
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
            .for_each(|(_, tex)| mm.free_image(tex));
        unsafe { device.logical().destroy_sampler(self.sampler, None) };
        mm.free_buffer(self.vertex_buffer);
        mm.free_buffer(self.index_buffer);
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
    device: &T,
    mm: &mut MemoryManager,
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
    device.immediate_execute(command);
    mm.free_buffer(cpu_buffer);
    gpu_buffer
}

fn load_indices_to_gpu<T: Device>(
    device: &T,
    mm: &mut MemoryManager,
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
    device.immediate_execute(command);
    mm.free_buffer(cpu_buffer);
    (converted_meshes, gpu_buffer)
}

fn load_materials_to_gpu<T: Device>(
    device: &T,
    textures: &FnvHashMap<u16, AllocatedImage>,
    sampler: vk::Sampler,
    materials: &[(u16, String, Material)],
    descriptor_creator: &mut DescriptorSetCreator,
) -> FnvHashMap<u16, (Material, Descriptor)> {
    let mut retval = FnvHashMap::with_capacity_and_hasher(materials.len(), Default::default());
    for (id, _, mat) in materials {
        let diffuse_id = mat.diffuse.unwrap_or(0); // TODO: set default texture
        let diffuse = textures.get(&diffuse_id).expect("Failed to find texture"); //TODO: add default texture
        let diffuse_image_info = vk::DescriptorImageInfo {
            sampler,
            image_view: diffuse.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let descriptor = descriptor_creator
            .new_set()
            .bind_image(
                diffuse_image_info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
            )
            .build();
        retval.insert(*id, (mat.clone(), descriptor));
    }
    retval
}

fn load_single_texture<T: Device>(
    device: &T,
    mm: &mut MemoryManager,
    texture: &Texture,
) -> AllocatedImage {
    let size = (texture.width() * texture.height() * 4) as u64;
    let extent = vk::Extent2D {
        width: texture.width(),
        height: texture.height(),
    };
    let buffer = mm.create_buffer(
        "TextureBuffer",
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );
    let image = mm.create_image_gpu(
        vk::Format::R8G8B8A8_SRGB,
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
        std::ptr::copy_nonoverlapping(texture.as_ptr(), mapped, size as usize);
    }
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
            width: texture.width(),
            height: texture.height(),
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
    device.immediate_execute(command);
    mm.free_buffer(buffer);
    image
}
