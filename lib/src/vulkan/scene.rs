use crate::materials::Pipeline;
use crate::{Camera, Material, Mesh, Scene, ShaderMat, Vertex};
use ash::vk;
use cgmath::{Matrix4, SquareMatrix};
use fnv::{FnvHashMap, FnvHashSet};
use gpu_allocator::MemoryLocation;

use super::device::Device;
use super::memory::{AllocatedBuffer, MemoryManager};

pub struct VulkanScene {
    pub current_cam: Camera,
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
    pub meshes: Vec<VulkanMesh>,
    pub materials: FnvHashMap<u16, Material>,
    pub pipelines: FnvHashMap<u8, Pipeline>,
}

pub struct VulkanMesh {
    pub index_offset: u32,
    pub index_count: u32,
    pub material: u16,
}

impl VulkanScene {
    pub fn load<T: Device>(device: &T, mm: &mut MemoryManager, scene: Scene) -> Self {
        let vertex_buffer = load_vertices_to_gpu(device, mm, &scene.vertices[..]);
        let (meshes, index_buffer) = load_indices_to_gpu(device, mm, &scene.meshes[..]);
        let materials = scene
            .materials
            .iter()
            .map(|(id, _, mat)| (*id, mat.clone()))
            .collect();
        let current_cam = scene.cameras[0].clone(); // parser automatically adds a default cam
        let pipelines = FnvHashMap::default();
        VulkanScene {
            current_cam,
            vertex_buffer,
            index_buffer,
            meshes,
            materials,
            pipelines,
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
        // collect all materials requiring a pipeline
        self.pipelines = self
            .materials
            .iter()
            .map(|(_, mat)| mat.shader_id)
            .collect::<FnvHashSet<_>>()
            .into_iter()
            .map(|id| {
                (
                    id,
                    ShaderMat::from_id(id)
                        .expect("Unexpected shader ID")
                        .build_pipeline()
                        .build(
                            device,
                            renderpass,
                            &viewports,
                            &scissors,
                            &[frame_desc_layout],
                        ),
                )
            })
            .collect::<FnvHashMap<_, _>>();
    }

    pub fn deinit_pipelines(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.pipelines.drain() {
            pipeline.destroy(device);
        }
    }

    pub fn unload(self, mm: &mut MemoryManager) {
        mm.free_buffer(self.vertex_buffer);
        mm.free_buffer(self.index_buffer);
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
    device.copy_buffer(&cpu_buffer, &gpu_buffer);
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
    device.copy_buffer(&cpu_buffer, &gpu_buffer);
    mm.free_buffer(cpu_buffer);
    (converted_meshes, gpu_buffer)
}
