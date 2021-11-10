use crate::materials::Pipeline;
use crate::{Camera, PerspectiveCam, Scene, ShaderMat, Vertex};
use ash::vk;
use cgmath::{Matrix4, Point3, SquareMatrix, Vector3 as Vec3};
use fnv::{FnvHashMap, FnvHashSet};
use gpu_allocator::MemoryLocation;

use super::device::Device;
use super::memory::{AllocatedBuffer, MemoryManager};

pub struct VulkanScene<const FRAMES_IN_FLIGHT: usize> {
    scene: Scene,
    pub current_cam: Camera,
    pub projview: [Matrix4<f32>; FRAMES_IN_FLIGHT],
    pub vertex_buffer: AllocatedBuffer,
    pub pipelines: FnvHashMap<u8, Pipeline>,
}

impl<const FRAMES_IN_FLIGHT: usize> VulkanScene<FRAMES_IN_FLIGHT> {
    pub fn load<T: Device>(device: &T, mm: &mut MemoryManager, scene: Scene) -> Self {
        let vertex_buffer = load_vertices_to_gpu(device, mm, &scene.vertices[..]);
        let current_cam = if scene.cameras.is_empty() {
            Camera::Perspective(PerspectiveCam {
                position: Point3::new(0.0, 0.0, 0.0),
                target: Point3::new(0.0, 0.0, 100.0),
                up: Vec3::new(0.0, 1.0, 0.0),
                fovx: 90.0,
            })
        } else {
            scene.cameras[0].clone()
        };
        let projview = [Matrix4::<f32>::identity(); FRAMES_IN_FLIGHT];
        let pipelines = FnvHashMap::default();
        VulkanScene {
            scene,
            current_cam,
            projview,
            vertex_buffer,
            pipelines,
        }
    }

    pub fn init_pipelines(
        &mut self,
        width: u32,
        height: u32,
        device: &ash::Device,
        renderpass: vk::RenderPass,
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
            .scene
            .materials
            .iter()
            .map(|(_, _, mat)| mat.shader_id)
            .collect::<FnvHashSet<_>>()
            .into_iter()
            .map(|id| {
                (
                    id,
                    ShaderMat::from_id(id)
                        .expect("Unexpected shader ID")
                        .build_pipeline()
                        .build(device, renderpass, &viewports, &scissors, &[]),
                )
            })
            .collect::<FnvHashMap<_, _>>();
    }

    pub fn deinit_pipelines(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.pipelines.drain() {
            pipeline.destroy(device);
        }
    }

    pub fn unload<T: Device>(self, mm: &mut MemoryManager) {
        mm.free_buffer(self.vertex_buffer);
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
    unsafe {
        let mapped = device
            .logical()
            .map_memory(
                cpu_buffer.allocation.memory(),
                cpu_buffer.allocation.offset(),
                size,
                vk::MemoryMapFlags::default(),
            )
            .expect("Failed to map memory") as *mut Vertex;
        mapped.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        device
            .logical()
            .unmap_memory(cpu_buffer.allocation.memory());
    }
    device.copy_buffer(&cpu_buffer, &gpu_buffer);
    mm.free_buffer(cpu_buffer);
    gpu_buffer
}
