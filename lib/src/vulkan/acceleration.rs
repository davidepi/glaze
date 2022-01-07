use super::{
    cmd::CommandManager,
    device::Device,
    memory::{AllocatedBuffer, MemoryManager},
    scene::VulkanMesh,
};
use crate::{Mesh, Vertex};
use ash::{extensions::khr::AccelerationStructure as AccelerationLoader, vk};
use gpu_allocator::MemoryLocation;
use std::ptr;

pub struct AllocatedAS {
    pub accel: vk::AccelerationStructureKHR,
    pub buffer: AllocatedBuffer,
}

impl AllocatedAS {
    pub fn destroy(self, mm: &mut MemoryManager, loader: &AccelerationLoader) {
        unsafe { loader.destroy_acceleration_structure(self.accel, None) };
        mm.free_buffer(self.buffer);
    }
}

struct SceneAS {
    blas: Vec<AllocatedAS>,
    tlas: AllocatedAS,
}

struct SceneASBuilder<'scene, 'renderer> {
    vkmeshes: Vec<&'scene VulkanMesh>,
    max_vertices: Vec<u32>,
    vb: &'scene AllocatedBuffer,
    ib: &'scene AllocatedBuffer,
    mm: &'renderer mut MemoryManager,
    cmdm: &'renderer mut CommandManager,
    device: &'renderer Device,
    loader: &'renderer AccelerationLoader,
}

impl<'scene, 'renderer> SceneASBuilder<'scene, 'renderer> {
    pub fn new(
        device: &'renderer Device,
        mm: &'renderer mut MemoryManager,
        cmdm: &'renderer mut CommandManager,
        vertex_buffer: &'scene AllocatedBuffer,
        index_buffer: &'scene AllocatedBuffer,
        loader: &'renderer AccelerationLoader,
    ) -> Self {
        SceneASBuilder {
            vkmeshes: Vec::new(),
            max_vertices: Vec::new(),
            vb: vertex_buffer,
            ib: index_buffer,
            mm,
            cmdm,
            device,
            loader,
        }
    }

    pub fn add_mesh(&mut self, mesh: &Mesh, vkmesh: &'scene VulkanMesh) {
        let max_vertex = mesh.indices.iter().max().copied().unwrap_or(0);
        self.max_vertices.push(max_vertex);
        self.vkmeshes.push(vkmesh);
    }

    pub fn build(mut self) -> SceneAS {
        let blas = self.build_blases();
        let tlas = self.build_tlas(&blas);
        SceneAS {
            blas,
            tlas,
        }
    }

    fn build_blases(&mut self) -> Vec<AllocatedAS> {
        let vkdevice = self.device.logical();
        let mut blas_ci = Vec::with_capacity(self.vkmeshes.len());
        let mut retval = Vec::with_capacity(self.vkmeshes.len());
        let mut scratch_size = 0;
        // first iteration: partial build and get the max memory usage
        for (&mesh, &max_vertex) in self.vkmeshes.iter().zip(self.max_vertices.iter()) {
            let vb_addr_info = vk::BufferDeviceAddressInfo {
                s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
                p_next: ptr::null(),
                buffer: self.vb.buffer,
            };
            let ib_addr_info = vk::BufferDeviceAddressInfo {
                s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
                p_next: ptr::null(),
                buffer: self.ib.buffer,
            };
            let vb_addr = unsafe { vkdevice.get_buffer_device_address(&vb_addr_info) };
            let ib_addr = unsafe { vkdevice.get_buffer_device_address(&ib_addr_info) };
            let geometry = vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                    s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                    p_next: ptr::null(),
                    vertex_format: vk::Format::R32G32B32_SFLOAT,
                    vertex_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: vb_addr,
                    },
                    vertex_stride: std::mem::size_of::<Vertex>() as u64,
                    max_vertex,
                    index_type: vk::IndexType::UINT32,
                    index_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: ib_addr,
                    },
                    transform_data: vk::DeviceOrHostAddressConstKHR { device_address: 0 },
                },
            };
            let geometry = [vk::AccelerationStructureGeometryKHR {
                s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                p_next: ptr::null(),
                geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                geometry,
                flags: vk::GeometryFlagsKHR::OPAQUE,
            }];
            let primitive_count = (mesh.index_count / 3) as u32;
            let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: (mesh.index_count / 3) as u32,
                primitive_offset: mesh.index_offset as u32,
                first_vertex: 0,
                transform_offset: 0,
            };
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
                s_type: vk::StructureType::ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                p_next: ptr::null(),
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION,
                mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                src_acceleration_structure: vk::AccelerationStructureKHR::null(),
                dst_acceleration_structure: vk::AccelerationStructureKHR::null(),
                geometry_count: geometry.len() as u32,
                p_geometries: geometry.as_ptr(),
                pp_geometries: ptr::null(),
                scratch_data: vk::DeviceOrHostAddressKHR { device_address: 0 },
            };
            let req_mem = unsafe {
                self.loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[primitive_count],
                )
            };
            scratch_size = std::cmp::max(scratch_size, req_mem.build_scratch_size);
            blas_ci.push((build_info, build_range, geometry, req_mem));
        }
        // allocates buffer
        let scratch_buf = self.mm.create_buffer(
            "BLAS scratch",
            scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        );
        let scratch_addr_info = vk::BufferDeviceAddressInfo {
            s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
            p_next: ptr::null(),
            buffer: scratch_buf.buffer,
        };
        let scratch_addr = unsafe { vkdevice.get_buffer_device_address(&scratch_addr_info) };
        // create blases in chunks of 512MB (compaction is done after ALL the blases are created)
        // thats why I need to split in chunks.
        const MAX_CHUNK_SIZE: u64 = 536870912;
        let mut chunks = vec![];
        let mut cur_chunk_size = u64::MAX;
        for ci in blas_ci {
            if cur_chunk_size > MAX_CHUNK_SIZE {
                chunks.push(Vec::new());
                cur_chunk_size = 0;
            }
            cur_chunk_size += ci.3.acceleration_structure_size;
            chunks.last_mut().unwrap().push(ci);
        }
        // create a query pool (needed to query the size of the compacted ci)
        let query_pool_ci = vk::QueryPoolCreateInfo {
            s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::QueryPoolCreateFlags::empty(),
            query_type: vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_count: chunks.iter().map(|c| c.len()).max().unwrap_or(0) as u32,
            pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
        };
        let query_pool = unsafe { vkdevice.create_query_pool(&query_pool_ci, None) }
            .expect("Failed to create query pool");
        // finally create the blases
        for blas_chunk in chunks {
            let mut fences = Vec::with_capacity(blas_chunk.len());
            let mut blas_tmp = Vec::with_capacity(blas_chunk.len());
            let blas_no = blas_chunk.len() as u32;
            unsafe { vkdevice.reset_query_pool(query_pool, 0, blas_chunk.len() as u32) };
            for (id, (mut build_info, build_range, geometry, req_mem)) in
                blas_chunk.into_iter().enumerate()
            {
                let cmd = self.cmdm.get_cmd_buffer();
                let blas = allocate_as(
                    self.loader,
                    self.mm,
                    req_mem.acceleration_structure_size,
                    true,
                );
                // geometry got moved into the vec so I need to set this pointer again
                build_info.p_geometries = geometry.as_ptr();
                build_info.dst_acceleration_structure = blas.accel;
                build_info.scratch_data = vk::DeviceOrHostAddressKHR {
                    device_address: scratch_addr,
                };
                blas_tmp.push(blas);
                // need to add a barrier, because the operations are all using the same scratch buffer
                let barrier = vk::MemoryBarrier {
                    s_type: vk::StructureType::MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                    dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                };
                let commands = unsafe {
                    |_: &ash::Device, cmd: vk::CommandBuffer| {
                        self.loader.cmd_build_acceleration_structures(
                            cmd,
                            &[build_info],
                            &[&[build_range]],
                        );
                        vkdevice.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                            vk::DependencyFlags::empty(),
                            &[barrier],
                            &[],
                            &[],
                        );
                        self.loader.cmd_write_acceleration_structures_properties(
                            cmd,
                            &[build_info.dst_acceleration_structure],
                            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                            query_pool,
                            id as u32,
                        );
                    }
                };
                let compute_queue = self.device.compute_queue();
                fences.push(self.device.immediate_execute(cmd, compute_queue, commands));
            }
            // wait for creation, then start compaction
            self.device.wait_completion(&fences);
            fences.clear();
            let mut compact_size = Vec::<vk::DeviceSize>::with_capacity(blas_no as usize);
            unsafe {
                vkdevice.get_query_pool_results(
                    query_pool,
                    0,
                    blas_no,
                    &mut compact_size,
                    vk::QueryResultFlags::WAIT,
                )
            }
            .expect("Failed to get compacted size");
            for (id, blas) in blas_tmp.iter().enumerate() {
                let cmd = self.cmdm.get_cmd_buffer();
                let compacted_size = compact_size[id];
                let compacted_blas = allocate_as(self.loader, self.mm, compacted_size, true);
                let copy_ci = vk::CopyAccelerationStructureInfoKHR {
                    s_type: vk::StructureType::COPY_ACCELERATION_STRUCTURE_INFO_KHR,
                    p_next: ptr::null(),
                    src: blas.accel,
                    dst: compacted_blas.accel,
                    mode: vk::CopyAccelerationStructureModeKHR::COMPACT,
                };
                retval.push(compacted_blas);
                let commands = unsafe {
                    |_: &ash::Device, cmd: vk::CommandBuffer| {
                        self.loader.cmd_copy_acceleration_structure(cmd, &copy_ci);
                    }
                };
                let compute_queue = self.device.compute_queue();
                fences.push(self.device.immediate_execute(cmd, compute_queue, commands));
            }
            // wait for compaction, then cleanup the tmp blases
            self.device.wait_completion(&fences);
            blas_tmp
                .into_iter()
                .for_each(|b| b.destroy(self.mm, self.loader));
        }
        // cleanup the scratch buffer and return
        self.mm.free_buffer(scratch_buf);
        retval
    }

    fn build_tlas(&mut self, blases: &[AllocatedAS]) -> AllocatedAS {
        unimplemented!()
    }
}

fn allocate_as(
    loader: &AccelerationLoader,
    mm: &mut MemoryManager,
    size: u64,
    is_blas: bool,
) -> AllocatedAS {
    let buffer = mm.create_buffer(
        "BLAS",
        size,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        MemoryLocation::GpuOnly,
    );
    let ty = if is_blas {
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
    } else {
        vk::AccelerationStructureTypeKHR::TOP_LEVEL
    };
    let ci = vk::AccelerationStructureCreateInfoKHR {
        s_type: vk::StructureType::ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        create_flags: vk::AccelerationStructureCreateFlagsKHR::empty(),
        buffer: buffer.buffer,
        offset: 0,
        size,
        ty,
        device_address: 0,
    };
    let accs =
        unsafe { loader.create_acceleration_structure(&ci, None) }.expect("Failed to create BLAS");
    AllocatedAS {
        accel: accs,
        buffer,
    }
}
