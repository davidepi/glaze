use super::cmd::CommandManager;
use super::device::Device;
use super::memory::{AllocatedBuffer, MemoryManager};
use super::scene::VulkanMesh;
use crate::vulkan::instance::Instance;
use crate::{Mesh, Vertex};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk::{self, AccelerationStructureReferenceKHR, Packed24_8};
use cgmath::{Matrix, Matrix4, SquareMatrix};
use gpu_allocator::MemoryLocation;
use std::ops::Deref;
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
    ccmdm: &'renderer mut CommandManager,
    device: &'renderer Device,
    loader: &'renderer AccelerationLoader,
}

impl<'scene, 'renderer> SceneASBuilder<'scene, 'renderer> {
    pub fn new(
        device: &'renderer Device,
        mm: &'renderer mut MemoryManager,
        ccmdm: &'renderer mut CommandManager,
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
            ccmdm,
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
        SceneAS { blas, tlas }
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
                let cmd = self.ccmdm.get_cmd_buffer();
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
                let cmd = self.ccmdm.get_cmd_buffer();
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
        let cmd = self.ccmdm.get_cmd_buffer();
        let vkdevice = self.device.logical();
        // populates the instances
        // TODO: add correct instancing and shader binding
        let instances = blases
            .iter()
            .enumerate()
            .map(|(id, blas)| {
                let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
                    s_type: vk::StructureType::ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    p_next: ptr::null(),
                    acceleration_structure: blas.accel,
                };
                let as_addr = unsafe {
                    self.loader
                        .get_acceleration_structure_device_address(&as_addr_info)
                };
                vk::AccelerationStructureInstanceKHR {
                    transform: to_transform_mat(Matrix4::identity()),
                    instance_custom_index_and_mask: Packed24_8::new(id as u32, 0xFF), //TODO add index
                    instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                        0,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                    ),
                    acceleration_structure_reference: AccelerationStructureReferenceKHR {
                        device_handle: as_addr,
                    },
                }
            })
            .collect::<Vec<_>>();
        // create a buffer to store all the instances (needed by subsequent methods)
        let inst_buf_size =
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len();
        let inst_cpu_buf = self.mm.create_buffer(
            "Instances CPU buffer",
            inst_buf_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
        );
        let inst_gpu_buf = self.mm.create_buffer(
            "Instances CPU buffer",
            inst_buf_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
        );
        let mapped = inst_cpu_buf
            .allocation
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        unsafe { std::ptr::copy_nonoverlapping(instances.as_ptr(), mapped, instances.len()) };
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: inst_cpu_buf.size,
        };
        let barrier = vk::MemoryBarrier {
            s_type: vk::StructureType::MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        };
        // fill the required structures
        let inst_addr_info = vk::BufferDeviceAddressInfo {
            s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
            p_next: ptr::null(),
            buffer: inst_gpu_buf.buffer,
        };
        let inst_addr = unsafe {
            self.device
                .logical()
                .get_buffer_device_address(&inst_addr_info)
        };
        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            p_next: ptr::null(),
            array_of_pointers: vk::FALSE,
            data: vk::DeviceOrHostAddressConstKHR {
                device_address: inst_addr,
            },
        };
        let geometry = [vk::AccelerationStructureGeometryKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            p_next: ptr::null(),
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            },
            flags: vk::GeometryFlagsKHR::empty(),
        }];
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            s_type: vk::StructureType::ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            p_next: ptr::null(),
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            src_acceleration_structure: vk::AccelerationStructureKHR::null(),
            dst_acceleration_structure: vk::AccelerationStructureKHR::null(),
            geometry_count: geometry.len() as u32,
            p_geometries: geometry.as_ptr(),
            pp_geometries: ptr::null(),
            scratch_data: vk::DeviceOrHostAddressKHR { device_address: 0 },
        };
        // get the scratch buf size and allocate everything
        let req_mem = unsafe {
            self.loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[instances.len() as u32],
            )
        };
        let tlas = allocate_as(
            self.loader,
            self.mm,
            req_mem.acceleration_structure_size,
            false,
        );
        let scratch_buf = self.mm.create_buffer(
            "TLAS scratch",
            req_mem.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        );
        let scratch_addr_info = vk::BufferDeviceAddressInfo {
            s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
            p_next: ptr::null(),
            buffer: scratch_buf.buffer,
        };
        let scratch_addr = unsafe { vkdevice.get_buffer_device_address(&scratch_addr_info) };
        // build the tlas
        build_info.dst_acceleration_structure = tlas.accel;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: instances.len() as u32,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        };
        let command = unsafe {
            |device: &ash::Device, cmd: vk::CommandBuffer| {
                device.cmd_copy_buffer(
                    cmd,
                    inst_cpu_buf.buffer,
                    inst_gpu_buf.buffer,
                    &[copy_region],
                );
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
                self.loader.cmd_build_acceleration_structures(
                    cmd,
                    &[build_info],
                    &[&[build_range]],
                );
            }
        };
        let compute = self.device.compute_queue();
        let fence = self.device.immediate_execute(cmd, compute, command);
        self.device.wait_completion(&[fence]);
        // free everything
        self.mm.free_buffer(inst_cpu_buf);
        self.mm.free_buffer(inst_gpu_buf);
        self.mm.free_buffer(scratch_buf);
        tlas
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

fn to_transform_mat(matrix: Matrix4<f32>) -> vk::TransformMatrixKHR {
    let transpose = matrix.transpose();
    let floats: [f32; 16] = *transpose.as_ref();
    let matrix: [f32; 12] = floats[..12].try_into().unwrap();
    vk::TransformMatrixKHR { matrix }
}

#[cfg(test)]
mod tests {
    use cgmath::Matrix4;

    use super::to_transform_mat;

    #[test]
    fn cgmath_vktransform_memory_layout() {
        let matrix = Matrix4::new(
            0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0,
        );
        let transform = to_transform_mat(matrix);
        assert_eq!(
            transform.matrix,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
    }
}
