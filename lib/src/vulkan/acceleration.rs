use super::cmd::CommandManager;
use super::device::Device;
use super::memory::{AllocatedBuffer, MemoryManager};
use super::scene::VulkanMesh;
use super::Instance;
use crate::{Material, MeshInstance, Transform, Vertex};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk;
use fnv::{FnvBuildHasher, FnvHashMap};
use gpu_allocator::MemoryLocation;
use std::ptr;
use std::sync::Arc;

pub struct AllocatedAS {
    pub accel: ash::vk::AccelerationStructureKHR,
    pub buffer: AllocatedBuffer,
    loader: Arc<AccelerationLoader>,
}

impl Drop for AllocatedAS {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_acceleration_structure(self.accel, None) };
    }
}

pub struct SceneAS {
    pub blas: FnvHashMap<u16, AllocatedAS>,
    pub tlas: AllocatedAS,
}

pub struct SceneASBuilder<'scene, 'renderer> {
    vkmeshes: &'scene [VulkanMesh],
    instances: &'scene [MeshInstance],
    transform: &'scene [Transform],
    materials: &'scene [Material],
    vb: &'scene AllocatedBuffer,
    ib: &'scene AllocatedBuffer,
    mm: &'renderer MemoryManager,
    ccmdm: &'renderer mut CommandManager,
    device: &'renderer Device,
    properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    loader: Arc<AccelerationLoader>,
}

impl<'scene, 'renderer> SceneASBuilder<'scene, 'renderer> {
    pub fn new<T: Instance>(
        instance: &'renderer T,
        loader: Arc<AccelerationLoader>,
        mm: &'renderer MemoryManager,
        ccmdm: &'renderer mut CommandManager,
        vertex_buffer: &'scene AllocatedBuffer,
        index_buffer: &'scene AllocatedBuffer,
    ) -> Self {
        let properties = unsafe {
            AccelerationLoader::get_properties(
                instance.instance(),
                instance.device().physical().device,
            )
        };
        SceneASBuilder {
            vkmeshes: &[],
            instances: &[],
            transform: &[],
            materials: &[],
            vb: vertex_buffer,
            ib: index_buffer,
            mm,
            ccmdm,
            device: instance.device(),
            properties,
            loader,
        }
    }

    pub fn with_meshes(
        mut self,
        meshes: &'scene [VulkanMesh],
        instances: &'scene [MeshInstance],
        transforms: &'scene [Transform],
        materials: &'scene [Material],
    ) -> Self {
        self.vkmeshes = meshes;
        self.instances = instances;
        self.transform = transforms;
        self.materials = materials;
        self
    }

    pub fn build(mut self) -> SceneAS {
        let blas = if !self.vkmeshes.is_empty() {
            self.build_blases()
        } else {
            Default::default()
        };
        let tlas = self.build_tlas(&blas);
        SceneAS { blas, tlas }
    }

    fn build_blases(&mut self) -> FnvHashMap<u16, AllocatedAS> {
        let vkdevice = self.device.logical();
        let mut blas_ci = Vec::with_capacity(self.vkmeshes.len());
        let mut retval =
            FnvHashMap::with_capacity_and_hasher(self.vkmeshes.len(), FnvBuildHasher::default());
        let mut scratch_size = 0;
        // first iteration: partial build and get the max memory usage
        for mesh in self.vkmeshes {
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
                    max_vertex: mesh.max_index,
                    index_type: vk::IndexType::UINT32,
                    index_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: ib_addr,
                    },
                    transform_data: vk::DeviceOrHostAddressConstKHR { device_address: 0 },
                },
            };
            let material = &self.materials[mesh.material as usize];
            let flags = if material.opacity == 0 {
                vk::GeometryFlagsKHR::OPAQUE
            } else {
                vk::GeometryFlagsKHR::empty()
            };
            let geometry = [vk::AccelerationStructureGeometryKHR {
                s_type: vk::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                p_next: ptr::null(),
                geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                geometry,
                flags,
            }];
            let primitive_count = mesh.index_count / 3;
            let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: mesh.index_count / 3,
                primitive_offset: (mesh.index_offset * std::mem::size_of::<u32>() as u32),
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
            blas_ci.push((mesh.mesh_id, build_info, build_range, geometry, req_mem));
        }
        // allocates buffer
        let min_align = self
            .properties
            .min_acceleration_structure_scratch_offset_alignment as u64;
        let scratch_buf = self.mm.create_buffer(
            "BLAS scratch",
            scratch_size + min_align,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        );
        let scratch_addr_info = vk::BufferDeviceAddressInfo {
            s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
            p_next: ptr::null(),
            buffer: scratch_buf.buffer,
        };
        let mut scratch_addr = unsafe { vkdevice.get_buffer_device_address(&scratch_addr_info) };
        scratch_addr += min_align - scratch_addr % min_align;
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
            cur_chunk_size += ci.4.acceleration_structure_size;
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
            // unsafe { vkdevice.cmd_reset_query_pool(cmd, query_pool, 0, blas_no) };
            for (index, (id, mut build_info, build_range, geometry, req_mem)) in
                blas_chunk.into_iter().enumerate()
            {
                let blas = allocate_as(
                    self.loader.clone(),
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
                blas_tmp.push((id, blas));
                // need to add a barrier, because the operations are all using the same scratch buffer
                let barrier = vk::MemoryBarrier {
                    s_type: vk::StructureType::MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                    dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                };
                let commands = unsafe {
                    |_: &ash::Device, cmd: vk::CommandBuffer| {
                        vkdevice.cmd_reset_query_pool(cmd, query_pool, index as u32, 1);
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
                            index as u32,
                        );
                    }
                };
                fences.push(self.device.submit_immediate(self.ccmdm, commands));
            }
            // wait for creation, then start compaction
            self.device.wait_completion(&fences);
            fences.clear();
            let mut compact_size = vec![0_u64; blas_no as usize];
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
            for (index, (id, blas)) in blas_tmp.iter().enumerate() {
                let compacted_size = compact_size[index];
                let compacted_blas =
                    allocate_as(self.loader.clone(), self.mm, compacted_size, true);
                let copy_ci = vk::CopyAccelerationStructureInfoKHR {
                    s_type: vk::StructureType::COPY_ACCELERATION_STRUCTURE_INFO_KHR,
                    p_next: ptr::null(),
                    src: blas.accel,
                    dst: compacted_blas.accel,
                    mode: vk::CopyAccelerationStructureModeKHR::COMPACT,
                };
                retval.insert(*id, compacted_blas);
                let commands = unsafe {
                    |_: &ash::Device, cmd: vk::CommandBuffer| {
                        self.loader.cmd_copy_acceleration_structure(cmd, &copy_ci);
                    }
                };
                fences.push(self.device.submit_immediate(self.ccmdm, commands));
            }
            // wait for compaction, then cleanup the tmp blases
            self.device.wait_completion(&fences);
        }
        // cleanup the query pool and return
        unsafe { vkdevice.destroy_query_pool(query_pool, None) };
        retval
    }

    fn build_tlas(&mut self, blases: &FnvHashMap<u16, AllocatedAS>) -> AllocatedAS {
        let vkdevice = self.device.logical();
        let mut tlas_ci = Vec::with_capacity(self.instances.len());
        // populates the instances
        for (instance_id, instance) in self.instances.iter().enumerate() {
            let blas = blases.get(&instance.mesh_id).unwrap();
            let transform = &self.transform[instance.transform_id as usize];
            let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR {
                s_type: vk::StructureType::ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                p_next: ptr::null(),
                acceleration_structure: blas.accel,
            };
            let as_addr = unsafe {
                self.loader
                    .get_acceleration_structure_device_address(&as_addr_info)
            };
            let ci = vk::AccelerationStructureInstanceKHR {
                transform: transform.to_vulkan_transform(),
                instance_custom_index_and_mask: vk::Packed24_8::new(instance_id as u32, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: as_addr,
                },
            };
            tlas_ci.push(ci);
        }
        // create a buffer to store all the instances (needed by subsequent methods)
        let inst_buf_size = u64::max(
            1,
            (std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * tlas_ci.len()) as u64,
        );
        let inst_cpu_buf = self.mm.create_buffer(
            "Instances CPU buffer",
            inst_buf_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        let inst_gpu_buf = self.mm.create_buffer(
            "Instances CPU buffer",
            inst_buf_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
        );
        let mapped = inst_cpu_buf
            .allocation()
            .mapped_ptr()
            .expect("Failed to map memory")
            .cast()
            .as_ptr();
        unsafe { std::ptr::copy_nonoverlapping(tlas_ci.as_ptr(), mapped, tlas_ci.len()) };
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
                &[tlas_ci.len() as u32],
            )
        };
        let tlas = allocate_as(
            self.loader.clone(),
            self.mm,
            req_mem.acceleration_structure_size,
            false,
        );
        let min_align = self
            .properties
            .min_acceleration_structure_scratch_offset_alignment as u64;
        let scratch_buf = self.mm.create_buffer(
            "TLAS scratch",
            req_mem.build_scratch_size + min_align,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        );
        let scratch_addr_info = vk::BufferDeviceAddressInfo {
            s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
            p_next: ptr::null(),
            buffer: scratch_buf.buffer,
        };
        let mut scratch_addr = unsafe { vkdevice.get_buffer_device_address(&scratch_addr_info) };
        scratch_addr += min_align - scratch_addr % min_align;
        // build the tlas
        build_info.dst_acceleration_structure = tlas.accel;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let build_range = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: tlas_ci.len() as u32,
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
        let fence = self.device.submit_immediate(self.ccmdm, command);
        self.device.wait_completion(&[fence]);
        tlas
    }
}

fn allocate_as(
    loader: Arc<AccelerationLoader>,
    mm: &MemoryManager,
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
        loader,
    }
}
