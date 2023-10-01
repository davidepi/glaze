use super::DeviceVulkan;
use crate::graphics::error::{ErrorCategory, GraphicError};
use crate::graphics::memory::{MemoryAllocatorInternal, MemoryType};
use ash::vk;
use std::ptr;

struct AllocatorVulkan<'device> {
    device: &'device DeviceVulkan,
    dedicated_index: u32,
    shared_index: u32,
    total_dedicated_memory: u64,
    total_shared_memory: u64,
}

impl<'device> AllocatorVulkan<'device> {
    pub fn new(device: &'device DeviceVulkan) -> Result<AllocatorVulkan<'device>, GraphicError> {
        let pdevice = device.physical();
        let memory_properties = unsafe {
            device
                .instance()
                .vk_instance()
                .get_physical_device_memory_properties(pdevice.device)
        };
        let dedicated_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let shared_flags =
            vk::MemoryPropertyFlags::HOST_COHERENT & vk::MemoryPropertyFlags::HOST_VISIBLE;
        let mut dedicated_heaps = vec![];
        let mut shared_heaps = vec![];
        for (index, mtype) in memory_properties.memory_types.iter().enumerate() {
            let size = memory_properties.memory_heaps[mtype.heap_index as usize].size;
            if mtype.property_flags.contains(dedicated_flags) {
                dedicated_heaps.push((index, size));
            }
            if mtype.property_flags.contains(shared_flags) {
                shared_heaps.push((index, size));
            }
        }
        // pick the biggest heap (i.e. Nvidia on the 3080 has 256M with both HOST_VISIBLE and
        // DEVICE_LOCAL)
        let (dedicated_index, dedicated_size) = dedicated_heaps
            .into_iter()
            .max_by_key(|(_, size)| *size)
            .ok_or_else(|| {
                GraphicError::new(ErrorCategory::DeviceMemory, "No dedicated memory detected")
            })?;
        let (shared_index, shared_size) = shared_heaps
            .into_iter()
            .max_by_key(|(_, size)| *size)
            .ok_or_else(|| {
                GraphicError::new(ErrorCategory::HostMemory, "No shared memory detected")
            })?;
        Ok(Self {
            device,
            dedicated_index: dedicated_index as u32,
            shared_index: shared_index as u32,
            total_dedicated_memory: dedicated_size,
            total_shared_memory: shared_size,
        })
    }

    pub fn index(&self, memory_type: MemoryType) -> u32 {
        match memory_type {
            MemoryType::Shared => self.shared_index,
            MemoryType::Dedicated => self.dedicated_index,
        }
    }
}

struct MemorySlabVulkan {
    memory: vk::DeviceMemory,
}

impl MemoryAllocatorInternal for AllocatorVulkan<'_> {
    type MemorySlab = MemorySlabVulkan;

    fn allocate_slab(
        &mut self,
        memory_type: MemoryType,
        size: u64,
    ) -> Result<Self::MemorySlab, GraphicError> {
        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: size,
            memory_type_index: self.index(memory_type),
        };
        let allocation = unsafe { self.device.logical().allocate_memory(&allocate_info, None) }?;
        Ok(MemorySlabVulkan { memory: allocation })
    }

    fn free_slab(&mut self, slab: Self::MemorySlab) {
        unsafe { self.device.logical().free_memory(slab.memory, None) };
    }

    fn total_dedicated(&self) -> u64 {
        self.total_dedicated_memory
    }

    fn total_shared(&self) -> u64 {
        self.total_shared_memory
    }
}

#[cfg(test)]
mod tests {
    use super::AllocatorVulkan;
    use crate::graphics::device::FeatureSet;
    use crate::graphics::error::GraphicError;
    use crate::graphics::memory::{MemoryAllocatorInternal, MemoryType};
    use crate::graphics::DeviceVulkan;

    #[test]
    fn create_internal_allocator() -> Result<(), GraphicError> {
        let device = DeviceVulkan::new(None, FeatureSet::Convert)?;
        let _allocator = AllocatorVulkan::new(&device)?;
        Ok(())
    }

    #[test]
    fn allocate_and_free_dedicated() -> Result<(), GraphicError> {
        let device = DeviceVulkan::new(None, FeatureSet::Convert)?;
        let mut allocator = AllocatorVulkan::new(&device)?;
        let slab = allocator.allocate_slab(MemoryType::Dedicated, 1048576)?;
        allocator.free_slab(slab);
        Ok(())
    }

    #[test]
    fn allocation_dedicated_too_big() -> Result<(), GraphicError> {
        let device = DeviceVulkan::new(None, FeatureSet::Convert)?;
        let mut allocator = AllocatorVulkan::new(&device)?;
        let maybe_slab = allocator.allocate_slab(MemoryType::Dedicated, u64::MAX);
        assert!(maybe_slab.is_err());
        Ok(())
    }

    #[test]
    fn allocate_and_free_shared() -> Result<(), GraphicError> {
        let device = DeviceVulkan::new(None, FeatureSet::Convert)?;
        let mut allocator = AllocatorVulkan::new(&device)?;
        let slab = allocator.allocate_slab(MemoryType::Shared, 1048576)?;
        allocator.free_slab(slab);
        Ok(())
    }

    #[test]
    fn allocation_shared_too_big() -> Result<(), GraphicError> {
        let device = DeviceVulkan::new(None, FeatureSet::Convert)?;
        let mut allocator = AllocatorVulkan::new(&device)?;
        let maybe_slab = allocator.allocate_slab(MemoryType::Shared, u64::MAX);
        assert!(maybe_slab.is_err());
        Ok(())
    }
}
