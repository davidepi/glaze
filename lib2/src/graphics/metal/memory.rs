use crate::graphics::memory::{MemoryAllocatorInternal, MemoryType};

use super::{error::MetalError, MetalDevice};

pub struct MetalMemorySlab {
    inner: metal::Heap,
}

pub struct MetalAllocator<'device> {
    device: &'device MetalDevice,
}

impl<'device> MetalAllocator<'device> {
    pub fn new(device: &'device MetalDevice) -> MetalAllocator<'device> {
        MetalAllocator {
            device
        }
    }
}

impl MemoryAllocatorInternal for MetalAllocator<'_> {

    type MemorySlab = MetalMemorySlab;
    type GraphicError = MetalError;

    fn allocate_slab(&mut self, mtype: MemoryType, size: usize) -> Result<Self::MemorySlab, Self::GraphicError> {
        let descriptor = metal::HeapDescriptor::new();
        let storage_mode = match mtype {
            MemoryType::Shared => metal::MTLStorageMode::Shared,
            MemoryType::Dedicated => metal::MTLStorageMode::Private,
        };
        descriptor.set_storage_mode(storage_mode);
        descriptor.set_size(size as u64);
        let inner = self.device.inner().new_heap(&descriptor);
        let slab = MetalMemorySlab {
            inner,
        };
        Ok(slab)
    }

    fn free_slab(&mut self, _slab: Self::MemorySlab) {
        // just drop the slab
    }
}
