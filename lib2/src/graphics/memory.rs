use super::error::{ErrorCategory, GraphicError};
use crate::math::util::{ilog2_ceil, Stats};
use crate::util::collections::IndexVec;
use std::backtrace;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

pub trait MemoryAllocatorInternal {
    type MemorySlab;

    // both metal and vulkan do not specify thread safety for this one, so I will enforce it
    // by myself
    fn allocate_slab(
        &mut self,
        memory_type: MemoryType,
        size: u64,
    ) -> Result<Self::MemorySlab, GraphicError>;
    fn free_slab(&mut self, slab: Self::MemorySlab);
    fn total_dedicated(&self) -> u64;
    fn total_shared(&self) -> u64;
}

#[derive(Debug, Copy, Clone)]
pub enum MemoryType {
    Shared,
    Dedicated,
}

/// A (sub)allocation performed by the [BuddyAllocator]
///
///  A `BuddyBlock` is a data structure used by the [BuddyAllocator] to represent a suballocation
/// offset within the managed memory. It encapsulates information about the position and size
/// of the allocated block within the memory managed by the Buddy Allocator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BuddyBlock {
    /// offset in the memory block. Must be first to correctly allow ordering by offset.
    offset: usize,
    /// 2^x bytes for this block.
    order: u8,
    /// True if this buddy is the left one.
    /// Although I can gather this info from the offset, this byte would be wasted by padding
    /// anyway
    left_buddy: bool,
    /// the index of the main memory block.
    memory_index: u32,
}

impl BuddyBlock {
    /// Generates the companion block (the "buddy"), starting from this one
    fn other_buddy(&self) -> BuddyBlock {
        let mut other = *self;
        let size = 0x1_usize << self.order;
        other.left_buddy = !other.left_buddy;
        if self.left_buddy {
            other.offset += size;
        } else {
            other.offset -= size;
        }
        other
    }

    /// Returns the offset within the memory page.
    ///
    /// The offset indicates the position of this buddy block within the main memory page.
    ///
    /// # Returns
    ///
    /// The offset as a `usize` value.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the size of the block in bytes.
    ///
    /// # Returns
    ///
    /// The size of the block in byts as a `usize` value.
    pub fn size(&self) -> usize {
        0x1_usize << self.order
    }
}

/// Buddy Allocator for Memory Management
///
/// The `BuddyAllocator` is a memory management system that allocates and frees memory in blocks,
/// where each block size is a power of two. It efficiently manages memory by dividing it into
/// fixed-size blocks and coalesces adjacent free blocks to minimize fragmentation.
///
/// This allocator operates in pages of `max_order` size, where `max_order` is defined in the
/// program configuration. Memory pages are allocated by an underlying allocator provided during
/// construction. This allocator is API specific and must implement MemoryAllocatorInternal.
///
/// This Buddy allocator manages a single type of memory: either Host memory or Device memory,
/// specified at construction time.
pub struct BuddyAllocator<T: MemoryAllocatorInternal> {
    block_allocator: T,
    memory_type: MemoryType,
    free_blocks: Vec<BTreeSet<BuddyBlock>>,
    // not actually needed to work, so track only the blocks external to the allocator.
    used_blocks: HashSet<BuddyBlock>,
    heap_blocks: IndexVec<T::MemorySlab>,
    // maximum size of a memory_block
    max_order: u8,
    // minimum size of an allocation
    min_order: u8,
}

impl<T: MemoryAllocatorInternal> BuddyAllocator<T> {
    /// Create a new `BuddyAllocator` instance.
    ///
    /// This method initializes a new `BuddyAllocator` that manages memory using the provided
    /// `allocator`. Memory is organized into pages of a size defined by 2^`max_order`, which is
    /// specified by the program configuration. The `memory_type` parameter allows to
    /// choose between Host memory or Device memory, based on the provided `MemoryType` enum.
    /// It is possible to specify the allocation alignment by using the `min_order` parameter.
    ///
    /// # Parameters
    ///
    /// - `allocator`: An underlying memory allocator that implements the `MemoryAllocatorInternal`
    ///   trait. This allocator is responsible for allocating memory pages.
    /// - `memory_type`: An enum value indicating the type of memory to allocate.
    /// - `max_order`: The logarithm base 2 of the maximum memory page size requested to the
    ///   underlying memory allocator.
    /// - `min_order`: The logarithm base 2 of the minimum size of each allocator. This implies
    ///   that any allocation will be aligned to this value.
    ///
    /// # Returns
    ///
    /// A new `BuddyAllocator` instance for managing memory.
    pub fn new(allocator: T, memory_type: MemoryType, max_order: u8, min_order: u8) -> Self {
        Self {
            block_allocator: allocator,
            free_blocks: vec![BTreeSet::new(); (max_order - min_order) as usize],
            used_blocks: HashSet::default(),
            heap_blocks: IndexVec::new(),
            max_order,
            min_order,
            memory_type,
        }
    }

    /// Allocate a memory block of the specified size.
    ///
    /// This method allocates a memory block of the given `size`. The `size` must be a power of two
    /// and not exceed 2^`max_order`, where `max_order` is defined during `BuddyAllocator`
    /// initialization.
    ///
    /// # Parameters
    ///
    /// - `size`: The size of the memory block to allocate. Must be a power of two and within the
    ///   maximum size specified by `max_order`.
    ///
    /// # Returns
    ///
    /// - `Ok(BuddyBlock)`: A `BuddyBlock` representing the allocated memory block.
    /// - `Err(GraphicError)`: An error is returned if the `size` exceeds 2^max_order or if the
    ///   memory has been exhausted.
    ///
    /// # Note
    ///
    /// The `BuddyBlock` returned by this method does not contain the actual allocated memory area
    /// but provides information about the offsets of the suballocation within the memory pages.
    pub fn allocate(&mut self, mut size: usize) -> Result<BuddyBlock, GraphicError> {
        // minimum alignment
        if size < (1_usize << self.min_order) {
            size = 1_usize << self.min_order;
        }
        let requested_order = ilog2_ceil(size as u64);
        let block = self.grab_block(requested_order)?;
        self.used_blocks.insert(block);
        Ok(block)
    }

    /// Recursive call to obtain the correct buddy and split all the bigger ones.
    fn grab_block(&mut self, order: u8) -> Result<BuddyBlock, GraphicError> {
        if order > self.max_order {
            let category = match self.memory_type {
                MemoryType::Shared => ErrorCategory::HostMemory,
                MemoryType::Dedicated => ErrorCategory::DeviceMemory,
            };
            Err(GraphicError::new(
                category,
                format!(
                    "Allocation too big. Requesting {} bytes, but the maximum allowed is {} bytes.",
                    0x1 << order,
                    0x1 << self.max_order
                ),
            ))
        } else if order == self.max_order {
            // terminating condition: request an entire heap block from the allocator
            let heap_block = self
                .block_allocator
                .allocate_slab(self.memory_type, 0x1 << self.max_order)?;
            let block_index = self.heap_blocks.insert(heap_block) as u32;
            let entire_block = BuddyBlock {
                offset: 0,
                order: self.max_order,
                memory_index: block_index,
                left_buddy: true,
            };
            Ok(entire_block)
        } else if let Some(block) = self.free_blocks[(order - self.min_order) as usize].pop_first()
        {
            // a free block of the correct order is present: pop the one with lowest offset
            Ok(block)
        } else {
            // no available block, need to split: recurse and create the two buddies. Add the right
            // one to the list of free blocks and return the left one.
            let parent_block = self.grab_block(order + 1)?;
            // remove the parent_block from the used blocks because I am the one using it.
            let left_buddy = BuddyBlock {
                offset: parent_block.offset,
                order,
                left_buddy: true,
                memory_index: parent_block.memory_index,
            };
            let right_buddy = left_buddy.other_buddy();
            self.free_blocks[(order - self.min_order) as usize].insert(right_buddy);
            Ok(left_buddy)
        }
    }

    /// Free a previously allocated memory block.
    ///
    /// This method frees a memory block represented by the provided `BuddyBlock`. An error is
    /// returned if the `BuddyBlock` has not been allocated or has already been freed.
    ///
    /// # Parameters
    ///
    /// - `block`: A `BuddyBlock` representing the memory block to be freed.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: The memory block has been successfully freed.
    /// - `Err(GraphicError)`: An error is returned if the `BuddyBlock` has not been allocated or
    ///   has already been freed.
    pub fn free(&mut self, block: BuddyBlock) -> Result<(), GraphicError> {
        if !self.used_blocks.remove(&block) {
            let category = match self.memory_type {
                MemoryType::Shared => ErrorCategory::HostMemory,
                MemoryType::Dedicated => ErrorCategory::DeviceMemory,
            };
            Err(GraphicError::new(
                category,
                "Failed to find allocated block. Double free?",
            ))
        } else {
            self.release(block);
            Ok(())
        }
    }

    /// Recursive internal function to free and coalesce blocks.
    fn release(&mut self, block: BuddyBlock) {
        let other_buddy = block.other_buddy();
        if block.order == self.max_order {
            // terminating condition: release the entire heap block
            let heap_block = self.heap_blocks.pop(block.memory_index as usize).expect(
                "Failed to find allocated heap block. This indicates an internal memory error.",
            );
            self.block_allocator.free_slab(heap_block);
        } else if self.free_blocks[(block.order - self.min_order) as usize].remove(&other_buddy) {
            // buddy is present: coalesce them and recurse with the coalesced block
            let mut merged_block = block.min(other_buddy); // keep the one with lowest offset
            merged_block.left_buddy = true;
            merged_block.order += 1;
            // add the merged block to the used ones: will be removed in the recursive call
            self.release(merged_block);
        } else {
            // buddy can not be found: add the current block as free block
            self.free_blocks[(block.order - self.min_order) as usize].insert(block);
        }
    }

    /// Returns the average fragmentation across all memory heaps managed by the buddy allocator.
    ///
    /// Fragmentation is a measure of memory wastage and inefficiency. This method calculates the
    /// average fragmentation across all memory heaps and returns it as a `Stats` object.
    ///
    /// Fragmentation is calculated using the formula: `1 - (max_contiguous / total_free)`,
    /// where `max_contiguous` represents the largest contiguous block of free memory, and
    /// `total_free` is the total amount of free memory.
    ///
    /// A lower fragmentation value indicates better memory utilization, while higher values suggest
    /// more fragmentation and potential memory wastage.
    ///
    /// # Returns
    ///
    /// A `Stats` object containing information about the average fragmentation.
    pub fn fragmentation(&self) -> Stats {
        let mut max_contiguous = vec![0.0; self.heap_blocks.len()];
        let mut total_free = vec![0.0; self.heap_blocks.len()];
        for free_block_by_order in &self.free_blocks {
            for block in free_block_by_order {
                let midx = block.memory_index as usize;
                max_contiguous[midx] =
                    f32::max(max_contiguous[midx], (0x1_u64 << block.order) as f32);
                total_free[midx] += block.size() as f32;
            }
        }
        let fragmentation = max_contiguous
            .into_iter()
            .zip(total_free)
            .map(|(max, total)| {
                if total != 0.0 {
                    1.0 - (max / total)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        Stats::new(&fragmentation)
    }
}

impl<T: MemoryAllocatorInternal> Drop for BuddyAllocator<T> {
    #[cfg(debug_assertions)]
    fn drop(&mut self) {
        if !self.used_blocks.is_empty() {
            log::warn!(
                "Freeing buddy allocator while it is still managing memory. Backtrace: {}",
                backtrace::Backtrace::capture()
            );
        }
    }
}

pub struct GpuAllocation<T> {
    slab: Arc<T>,
    offset: u32,
    size: u32,
}

impl<T> GpuAllocation<T> {
    pub fn slab(&self) -> &T {
        &self.slab
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn size(&self) -> u32 {
        self.size
    }
}

pub struct GpuAllocator<T: MemoryAllocatorInternal> {
    allocator: T,
    slabs: Vec<T::MemorySlab>,
    next_free: Vec<Vec<u32>>,
    used: HashSet<u32>,
}

#[cfg(test)]
mod tests {
    use super::{MemoryAllocatorInternal, MemoryType};
    use crate::graphics::error::{ErrorCategory, GraphicError};
    use crate::graphics::memory::{BuddyAllocator, BuddyBlock};
    use float_cmp::assert_approx_eq;
    use std::collections::HashSet;

    struct FakeAllocator {
        max_dedicated: u64,
        used_dedicated: u64,
        max_shared: u64,
        used_shared: u64,
    }
    impl FakeAllocator {
        pub fn new(max_dedicated: u64, max_shared: u64) -> Self {
            Self {
                max_dedicated,
                used_dedicated: 0,
                max_shared,
                used_shared: 0,
            }
        }
    }
    struct FakeHeap {
        size: u64,
        memory_type: MemoryType,
    }
    impl MemoryAllocatorInternal for FakeAllocator {
        type MemorySlab = FakeHeap;

        fn allocate_slab(
            &mut self,
            memory_type: MemoryType,
            size: u64,
        ) -> Result<Self::MemorySlab, GraphicError> {
            match memory_type {
                MemoryType::Shared => {
                    if self.used_shared + size > self.max_shared {
                        Err(GraphicError::new(
                            ErrorCategory::HostMemory,
                            "not enough memory",
                        ))
                    } else {
                        self.used_shared += size;
                        Ok(Self::MemorySlab { size, memory_type })
                    }
                }
                MemoryType::Dedicated => {
                    if self.used_dedicated + size > self.max_dedicated {
                        Err(GraphicError::new(
                            ErrorCategory::DeviceMemory,
                            "not enough memory",
                        ))
                    } else {
                        self.used_dedicated += size;
                        Ok(Self::MemorySlab { size, memory_type })
                    }
                }
            }
        }

        fn free_slab(&mut self, slab: Self::MemorySlab) {
            match slab.memory_type {
                MemoryType::Shared => self.used_shared -= slab.size,
                MemoryType::Dedicated => self.used_dedicated -= slab.size,
            }
        }

        fn total_dedicated(&self) -> u64 {
            self.max_dedicated
        }

        fn total_shared(&self) -> u64 {
            self.max_shared
        }
    }

    #[test]
    /// Basic usage of the allocator: allocates some blocks and frees them. Every allocation should
    /// be a power of two.
    fn buddy_allocation_deallocation() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(11), 2_u64.pow(11));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 11, 0);
        // Allocate a block and check its correctness
        let block_size = 64;
        let allocated_block = ballocator.allocate(block_size)?;
        assert_eq!(allocated_block.size(), block_size);
        // Non power of two allocation
        let non_power_of_two_size = 1000;
        let result = ballocator.allocate(non_power_of_two_size)?;
        assert_eq!(result.size(), 1024);
        // Allocate multiple blocks and verify correctness
        let block_sizes = [32, 64, 128, 256];
        let mut allocated_blocks = Vec::new();
        for &size in &block_sizes {
            let block = ballocator.allocate(size)?;
            assert_eq!(block.size(), size);
            allocated_blocks.push(block);
        }
        // Deallocate a block and check that it's reusable
        let block_to_free = allocated_blocks.pop().unwrap();
        ballocator.free(block_to_free)?;
        let recycled_block = ballocator.allocate(block_size)?;
        assert_eq!(recycled_block.size(), block_size);
        Ok(())
    }

    #[test]
    /// Allocating blocks should increase fragmentation, deallocating all of them should reduce it.
    fn buddy_fragmentation() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(30), 2_u64.pow(30));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 28, 0);
        // Allocate blocks to induce fragmentation
        let block_sizes = [32, 32];
        let mut allocated_blocks = Vec::new();
        for &size in &block_sizes {
            let block = ballocator.allocate(size * 1024 * 1024)?;
            allocated_blocks.push(block);
        }
        // Check fragmentation should be ~0.3 (128M max allocation, 192M free)
        let fragmentation = ballocator.fragmentation();
        assert_approx_eq!(f32, fragmentation.mean(), 0.3333333, epsilon = 1e-5);
        // Free all allocated blocks and check that fragmentation returns to 0.0
        for block in allocated_blocks {
            ballocator.free(block)?;
        }
        let final_stats = ballocator.fragmentation();
        assert_eq!(final_stats.mean(), 0.0);
        Ok(())
    }

    #[test]
    /// Allocating a page larger than the entire system memory should fail.
    fn buddy_maximum_memory_page() {
        let internal = FakeAllocator::new(2_u64.pow(5), 2_u64.pow(5));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 28, 0);
        let oversized_block = ballocator.allocate(2_usize.pow(20));
        assert!(oversized_block.is_err()); // Allocation should fail
    }

    #[test]
    // Allocating more memory than the page size should fail.
    fn buddy_maximum_memory_allocation() {
        let internal = FakeAllocator::new(2_u64.pow(30), 2_u64.pow(30));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 28, 0);
        // Attempt to allocate a block larger than the maximum memory page size
        let max_page_size = 2_usize.pow(28);
        let oversized_block = ballocator.allocate(max_page_size + 1);
        assert!(oversized_block.is_err()); // Allocation should fail
    }

    #[test]
    /// Allocating less than min_order bytes should return min_order bytes.
    fn buddy_minimum_memory_allocation() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(5), 2_u64.pow(5));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 5, 3);
        // Perform an allocation smaller than 2^min_order
        let allocated_block = ballocator.allocate(4)?;
        // Verify that the allocated block size is 2^min_order
        assert_eq!(allocated_block.size(), 8);
        Ok(())
    }

    #[test]
    /// Every allocation should be aligned to a `min_order` boundary.
    fn buddy_alignment() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(18), 2_u64.pow(18));
        let min_order = 5;
        let alignment = (0x1 << min_order) as usize;
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 18, min_order);
        // Allocate various block of different size
        let block_size = vec![64, 15, 125, 500, 320, 640, 1080, 151, 33, 87];
        let allocated_blocks = block_size
            .into_iter()
            .map(|size| ballocator.allocate(size))
            .collect::<Result<Vec<_>, GraphicError>>()?;
        // Verify that all blocks are aligned to min_order
        for block in allocated_blocks {
            assert_eq!(block.offset() % alignment, 0);
        }
        Ok(())
    }

    #[test]
    /// Allocating the smallest and biggest buddies possible should not go out of bounds.
    fn buddy_first_order_and_last_order_buddies() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(5), 2_u64.pow(5));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 5, 3);
        // Perform an allocation equal to max_order
        let allocated_block = ballocator.allocate(2_usize.pow(5))?;
        assert_eq!(allocated_block.size(), 2_usize.pow(5));
        ballocator.free(allocated_block)?;
        // Perform an allocation equal to min_order
        let allocated_block = ballocator.allocate(2_usize.pow(3))?;
        assert_eq!(allocated_block.size(), 2_usize.pow(3));
        ballocator.free(allocated_block)?;
        // Create a new BuddyAllocator with min_order 0 and assert that it's possible to allocate
        // 1-byte blocks ans 0-bytes blocks.
        let internal = FakeAllocator::new(2_u64.pow(5), 2_u64.pow(5));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 3, 0);
        let allocated_block = ballocator.allocate(2_usize.pow(0))?;
        assert_eq!(allocated_block.size(), 2_usize.pow(0));
        ballocator.free(allocated_block)?;
        let allocated_block = ballocator.allocate(0)?;
        assert_eq!(allocated_block.size(), 1);
        ballocator.free(allocated_block)?;
        Ok(())
    }

    #[test]
    /// Freeing adjacent blocks should merge them into a bigger one.
    fn buddy_freeing_and_coalescing() -> Result<(), GraphicError> {
        // Limit memory to 1 page of 256 bytes in this one.
        let internal = FakeAllocator::new(2_u64.pow(8), 2_u64.pow(8));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 8, 0);
        // Allocate 4 blocks of 64 bytes to fill the memory
        let block_size = 64;
        let mut allocated_blocks = Vec::new();
        for _ in 0..4 {
            let block = ballocator.allocate(block_size)?;
            allocated_blocks.push(block);
        }
        // Assert that no more memory can be allocated (no fragmentation)
        let alloc_failed = ballocator.allocate(block_size);
        assert!(alloc_failed.is_err());
        assert_eq!(ballocator.fragmentation().mean(), 0.0);
        // Deallocate two blocks to induce coalescing. All the free memory should be merged into a
        // single block and thus fragmentation should remain 0.0
        ballocator.free(allocated_blocks.pop().unwrap())?;
        ballocator.free(allocated_blocks.pop().unwrap())?;
        assert_eq!(ballocator.fragmentation().mean(), 0.0);
        // Allocate a new block, which should coalesce with the freed blocks and be twice the size
        let coalesced_block = ballocator.allocate(block_size * 2)?;
        assert_eq!(coalesced_block.size(), block_size * 2);
        Ok(())
    }

    #[test]
    /// Multiple memory pages should be used if one is not sufficient, for allocations smaller than
    /// a page size.
    fn buddy_multiple_memory_pages() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(30), 2_u64.pow(30));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 26, 0);
        let block_size = 64 * 1024 * 1024;
        // Allocate blocks until the memory is exhausted
        let mut allocated_blocks = Vec::new();
        while let Ok(block) = ballocator.allocate(block_size) {
            allocated_blocks.push(block);
        }
        // Ensure something has been allocated
        assert!(!allocated_blocks.is_empty());
        // Check that at least two different pages are used
        let unique_memory_indices = allocated_blocks
            .iter()
            .map(|block| block.memory_index)
            .collect::<HashSet<_>>();
        assert!(unique_memory_indices.len() >= 2);
        Ok(())
    }

    #[test]
    /// Allocation should fail if no memory is available.
    fn buddy_memory_exhausted() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(30), 2_u64.pow(30));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 28, 0);
        let block_size = 64 * 1024 * 1024;
        // Allocate blocks until the memory is exhausted
        let mut allocated_blocks = Vec::new();
        while let Ok(block) = ballocator.allocate(block_size) {
            allocated_blocks.push(block);
        }
        // Attempt to allocate a block, should fail
        let allocation_result = ballocator.allocate(512);
        assert!(allocation_result.is_err());
        Ok(())
    }

    #[test]
    /// Deallocation should fail if the block was never allocated to begin with.
    fn buddy_invalid_deallocation() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(5), 2_u64.pow(5));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 3, 0);
        // Attempt to deallocate a block that was not previously allocated, should result in an error
        let invalid_block = BuddyBlock {
            offset: 0,
            order: 0,
            memory_index: 0,
            left_buddy: true,
        };
        let deallocation_result = ballocator.free(invalid_block);
        assert!(deallocation_result.is_err());
        Ok(())
    }

    #[test]
    /// Deallocation should fail if a block is deallocated a second time.
    fn buddy_double_free() -> Result<(), GraphicError> {
        let internal = FakeAllocator::new(2_u64.pow(5), 2_u64.pow(5));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 3, 0);
        // Attempt to deallocate a block twice
        let allocation = ballocator.allocate(4)?;
        let deallocation_result0 = ballocator.free(allocation);
        assert!(deallocation_result0.is_ok());
        let deallocation_result1 = ballocator.free(allocation);
        assert!(deallocation_result1.is_err());
        Ok(())
    }

    #[test]
    /// Freed memory slots should be recycled.
    fn buddy_memory_recycling() -> Result<(), GraphicError> {
        // Limit memory to 1 page of 256M in this one.
        let internal = FakeAllocator::new(2_u64.pow(28), 2_u64.pow(28));
        let mut ballocator = BuddyAllocator::new(internal, MemoryType::Dedicated, 28, 0);
        let block_size = 2_u64.pow(24) as usize;
        // Allocate blocks until the memory is completely filled
        let mut allocated_blocks = Vec::new();
        while let Ok(block) = ballocator.allocate(block_size) {
            allocated_blocks.push(block);
        }
        // Assert that no extra block can be allocated
        let extra_allocation_result = ballocator.allocate(block_size);
        assert!(extra_allocation_result.is_err());
        // Free two blocks
        ballocator.free(allocated_blocks.pop().unwrap())?;
        ballocator.free(allocated_blocks.pop().unwrap())?;
        // Fill the blank with 4 blocks, half the size of the two blocks removed. This should not fail.
        let smaller_block_size = block_size / 2;
        for _ in 0..4 {
            let block = ballocator.allocate(smaller_block_size)?;
            allocated_blocks.push(block);
        }
        // Assert that fragmentation is again 0 because the memory is completely full
        let final_fragmentation_stats = ballocator.fragmentation();
        assert_eq!(final_fragmentation_stats.mean(), 0.0);
        Ok(())
    }
}
