use ash::vk;
use std::ptr;
use std::sync::Arc;

/// Minimum number of command pools before the reset happens.
/// Pools are never resets if the manager has less than this amount of USED pools.
const MIN_POOLS: usize = 15;
/// Number of command buffers per command pool.
const BUFFERS_PER_POOL: u8 = 30;

/// Manages a single command pool.
struct PoolManager {
    /// The managed command pool.
    pool: vk::CommandPool,
    /// Command buffers in this pool.
    buffers: Vec<vk::CommandBuffer>,
    /// Number of served command buffers.
    used: u8,
}

impl PoolManager {
    /// Creates a new command pool for a given device and queue family
    fn create(device: &ash::Device, queue_family_index: u32) -> PoolManager {
        let pool_ci = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index,
        };
        let pool = unsafe {
            device
                .create_command_pool(&pool_ci, None)
                .expect("Failed to allocate command pool")
        };
        let alloc_ci = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: BUFFERS_PER_POOL as u32,
        };
        let buffers = unsafe {
            device
                .allocate_command_buffers(&alloc_ci)
                .expect("Failed to allocate command buffers")
        };
        PoolManager {
            pool,
            buffers,
            used: 0,
        }
    }

    /// Returns a command buffer from the pool if there is one available. None otherwise.
    fn get(&mut self) -> Option<vk::CommandBuffer> {
        if self.used == BUFFERS_PER_POOL {
            None
        } else {
            let retval = self.buffers[self.used as usize];
            self.used += 1;
            Some(retval)
        }
    }

    /// Resets the pool and invalidates all command buffers.
    #[must_use]
    fn reset(mut self, device: &ash::Device) -> PoolManager {
        unsafe {
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .unwrap()
        };
        self.used = 0;
        self
    }

    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Manages a set of command pools.
pub struct CommandManager {
    /// The current pool being used to distribute buffers around.
    current_pool: Option<PoolManager>,
    /// List of pools not completely used yet (likely empty).
    free_pools: Vec<PoolManager>,
    /// List of pools completely used.
    used_pools: Vec<PoolManager>,
    /// Device being used.
    device: Arc<ash::Device>,
    /// Queue family for which the command pools are allocated.
    queue_family_index: u32,
}

impl CommandManager {
    /// Creates a new command manager for a given device and queue family.
    /// Pre-allocates the given number of command pools.
    /// Additional pools are allocated on demand.
    pub fn new(device: Arc<ash::Device>, queue_family_index: u32, num_pools: usize) -> Self {
        let mut free_pools = (0..num_pools)
            .into_iter()
            .map(|_| PoolManager::create(&device, queue_family_index))
            .collect::<Vec<_>>();
        let current_pool = free_pools.pop();
        CommandManager {
            free_pools,
            current_pool,
            used_pools: Vec::with_capacity(num_pools),
            device,
            queue_family_index,
        }
    }

    /// Returns a command buffer from the pool. Changes and resets pools if necessary.
    pub fn get_cmd_buffer(&mut self) -> vk::CommandBuffer {
        if let Some(pool) = &mut self.current_pool {
            // a pool is available
            if let Some(buffer) = pool.get() {
                // buffer available, from the current pool
                buffer
            } else {
                // current pool is full, replace with none and repeat recursion
                self.used_pools.push(self.current_pool.take().unwrap());
                self.get_cmd_buffer()
            }
        } else if let Some(next_pool) = self.free_pools.pop() {
            // there is a free pool available
            self.current_pool = Some(next_pool);
            self.get_cmd_buffer()
        } else if self.used_pools.len() > MIN_POOLS / 2 {
            // no free pools, but there are a decent amount of used one. Resets half of them
            // half -> so the most recent ones have time to complete
            let half = self
                .used_pools
                .drain(..MIN_POOLS / 2)
                .map(|pool| pool.reset(&self.device))
                .collect::<Vec<_>>();
            self.free_pools.extend(half);
            self.get_cmd_buffer()
        } else {
            // no free pools, and no pools available. This can happen after a call to
            // thread_exclusive_manager. Just create a new one. Only one because this call is
            // EXTREMELY unlikely (or so do I think until I profile the application)
            self.free_pools
                .push(PoolManager::create(&self.device, self.queue_family_index));
            self.get_cmd_buffer()
        }
    }
}

impl Drop for CommandManager {
    fn drop(&mut self) {
        self.free_pools
            .drain(..)
            .for_each(|pool| pool.destroy(&self.device));
        self.used_pools
            .drain(..)
            .for_each(|pool| pool.destroy(&self.device));
        if let Some(pool) = self.current_pool.take() {
            pool.destroy(&self.device);
        }
    }
}
