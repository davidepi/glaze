use std::ptr;
use std::sync::Arc;

use ash::vk;

const POOL_NO: usize = 15;
const BUFFERS_PER_POOL: u8 = 30;

struct PoolManager {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    used: u8,
}

impl PoolManager {
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

    fn get(&mut self) -> Option<vk::CommandBuffer> {
        if self.used == BUFFERS_PER_POOL {
            None
        } else {
            let retval = self.buffers[self.used as usize];
            self.used += 1;
            Some(retval)
        }
    }

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

pub struct CommandManager {
    current_pool: Option<PoolManager>,
    free_pools: Vec<PoolManager>,
    used_pools: Vec<PoolManager>,
    device: Arc<ash::Device>,
    queue_family_index: u32,
}

impl CommandManager {
    pub fn new(device: Arc<ash::Device>, queue_family_index: u32) -> CommandManager {
        let mut free_pools = (0..POOL_NO)
            .into_iter()
            .map(|_| PoolManager::create(&device, queue_family_index))
            .collect::<Vec<_>>();
        let current_pool = free_pools.pop();
        CommandManager {
            free_pools,
            current_pool,
            used_pools: Vec::with_capacity(POOL_NO),
            device,
            queue_family_index,
        }
    }

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
        } else if self.used_pools.len() > POOL_NO / 2 {
            // no free pools, but there are a decent amount of used one. Resets half of them
            // half -> so the most recent ones have time to complete
            let half = self
                .used_pools
                .drain(..POOL_NO / 2)
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

    pub fn thread_exclusive(&mut self) -> Self {
        CommandManager {
            current_pool: self.free_pools.pop(), // the manager will auto expand if this is none
            free_pools: Vec::new(),
            used_pools: Vec::new(),
            device: self.device.clone(),
            queue_family_index: self.queue_family_index,
        }
    }

    pub fn merge(&mut self, mut other: Self) {
        if let Some(other_current) = other.current_pool.take() {
            self.used_pools.push(other_current);
        }
        self.used_pools.append(&mut other.used_pools);
        self.free_pools.append(&mut other.free_pools);
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
