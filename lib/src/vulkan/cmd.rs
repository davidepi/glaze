use std::ptr;

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
            return None;
        } else {
            let retval = self.buffers[self.used as usize];
            self.used += 1;
            Some(retval)
        }
    }

    fn reset(mut self, device: &ash::Device) -> PoolManager {
        unsafe {
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .unwrap()
        };
        self.used = 0;
        self
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
    }
}

pub struct CommandManager {
    current_pool: PoolManager,
    free_pools: Vec<PoolManager>,
    used_pools: Vec<PoolManager>,
    device: ash::Device,
}

impl CommandManager {
    pub fn new(device: ash::Device, queue_family_index: u32) -> CommandManager {
        let mut free_pools = (0..POOL_NO)
            .into_iter()
            .map(|_| PoolManager::create(&device, queue_family_index))
            .collect::<Vec<_>>();
        let current_pool = free_pools.pop().unwrap();
        CommandManager {
            free_pools,
            current_pool,
            used_pools: Vec::with_capacity(POOL_NO),
            device,
        }
    }

    pub fn get_cmd_buffer(&mut self) -> vk::CommandBuffer {
        if let Some(buffer) = self.current_pool.get() {
            // buffer available, from the current pool
            buffer
        } else {
            if let Some(mut next_pool) = self.free_pools.pop() {
                // current pool is full, but there are free pools
                std::mem::swap(&mut self.current_pool, &mut next_pool);
                self.used_pools.push(next_pool);
                self.get_cmd_buffer()
            } else {
                // no free pools available, pop HALF the used one and populate the free pools
                // half -> so the most recent ones have time to complete
                let half = self
                    .used_pools
                    .drain(..POOL_NO / 2)
                    .map(|pool| pool.reset(&self.device))
                    .collect::<Vec<_>>();
                self.free_pools.extend(half);
                self.get_cmd_buffer()
            }
        }
    }

    pub fn destroy(self) {
        self.free_pools
            .into_iter()
            .for_each(|pool| pool.destroy(&self.device));
        self.used_pools
            .into_iter()
            .for_each(|pool| pool.destroy(&self.device));
        self.current_pool.destroy(&self.device);
    }
}
