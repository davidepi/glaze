// this entire file WAS based on https://vkguide.dev/docs/extra-chapter/abstracting_descriptors/

use ash::vk;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::{Arc, Mutex};

const MAX_SETS: usize = 512;
const MAX_POOLS: usize = 4;

#[derive(Debug, Copy, Clone)]
pub struct Descriptor {
    pub set: vk::DescriptorSet,
    pub layout: vk::DescriptorSetLayout,
}

struct DescriptorAllocator {
    current_pool: Option<vk::DescriptorPool>,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    free_pools: Vec<vk::DescriptorPool>,
    used_pools: Vec<vk::DescriptorPool>,
    device: Arc<ash::Device>,
}

impl DescriptorAllocator {
    fn new(
        device: Arc<ash::Device>,
        avg_desc: &[(vk::DescriptorType, f32)],
    ) -> DescriptorAllocator {
        let pool_sizes = avg_desc
            .iter()
            .map(|(ty, avg)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: (*avg * MAX_SETS as f32).ceil() as u32,
            })
            .collect::<Vec<_>>();
        let mut free_pools = (0..MAX_POOLS)
            .map(|_| create_descriptor_pool(&device, &pool_sizes))
            .collect::<Vec<_>>();
        let current_pool = free_pools.pop();
        DescriptorAllocator {
            current_pool,
            pool_sizes,
            free_pools,
            used_pools: Vec::with_capacity(MAX_POOLS),
            device,
        }
    }

    fn alloc(&mut self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let res = if let Some(descriptor_pool) = self.current_pool {
            let layouts = [layout];
            let alloc_ci = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: layouts.as_ptr(),
            };
            unsafe { self.device.allocate_descriptor_sets(&alloc_ci) }
        } else {
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
        };
        match res {
            Ok(mut desc) => desc.pop().unwrap(),
            Err(vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                let new_pool = self
                    .free_pools
                    .pop()
                    .unwrap_or_else(|| create_descriptor_pool(&self.device, &self.pool_sizes));
                if let Some(pool) = self.current_pool.take() {
                    self.used_pools.push(pool);
                }
                self.current_pool = Some(new_pool);
                self.alloc(layout)
            }
            _ => panic!("Failed to allocate descriptor set"),
        }
    }

    fn thread_exclusive(&mut self) -> Self {
        DescriptorAllocator {
            current_pool: self.free_pools.pop(),
            pool_sizes: self.pool_sizes.clone(),
            free_pools: Vec::new(),
            used_pools: Vec::new(),
            device: self.device.clone(),
        }
    }

    fn reset_pools(&mut self, reset_current: bool) {
        self.used_pools.iter().for_each(|pool| {
            unsafe {
                self.device
                    .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())
            }
            .expect("Failed to reset descriptor pool")
        });
        self.free_pools.append(&mut self.used_pools);
        if reset_current {
            if let Some(pool) = self.current_pool.take() {
                unsafe {
                    self.device
                        .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                }
                .expect("Failed to reset descriptor pool");
            }
        }
    }

    fn merge(&mut self, mut other: Self) {
        if let Some(other_current) = other.current_pool.take() {
            self.used_pools.push(other_current);
        }
        self.used_pools.append(&mut other.used_pools);
        self.free_pools.append(&mut other.free_pools);
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        self.free_pools
            .drain(..)
            .chain(self.used_pools.drain(..))
            .for_each(|pool| unsafe {
                self.device.destroy_descriptor_pool(pool, None);
            });
        if let Some(pool) = self.current_pool.take() {
            unsafe { self.device.destroy_descriptor_pool(pool, None) };
        }
    }
}

fn create_descriptor_pool(
    device: &ash::Device,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> vk::DescriptorPool {
    let pool_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DescriptorPoolCreateFlags::empty(),
        max_sets: MAX_SETS as u32,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
    };

    unsafe { device.create_descriptor_pool(&pool_info, None) }
        .expect("Failed to allocate descriptor pool")
}
// goes around Rust orphan rule and allows implementing Eq and Hash
// remove also the pointer from DescriptorSetLayoutBinding to have trait "Send"
#[derive(Debug, Clone)]
struct DescriptorSetLayoutBindingWrapper {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}

impl DescriptorSetLayoutBindingWrapper {
    pub fn new(dsbin: vk::DescriptorSetLayoutBinding) -> DescriptorSetLayoutBindingWrapper {
        debug_assert!(
            dsbin.p_immutable_samplers.is_null(),
            "immutable samplers not supported"
        );
        DescriptorSetLayoutBindingWrapper {
            binding: dsbin.binding,
            descriptor_type: dsbin.descriptor_type,
            descriptor_count: dsbin.descriptor_count,
            stage_flags: dsbin.stage_flags,
        }
    }
}

impl PartialEq for DescriptorSetLayoutBindingWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.binding == other.binding
            && self.descriptor_count == other.descriptor_count
            && self.descriptor_type == other.descriptor_type
            && self.stage_flags == other.stage_flags
    }
}

impl Eq for DescriptorSetLayoutBindingWrapper {}

impl Hash for DescriptorSetLayoutBindingWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.binding.hash(state);
        self.descriptor_count.hash(state);
        self.descriptor_type.hash(state);
        self.stage_flags.hash(state);
    }
}

// DescriptorSetLayout uses pointers and, in my code, is created in a different function than the
// build one. Of course, in the optimized build, the target of this pointers is deallocated so I'm
// forced to store it in this wrapper and build the DescriptorSetLayoutWrite function in the
// DescriptorSetBuilder::build()
#[derive(Debug, Copy, Clone)]
enum BufOrImgInfo {
    Buf(vk::DescriptorBufferInfo),
    Img(vk::DescriptorImageInfo),
}

struct DescriptorSetLayoutCache {
    cache: HashMap<Vec<DescriptorSetLayoutBindingWrapper>, vk::DescriptorSetLayout>,
    device: Arc<ash::Device>,
}

impl DescriptorSetLayoutCache {
    fn new(device: Arc<ash::Device>) -> DescriptorSetLayoutCache {
        DescriptorSetLayoutCache {
            cache: HashMap::new(),
            device,
        }
    }

    fn get(&mut self, desc: &[vk::DescriptorSetLayoutBinding]) -> vk::DescriptorSetLayout {
        let wrapping = desc
            .iter()
            .cloned()
            .map(DescriptorSetLayoutBindingWrapper::new)
            .collect::<Vec<_>>();
        // wrapping.sort_by_key(|d| d.val.binding); // should be sorted already
        match self.cache.entry(wrapping) {
            std::collections::hash_map::Entry::Occupied(val) => *val.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let ci = vk::DescriptorSetLayoutCreateInfo {
                    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    p_next: ptr::null(),
                    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: desc.len() as u32,
                    p_bindings: desc.as_ptr(),
                };
                let layout = unsafe { self.device.create_descriptor_set_layout(&ci, None) }
                    .expect("Failed to create Descriptor Set Layout");
                entry.insert(layout);
                layout
            }
        }
    }
}

impl Drop for DescriptorSetLayoutCache {
    fn drop(&mut self) {
        self.cache.iter().for_each(|(_, layout)| unsafe {
            self.device.destroy_descriptor_set_layout(*layout, None)
        });
    }
}

pub struct DescriptorSetCreator {
    cache: Arc<Mutex<DescriptorSetLayoutCache>>,
    alloc: DescriptorAllocator,
}

impl DescriptorSetCreator {
    pub fn new(
        device: Arc<ash::Device>,
        avg_desc: &[(vk::DescriptorType, f32)],
    ) -> DescriptorSetCreator {
        let alloc = DescriptorAllocator::new(device.clone(), avg_desc);
        let cache = Arc::new(Mutex::new(DescriptorSetLayoutCache::new(device)));
        DescriptorSetCreator { cache, alloc }
    }

    pub fn new_set(&mut self) -> DescriptorSetBuilder {
        DescriptorSetBuilder {
            alloc: &mut self.alloc,
            cache: self.cache.clone(),
            bindings: Vec::new(),
            info: Vec::new(),
        }
    }

    pub fn thread_exclusive(&mut self) -> Self {
        DescriptorSetCreator {
            cache: self.cache.clone(),
            alloc: self.alloc.thread_exclusive(),
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.alloc.merge(other.alloc);
    }
}

pub struct DescriptorSetBuilder<'a> {
    cache: Arc<Mutex<DescriptorSetLayoutCache>>,
    alloc: &'a mut DescriptorAllocator,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    info: Vec<BufOrImgInfo>,
}

impl<'a> DescriptorSetBuilder<'a> {
    #[must_use]
    pub fn bind_buffer(
        mut self,
        buf_info: vk::DescriptorBufferInfo,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let binding = self.bindings.len() as u32;
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type,
            descriptor_count: 1,
            stage_flags,
            p_immutable_samplers: ptr::null(),
        };
        self.bindings.push(layout_binding);
        self.info.push(BufOrImgInfo::Buf(buf_info));
        self
    }
    #[must_use]
    pub fn bind_image(
        mut self,
        image_info: vk::DescriptorImageInfo,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let binding = self.bindings.len() as u32;
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type,
            descriptor_count: 1,
            stage_flags,
            p_immutable_samplers: ptr::null(),
        };
        self.bindings.push(layout_binding);
        self.info.push(BufOrImgInfo::Img(image_info));
        self
    }

    pub fn build(self) -> Descriptor {
        let layout = self.cache.lock().unwrap().get(&self.bindings);
        let set = self.alloc.alloc(layout);
        let mut writes = Vec::new();
        for i in 0..self.bindings.len() {
            let binding = self.bindings[i];
            let mut write = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: set,
                dst_binding: binding.binding,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: binding.descriptor_type,
                p_image_info: ptr::null(),
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
            };
            match &self.info[i] {
                BufOrImgInfo::Buf(buf) => {
                    write.p_buffer_info = buf;
                }
                BufOrImgInfo::Img(img) => {
                    write.p_image_info = img;
                }
            }
            writes.push(write);
        }
        unsafe {
            self.alloc.device.update_descriptor_sets(&writes, &[]);
        }
        Descriptor { set, layout }
    }
}
