// this entire file is based on https://vkguide.dev/docs/extra-chapter/abstracting_descriptors/
use ash::vk;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ptr;

const MAX_SETS: usize = 512;
const MAX_POOLS: usize = 4;

#[derive(Debug, Copy, Clone)]
pub struct Descriptor {
    pub set: vk::DescriptorSet,
    pub layout: vk::DescriptorSetLayout,
}

pub struct DescriptorAllocator {
    current_pool: vk::DescriptorPool,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    free_pools: Vec<vk::DescriptorPool>,
    used_pools: Vec<vk::DescriptorPool>,
    device: ash::Device,
}

impl DescriptorAllocator {
    pub fn new(device: ash::Device, avg_desc: &[(vk::DescriptorType, f32)]) -> DescriptorAllocator {
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
        let current_pool = free_pools.pop().unwrap();
        DescriptorAllocator {
            current_pool,
            pool_sizes,
            free_pools,
            used_pools: Vec::with_capacity(MAX_POOLS),
            device,
        }
    }

    pub fn alloc(&mut self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let alloc_ci = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.current_pool,
            descriptor_set_count: 1,
            p_set_layouts: [layout].as_ptr(),
        };
        let res = unsafe { self.device.allocate_descriptor_sets(&alloc_ci) };
        match res {
            Ok(mut desc) => desc.pop().unwrap(),
            Err(vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                let new_pool = self
                    .free_pools
                    .pop()
                    .unwrap_or_else(|| create_descriptor_pool(&self.device, &self.pool_sizes));
                self.used_pools.push(self.current_pool);
                self.current_pool = new_pool;
                unsafe { self.device.allocate_descriptor_sets(&alloc_ci) }
                    .expect("Failed to allocate descriptor set")
                    .pop()
                    .unwrap()
            }
            _ => panic!("Failed to allocate descriptor set"),
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
            unsafe {
                self.device
                    .reset_descriptor_pool(self.current_pool, vk::DescriptorPoolResetFlags::empty())
            }
            .expect("Failed to reset descriptor pool");
        }
    }

    pub fn reset_all_pools(&mut self) {
        self.reset_pools(true);
    }

    pub fn destroy(self) {
        self.free_pools
            .into_iter()
            .chain(self.used_pools)
            .for_each(|pool| unsafe {
                self.device.destroy_descriptor_pool(pool, None);
            });
        unsafe { self.device.destroy_descriptor_pool(self.current_pool, None) };
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
//goes around Rust orphan rule and allows implementing Eq and Hash
#[derive(Debug, Clone)]
struct DescriptorSetLayoutBindingWrapper {
    val: vk::DescriptorSetLayoutBinding,
}

impl PartialEq for DescriptorSetLayoutBindingWrapper {
    fn eq(&self, other: &Self) -> bool {
        debug_assert_eq!(
            self.val.p_immutable_samplers,
            ptr::null(),
            "immutable samplers not supported"
        );
        self.val.binding == other.val.binding
            && self.val.descriptor_count == other.val.descriptor_count
            && self.val.descriptor_type == other.val.descriptor_type
            && self.val.stage_flags == other.val.stage_flags
    }
}

impl Eq for DescriptorSetLayoutBindingWrapper {}

impl Hash for DescriptorSetLayoutBindingWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        debug_assert_eq!(
            self.val.p_immutable_samplers,
            ptr::null(),
            "immutable samplers not supported"
        );
        self.val.binding.hash(state);
        self.val.descriptor_count.hash(state);
        self.val.descriptor_type.hash(state);
        self.val.stage_flags.hash(state);
    }
}

pub struct DescriptorSetLayoutCache {
    cache: HashMap<Vec<DescriptorSetLayoutBindingWrapper>, vk::DescriptorSetLayout>,
    device: ash::Device,
}

impl DescriptorSetLayoutCache {
    pub fn new(device: ash::Device) -> DescriptorSetLayoutCache {
        DescriptorSetLayoutCache {
            cache: HashMap::new(),
            device,
        }
    }

    pub fn get(&mut self, desc: &[vk::DescriptorSetLayoutBinding]) -> vk::DescriptorSetLayout {
        let mut wrapping = desc
            .iter()
            .cloned()
            .map(|x| DescriptorSetLayoutBindingWrapper { val: x })
            .collect::<Vec<_>>();
        wrapping.sort_by_key(|d| d.val.binding);
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

    pub fn destroy(self) {
        self.cache.into_iter().for_each(|(_, layout)| unsafe {
            self.device.destroy_descriptor_set_layout(layout, None)
        });
    }
}

pub struct DescriptorSetCreator {
    cache: DescriptorSetLayoutCache,
    alloc: DescriptorAllocator,
}

impl DescriptorSetCreator {
    pub fn new(
        alloc: DescriptorAllocator,
        cache: DescriptorSetLayoutCache,
    ) -> DescriptorSetCreator {
        DescriptorSetCreator { cache, alloc }
    }

    pub fn new_set(&mut self) -> DescriptorSetBuilder {
        DescriptorSetBuilder {
            alloc: &mut self.alloc,
            cache: &mut self.cache,
            writes: Vec::new(),
            bindings: Vec::new(),
        }
    }

    pub fn destroy(self) {
        self.cache.destroy();
        self.alloc.destroy();
    }
}

pub struct DescriptorSetBuilder<'a> {
    cache: &'a mut DescriptorSetLayoutCache,
    alloc: &'a mut DescriptorAllocator,
    writes: Vec<vk::WriteDescriptorSet>,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
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
        let write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: vk::DescriptorSet::null(), //tmp value
            dst_binding: binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type,
            p_image_info: ptr::null(),
            p_buffer_info: &buf_info,
            p_texel_buffer_view: ptr::null(),
        };
        self.bindings.push(layout_binding);
        self.writes.push(write);
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
        let write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: vk::DescriptorSet::null(), //tmp value
            dst_binding: binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type,
            p_image_info: &image_info,
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
        };
        self.bindings.push(layout_binding);
        self.writes.push(write);
        self
    }

    pub fn build(mut self) -> Descriptor {
        let layout = self.cache.get(&self.bindings);
        let set = self.alloc.alloc(layout);
        self.writes.iter_mut().for_each(|w| w.dst_set = set);
        unsafe {
            self.alloc.device.update_descriptor_sets(&self.writes, &[]);
        }
        Descriptor { set, layout }
    }
}
