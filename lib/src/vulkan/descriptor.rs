use super::memory::AllocatedBuffer;
use super::AllocatedImage;
use ash::vk;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::{Arc, Mutex};

/// Initial number of allocated descriptor pools.
const MAX_POOLS: usize = 4;
/// Number of descriptor sets per pool.
const MAX_SETS: usize = 512;

/// Wraps a descriptor set and a descriptor set layout in a single struct.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub set: vk::DescriptorSet,
    pub layout: vk::DescriptorSetLayout,
}

impl PartialOrd for Descriptor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.set.partial_cmp(&other.set)
    }
}

impl Ord for Descriptor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.set.cmp(&other.set)
    }
}

/// Allocator manager for descriptor pools and sets.
struct DescriptorAllocator {
    /// Current pool in use.
    current_pool: Option<vk::DescriptorPool>,
    /// Typical amount of each descriptor type contained in this pool.
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    /// Unused descriptor pools.
    free_pools: Vec<vk::DescriptorPool>,
    /// Filled descriptor pools.
    used_pools: Vec<vk::DescriptorPool>,
    device: Arc<ash::Device>,
}

impl DescriptorAllocator {
    /// Creates a new descriptor allocator with the given usage.
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

    /// Allocates a new descriptor set with the give layout from the current pool.
    /// Changes pool if necessary.
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

/// Creates a raw descriptor pool with the given size
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

/// A Wrapper for a descriptor set binding
///
/// Goes around Rust orphan rule and allows implementing Eq and Hash
/// Removes also the immutable sampler from DescriptorSetLayoutBinding to have trait "Send".
/// Currently immutable samplers are not used in this lib.
#[derive(Debug, Clone)]
struct DescriptorSetLayoutBindingWrapper {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}

impl DescriptorSetLayoutBindingWrapper {
    /// Creates a new wrapper from a raw descriptor set layout binding
    pub fn new(dsbin: vk::DescriptorSetLayoutBinding) -> DescriptorSetLayoutBindingWrapper {
        debug_assert!(
            dsbin.p_immutable_samplers.is_null(),
            "immutable samplers not supported in this lib"
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

/// Wrapper containing a DescriptorBufferInfo or a DescriptorImageInfo
///
/// DescriptorSetLayout uses pointers and, in this lib, is created in a different function than the
/// build one. Therefore, in the optimized build, the target of this pointers is deallocated so I'm
/// forced to store it in this wrapper and build the DescriptorSetLayoutWrite function in the
/// DescriptorSetBuilder::build()
#[derive(Debug, Clone)]
enum BufOrImgInfo {
    Buf(vk::DescriptorBufferInfo),
    Img(Vec<vk::DescriptorImageInfo>),
    Acc(vk::WriteDescriptorSetAccelerationStructureKHR),
}

/// Cache for descriptor set layouts
pub struct DescriptorSetLayoutCache {
    cache: HashMap<Vec<DescriptorSetLayoutBindingWrapper>, vk::DescriptorSetLayout>,
    device: Arc<ash::Device>,
}

impl DescriptorSetLayoutCache {
    /// Creates an empty descriptor set layout cache
    pub fn new(device: Arc<ash::Device>) -> DescriptorSetLayoutCache {
        DescriptorSetLayoutCache {
            cache: HashMap::new(),
            device,
        }
    }

    /// Retrieves a descriptor set layout from the cache if it exists, otherwise it creates and adds
    /// it to the cache. In any case the descriptor set layout is returned.
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

/// Application-wide cache for sharing descriptor set layouts.
pub type DLayoutCache = Arc<Mutex<DescriptorSetLayoutCache>>;

/// Manages the allocation and reuse of descriptor sets and layouts.
pub struct DescriptorSetManager {
    cache: DLayoutCache,
    alloc: DescriptorAllocator,
}

impl DescriptorSetManager {
    /// Creates a new empty descriptor set manager, sharing an existing cache.
    /// The cache can be obtained from an existing manager with the [DescriptorSetManager::cache]
    /// function.
    pub fn new(
        device: Arc<ash::Device>,
        avg_desc: &[(vk::DescriptorType, f32)],
        cache: DLayoutCache,
    ) -> DescriptorSetManager {
        let alloc = DescriptorAllocator::new(device, avg_desc);
        DescriptorSetManager { cache, alloc }
    }

    /// Returns the descriptor set layout cache used by this descriptor set manager
    #[cfg(feature = "vulkan-interactive")]
    pub fn cache(&self) -> Arc<Mutex<DescriptorSetLayoutCache>> {
        self.cache.clone()
    }

    /// Creates a new descriptor set
    pub fn new_set(&mut self) -> DescriptorSetBuilder {
        DescriptorSetBuilder {
            alloc: &mut self.alloc,
            cache: self.cache.clone(),
            bindings: Vec::new(),
            info: Vec::new(),
        }
    }
}

/// Builder for a descriptor set.
///
/// Creates a descriptor set with a builder pattern.
pub struct DescriptorSetBuilder<'a> {
    cache: DLayoutCache,
    alloc: &'a mut DescriptorAllocator,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    info: Vec<BufOrImgInfo>,
}

impl<'a> DescriptorSetBuilder<'a> {
    /// Binds a buffer to the current set.
    /// The buffer is bound in its entirety.
    /// Note that bindings are ordered.
    #[must_use]
    pub fn bind_buffer(
        self,
        buffer: &AllocatedBuffer,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let buf_info = vk::DescriptorBufferInfo {
            buffer: buffer.buffer,
            offset: 0,
            range: buffer.size,
        };
        self.bind_buffer_with_info(buf_info, descriptor_type, stage_flags)
    }

    /// Binds a buffer to the current set, but requires the buffer extent to be specified.
    /// Note that bindings are ordered.
    pub fn bind_buffer_with_info(
        mut self,
        info: vk::DescriptorBufferInfo,
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
        self.info.push(BufOrImgInfo::Buf(info));
        self
    }

    /// Binds an image to the current set.
    /// Note that bindings are ordered.
    #[must_use]
    pub fn bind_image(
        mut self,
        image: &AllocatedImage,
        layout: vk::ImageLayout,
        sampler: vk::Sampler,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let image_info = vec![vk::DescriptorImageInfo {
            sampler,
            image_view: image.image_view,
            image_layout: layout,
        }];
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

    /// Binds several images to the current set.
    /// All images are expected to have the same layout.
    /// Note that bindings are ordered.
    #[must_use]
    pub fn bind_image_array(
        mut self,
        images: &[vk::ImageView],
        layout: vk::ImageLayout,
        sampler: vk::Sampler,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let image_info = images
            .iter()
            .map(|image_view| vk::DescriptorImageInfo {
                sampler,
                image_view: *image_view,
                image_layout: layout,
            })
            .collect::<Vec<_>>();
        let binding = self.bindings.len() as u32;
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: image_info.len() as u32,
            stage_flags,
            p_immutable_samplers: ptr::null(),
        };
        self.bindings.push(layout_binding);
        self.info.push(BufOrImgInfo::Img(image_info));
        self
    }

    /// Binds an acceleration structure to the current set.
    /// Note that bindings are ordered.
    #[must_use]
    pub fn bind_acceleration_structure(
        mut self,
        acc: &vk::AccelerationStructureKHR,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let binding = self.bindings.len() as u32;
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
            stage_flags,
            p_immutable_samplers: ptr::null(),
        };
        let layout_extension = vk::WriteDescriptorSetAccelerationStructureKHR {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            p_next: ptr::null(),
            acceleration_structure_count: 1,
            p_acceleration_structures: acc as *const _,
        };
        self.bindings.push(layout_binding);
        self.info.push(BufOrImgInfo::Acc(layout_extension));
        self
    }

    /// Builds the descriptor set.
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
                    write.descriptor_count = img.len() as u32;
                    write.p_image_info = img.as_ptr();
                }
                BufOrImgInfo::Acc(acc) => {
                    write.p_next = acc as *const _ as *const c_void;
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
