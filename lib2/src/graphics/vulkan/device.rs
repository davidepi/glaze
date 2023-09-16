use super::error::VulkanError;
use super::extensions::VulkanFeatures;
use super::instance::InstanceVulkan;
use crate::graphics::format::FeatureSet;
use crate::graphics::vulkan::physical::PhysicalDeviceVulkan;
use ash::vk;
use std::ffi::c_void;
use std::ptr;

pub struct DeviceVulkan {
    logical: ash::Device,
    instance: InstanceVulkan,
}

impl DeviceVulkan {
    pub fn new(device_id: Option<u64>, features: FeatureSet) -> Result<Self, VulkanError> {
        let instance = InstanceVulkan::new(features)?;
        let pdevice = if let Some(id) = device_id {
            match PhysicalDeviceVulkan::with_id(&instance, id) {
                Ok(d) => Ok(d),
                Err(e) => {
                    log::error!("{e}. Using default device.");
                    PhysicalDeviceVulkan::with_default(&instance)
                }
            }
        } else {
            PhysicalDeviceVulkan::with_default(&instance)
        }?;
        let queues = application_required_queue_families(&instance, &pdevice)?;
        let mut queues_ci = [vk::DeviceQueueCreateInfo::default(); 3];
        let priorities = [1.0; 1];
        for i in 0..3 {
            queues_ci[i].queue_family_index = queues[i].0;
            queues_ci[i].queue_count = 1;
            queues_ci[i].p_queue_priorities = priorities.as_ptr();
        }
        let exts = &instance.features().required_device_extensions();
        let exts_pp = exts.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
        let features = VulkanFeatures::new(instance.features());
        let device_ci = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: features.as_ffi() as *const c_void,
            flags: vk::DeviceCreateFlags::default(),
            queue_create_info_count: queues_ci.len() as u32,
            p_queue_create_infos: queues_ci.as_ptr(),
            enabled_extension_count: exts_pp.len() as u32,
            pp_enabled_extension_names: exts_pp.as_ptr(),
            p_enabled_features: ptr::null(),
            ..Default::default()
        };
        let logical = unsafe {
            instance
                .vk_instance()
                .create_device(pdevice.device, &device_ci, None)
        }?;
        Ok(Self { logical, instance })
    }
}

// finds the queue family with the specified flags and returns the family with the most queues
// available
fn find_queue<F>(props: &[vk::QueueFamilyProperties], find_queue: F) -> Option<usize>
where
    F: Fn(vk::QueueFlags) -> bool,
{
    props
        .iter()
        .enumerate()
        .filter(|(_, &prop)| find_queue(prop.queue_flags))
        .max_by_key(|(_, prop)| prop.queue_count)
        .map(|prop| prop.0)
}

/// Gather one queue for each type. Returns family and capabilities.
/// One queue for each type in the order: GRAPHICS, TRANSFER, COMPUTE.
/// I take surface support for granted...
fn application_required_queue_families(
    instance: &InstanceVulkan,
    pdevice: &PhysicalDeviceVulkan,
) -> Result<[(u32, vk::QueueFlags); 3], VulkanError> {
    let mut props = unsafe {
        instance
            .vk_instance()
            .get_physical_device_queue_family_properties(pdevice.device)
    };
    let mut retval = [(0, vk::QueueFlags::empty()); 3];
    // first find a queue with graphics, this will be the generic one for the main rendering job.
    let graphic_index = find_queue(&props, |q| q.contains(vk::QueueFlags::GRAPHICS))
        .ok_or_else(|| VulkanError::new("Not enough Graphics queues available"))?;
    if props[graphic_index].queue_count >= 1 {
        props[graphic_index].queue_count -= 1;
        retval[0] = (graphic_index as u32, props[graphic_index].queue_flags);
    } else {
        return Err(VulkanError::new("Not enough Graphics queues available"));
    }
    // then find the unique transfer queue. If not exist, use any queue with transfer support.
    let mut transfer_index = find_queue(&props, |q| {
        q.contains(vk::QueueFlags::TRANSFER)
            && !q.contains(vk::QueueFlags::GRAPHICS)
            && !q.contains(vk::QueueFlags::COMPUTE)
    });
    if transfer_index.is_none() {
        transfer_index = find_queue(&props, |q| q.contains(vk::QueueFlags::TRANSFER));
    }
    let transfer_index =
        transfer_index.ok_or_else(|| VulkanError::new("Not enough Transfer queues available"))?;
    if props[transfer_index].queue_count >= 1 {
        props[transfer_index].queue_count -= 1;
        retval[1] = (transfer_index as u32, props[transfer_index].queue_flags);
    } else {
        return Err(VulkanError::new("Not enough Transfer queues available"));
    }
    // Finally, try to find a Compute + Transfer (Compute Async), or use any compute otherwise.
    let mut compute_index = find_queue(&props, |q| {
        q.contains(vk::QueueFlags::TRANSFER)
            && !q.contains(vk::QueueFlags::GRAPHICS)
            && q.contains(vk::QueueFlags::COMPUTE)
    });
    if compute_index.is_none() {
        compute_index = find_queue(&props, |q| q.contains(vk::QueueFlags::COMPUTE));
    }
    let compute_index =
        compute_index.ok_or_else(|| VulkanError::new("Not enough Compute queues available"))?;
    if props[compute_index].queue_count >= 1 {
        props[compute_index].queue_count -= 1;
        retval[2] = (compute_index as u32, props[compute_index].queue_flags);
    } else {
        return Err(VulkanError::new("Not enough Compute queues available"));
    }
    Ok(retval)
}

impl Drop for DeviceVulkan {
    fn drop(&mut self) {
        unsafe {
            self.logical.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceVulkan;
    use crate::graphics::format::FeatureSet;
    use crate::graphics::vulkan::error::VulkanError;

    #[test]
    fn create_convert() -> Result<(), VulkanError> {
        DeviceVulkan::new(None, FeatureSet::Convert)?;
        Ok(())
    }

    #[test]
    fn create_present() -> Result<(), VulkanError> {
        DeviceVulkan::new(None, FeatureSet::Present)?;
        Ok(())
    }
}
