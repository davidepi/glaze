use super::device::Device;
use ash::vk;
use std::convert::TryInto;
use std::ptr;

/// Contains the necessary synchronization objects used during a draw operation for a single frame.
#[derive(Debug)]
pub struct PresentFrameSync {
    /// Fence used to stop the CPU until an image is ready to be acquired.
    pub acquire: vk::Fence,
    /// Semaphore used to signal when the image is ready to be acquired.
    pub image_available: vk::Semaphore,
    /// Semaphore used to signal when the image is ready to be presented.
    pub render_finished: vk::Semaphore,
}

impl PresentFrameSync {
    /// Creates a new PresentFrameSync object
    fn create<T: Device>(device: &T) -> Self {
        PresentFrameSync {
            acquire: create_fence(device, true),
            image_available: create_semaphore(device),
            render_finished: create_semaphore(device),
        }
    }

    /// Destroys the PresentFrameSync object
    fn destroy<T: Device>(self, device: &T) {
        let device = device.logical();
        unsafe {
            device.destroy_fence(self.acquire, None);
            device.destroy_semaphore(self.image_available, None);
            device.destroy_semaphore(self.render_finished, None);
        }
    }

    /// waits until the image is ready to be acquired
    pub fn wait_acquire<T: Device>(&mut self, device: &T) {
        let device = device.logical();
        let fence = &[self.acquire];
        unsafe {
            device
                .wait_for_fences(fence, true, u64::MAX)
                .expect("Failed to wait on fence");
            device.reset_fences(fence).expect("Failed to reset fence");
        }
    }
}

/// Constains all the synchronization objects for all frames.
pub struct PresentSync<const FRAMES_IN_FLIGHT: usize> {
    frames: [PresentFrameSync; FRAMES_IN_FLIGHT],
}

impl<const FRAMES_IN_FLIGHT: usize> PresentSync<FRAMES_IN_FLIGHT> {
    /// Creates a PresentSync object
    pub fn create<T: Device>(device: &T) -> Self {
        let frames = (0..FRAMES_IN_FLIGHT)
            .map(|_| PresentFrameSync::create(device))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        PresentSync { frames }
    }

    /// Returns the correct synchronization objects for the current frame.
    pub fn get(&mut self, frame_no: usize) -> &mut PresentFrameSync {
        let idx = frame_no % FRAMES_IN_FLIGHT;
        unsafe { self.frames.get_unchecked_mut(idx) }
    }

    /// Destroys all synchronization objects
    pub fn destroy<T: Device>(self, device: &T) {
        for frame in self.frames {
            frame.destroy(device)
        }
    }
}

/// Creates a fence
fn create_fence<T: Device>(device: &T, signaled: bool) -> vk::Fence {
    let ci = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        },
    };
    unsafe { device.logical().create_fence(&ci, None) }.expect("Failed to create fence")
}

/// Creates a semaphore
fn create_semaphore<T: Device>(device: &T) -> vk::Semaphore {
    let ci = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };
    unsafe { device.logical().create_semaphore(&ci, None) }.expect("Failed to create semaphore")
}
