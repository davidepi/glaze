use super::Device;
use ash::vk;
use std::convert::TryInto;
use std::ptr;

const FRAMES_INFLIGHT: usize = 2;

#[derive(Debug)]
pub struct PresentFrameSync {
    acquire: vk::Fence,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
}

impl PresentFrameSync {
    fn create<T: Device>(device: &T) -> Self {
        PresentFrameSync {
            acquire: create_fence(device, true),
            image_available: create_semaphore(device),
            render_finished: create_semaphore(device),
        }
    }

    fn destroy<T: Device>(self, device: &T) {
        let device = device.logical();
        unsafe {
            device.destroy_fence(self.acquire, None);
            device.destroy_semaphore(self.image_available, None);
            device.destroy_semaphore(self.render_finished, None);
        }
    }

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

    pub fn acquire_fence(&self) -> vk::Fence {
        self.acquire
    }

    pub fn image_available(&self) -> vk::Semaphore {
        self.image_available
    }

    pub fn render_finished(&self) -> vk::Semaphore {
        self.render_finished
    }
}

pub struct PresentSync {
    frames: [PresentFrameSync; FRAMES_INFLIGHT],
    current_frame: u8,
}

impl PresentSync {
    pub fn create<T: Device>(device: &T) -> Self {
        let frames = (0..FRAMES_INFLIGHT)
            .map(|_| PresentFrameSync::create(device))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        PresentSync {
            frames,
            current_frame: 0,
        }
    }

    pub fn next(&mut self) -> &mut PresentFrameSync {
        let frame = unsafe { self.frames.get_unchecked_mut(self.current_frame as usize) };
        self.current_frame = ((self.current_frame as usize + 1) % FRAMES_INFLIGHT) as u8;
        frame
    }

    pub fn destroy<T: Device>(self, device: &T) {
        for frame in self.frames {
            frame.destroy(device)
        }
    }
}

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

fn create_semaphore<T: Device>(device: &T) -> vk::Semaphore {
    let ci = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };
    unsafe { device.logical().create_semaphore(&ci, None) }.expect("Failed to create semaphore")
}
