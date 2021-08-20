use super::{platform, VkInstance};
use ash::vk;
use winit::window::Window;

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub loader: ash::extensions::khr::Surface,
}

impl Surface {
    pub fn new(instance: &VkInstance, window: &Window) -> Self {
        let surface =
            unsafe { platform::create_surface(&instance.entry, &instance.instance, window) }
                .expect("Failed to create surface");
        let surface_loader =
            ash::extensions::khr::Surface::new(&instance.entry, &instance.instance);
        Surface {
            surface,
            loader: surface_loader,
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}
