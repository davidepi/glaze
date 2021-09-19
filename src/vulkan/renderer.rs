use winit::window::Window;

use super::Device;
use super::Instance;
use super::PresentInstance;
use super::PresentSync;
use super::Swapchain;

pub struct RealtimeRenderer {
    pub instance: PresentInstance,
    pub swapchain: Swapchain,
    pub sync: PresentSync,
    render_width: u32,
    render_height: u32,
}

impl RealtimeRenderer {
    pub fn create(window: &Window, width: u32, height: u32) -> Self {
        let mut instance = PresentInstance::new(&window);
        let swapchain = Swapchain::create(&mut instance, width, height);
        let sync = PresentSync::create(instance.device());
        RealtimeRenderer {
            instance,
            swapchain,
            sync,
            render_width: width,
            render_height: height,
        }
    }

    pub fn wait_idle(&self) {
        unsafe { self.instance.device().logical().device_wait_idle() }.expect("Failed to wait idle")
    }

    pub fn change_render_size(&mut self, width: u32, height: u32) {
        self.wait_idle();
        self.render_width = width;
        self.render_height = height;
        self.swapchain.re_create(&mut self.instance, width, height);
    }

    pub fn destroy(self) {
        self.sync.destroy(self.instance.device());
        self.swapchain.destroy(&self.instance);
    }
}
