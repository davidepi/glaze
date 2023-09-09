use super::MetalDevice;
use crate::geometry::Extent2D;
use crate::graphics::format::{ImageFormat, PresentMode};
use crate::graphics::swapchain::{PresentDevice, Swapchain};
use cocoa::appkit::NSView;
use winit::platform::macos::WindowExtMacOS;
use winit::window::Window;

pub struct MetalSwapchain {
    inner: metal::MetalLayer,
    size: Extent2D<u32>,
}

impl PresentDevice for MetalDevice {
    type Swapchain = MetalSwapchain;

    fn new_swapchain(
        &mut self,
        mode: PresentMode,
        format: ImageFormat,
        size: Extent2D<u32>,
        window: &Window,
        triple_buffering: bool,
        _wayland: bool,
    ) -> Result<Self::Swapchain, Self::GraphicError> {
        let layer = metal::MetalLayer::new();
        match mode {
            PresentMode::Fifo | PresentMode::Mailbox => layer.set_display_sync_enabled(true),
            PresentMode::Immediate => layer.set_display_sync_enabled(false),
        }
        layer.set_pixel_format(format.to_metal());
        layer.set_drawable_size(size.to_cgsize());
        if triple_buffering {
            layer.set_maximum_drawable_count(3);
        } else {
            layer.set_maximum_drawable_count(2);
        }
        // link surface and window
        unsafe {
            let view = window.ns_window() as cocoa::base::id;
            view.setWantsLayer(metal::objc::runtime::YES);
            #[allow(clippy::useless_transmute)] // clippy suggestion doesn't compile...
            view.setLayer(std::mem::transmute(layer.as_ref()));
        }
        let ret = MetalSwapchain { inner: layer, size };
        Ok(ret)
    }
}

impl Swapchain for MetalSwapchain {
    fn size(&self) -> Extent2D<u32> {
        self.size
    }

    fn set_size(&mut self, size: Extent2D<u32>) {
        self.inner.set_drawable_size(size.to_cgsize());
    }
}
