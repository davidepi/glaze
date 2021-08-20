use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::vulkan::{VkInstance, VkPresentedInstance};

pub const DEFAULT_WIDTH: u32 = 800;
pub const DEFAULT_HEIGHT: u32 = 600;

pub struct GlazeApp {
    window: Window,
    instance: VkPresentedInstance,
}

impl GlazeApp {
    pub fn new(event_loop: &EventLoop<()>) -> GlazeApp {
        let window = WindowBuilder::new()
            .with_title(env!("CARGO_PKG_NAME"))
            .with_inner_size(winit::dpi::LogicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT))
            .build(event_loop)
            .unwrap();
        let instance = VkInstance::new(&[]);
        let instance = VkPresentedInstance::new(instance, &window);
        GlazeApp { window, instance }
    }

    pub fn main_loop(self, event_loop: EventLoop<()>) -> ! {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event,
                window_id: _,
            } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            Event::MainEventsCleared => self.window.request_redraw(),
            _ => (),
        })
    }
}
