use crate::vulkan::RealtimeRenderer;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

pub const DEFAULT_WIDTH: u32 = 800;
pub const DEFAULT_HEIGHT: u32 = 600;

pub struct GlazeApp {
    window: Window,
    renderer: RealtimeRenderer,
}

impl GlazeApp {
    pub fn create(event_loop: &EventLoop<()>) -> GlazeApp {
        let render_width = DEFAULT_WIDTH;
        let render_height = DEFAULT_HEIGHT;
        let window = WindowBuilder::new()
            .with_title(env!("CARGO_PKG_NAME"))
            .with_inner_size(winit::dpi::LogicalSize::new(render_width, render_height))
            .with_resizable(false)
            .build(event_loop)
            .unwrap();
        let renderer = RealtimeRenderer::create(&window, render_width, render_height);
        GlazeApp { window, renderer }
    }

    pub fn main_loop(&mut self, mut event_loop: EventLoop<()>) {
        event_loop.run_return(|event, _, control_flow| match event {
            Event::WindowEvent {
                event,
                window_id: _,
            } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::MainEventsCleared => self.renderer.draw_frame(),
            Event::LoopDestroyed => self.renderer.wait_idle(),
            _ => (),
        });
    }

    pub fn destroy(self) {
        self.renderer.destroy();
    }
}
