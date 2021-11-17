use std::error::Error;

use glaze::RealtimeRenderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

use crate::ui::{draw_ui, UiState};

pub struct InteractiveView {
    window: Window,
    renderer: RealtimeRenderer,
    platform: WinitPlatform,
    imgui: imgui::Context,
    state: UiState,
}

impl InteractiveView {
    pub fn new(event_loop: &mut EventLoop<()>) -> Result<InteractiveView, Box<dyn Error>> {
        if let Some(monitor) = event_loop.primary_monitor() {
            let monitor_size = monitor.size();
            let default_size = PhysicalSize::new(monitor_size.width / 2, monitor_size.height / 2);
            let window = WindowBuilder::new()
                .with_title(env!("CARGO_PKG_NAME"))
                .with_inner_size(default_size)
                .with_resizable(false)
                .build(&event_loop)
                .unwrap();
            let mut imgui = imgui::Context::create();
            let mut platform = WinitPlatform::init(&mut imgui);
            platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);
            let renderer = RealtimeRenderer::create(
                &window,
                &mut imgui,
                default_size.width,
                default_size.height,
            );
            let state = UiState::new();
            Ok(InteractiveView {
                window,
                renderer,
                platform,
                imgui,
                state,
            })
        } else {
            return Err("No monitor found".into());
        }
    }

    pub fn main_loop(&mut self, event_loop: &mut EventLoop<()>) {
        event_loop.run_return(|event, _, control_flow| {
            self.platform
                .handle_event(self.imgui.io_mut(), &self.window, &event);
            match event {
                Event::WindowEvent {
                    event,
                    window_id: _,
                } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    self.platform
                        .prepare_frame(self.imgui.io_mut(), &self.window)
                        .expect("Failed to prepare frame");
                    self.window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    self.platform
                        .prepare_frame(self.imgui.io_mut(), &self.window)
                        .expect("Failed to prepare frame");
                    let mut ui = self.imgui.frame();
                    draw_ui(&mut ui, &mut self.state, &mut self.renderer);
                    self.platform.prepare_render(&ui, &self.window);
                    let draw_data = ui.render();
                    self.renderer.draw_frame(Some(draw_data));
                }
                Event::LoopDestroyed => self.renderer.wait_idle(),
                _ => (),
            }
        });
    }

    pub fn destroy(self) {
        self.renderer.wait_idle();
        self.renderer.destroy()
    }
}
