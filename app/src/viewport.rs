use cgmath::Vector3 as Vec3;
use glaze::RealtimeRenderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::error::Error;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

use crate::ui::{draw_ui, UiState};

pub struct InteractiveView {
    window: Window,
    renderer: RealtimeRenderer,
    platform: WinitPlatform,
    imgui: imgui::Context,
    mouse_pos: (f32, f32),
    lmb_down: bool,
    mmb_down: bool,
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
                .build(&event_loop)
                .unwrap();
            let mut imgui = imgui::Context::create();
            let mut platform = WinitPlatform::init(&mut imgui);
            platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);
            let renderer = RealtimeRenderer::create(
                &window,
                &mut imgui,
                default_size.width,
                default_size.height,
                1.0,
            );
            let state = UiState::new();
            Ok(InteractiveView {
                window,
                renderer,
                platform,
                imgui,
                mouse_pos: (0.0, 0.0),
                lmb_down: false,
                mmb_down: false,
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
                    WindowEvent::Resized(size) => {
                        let scale = self.renderer.render_scale();
                        self.renderer
                            .update_render_size(size.width, size.height, scale);
                    }
                    WindowEvent::KeyboardInput { input, .. } => handle_keyboard(input, self),
                    WindowEvent::CursorMoved { position, .. } => mouse_moved(position, self),
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left {
                            if state == ElementState::Pressed {
                                self.lmb_down = true;
                            } else {
                                self.lmb_down = false;
                            }
                        } else if button == MouseButton::Middle {
                            if state == ElementState::Pressed {
                                self.mmb_down = true;
                            } else {
                                self.mmb_down = false;
                            }
                        }
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
                    draw_ui(
                        &mut ui,
                        &mut self.state,
                        &mut self.window,
                        &mut self.renderer,
                    );
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

fn handle_keyboard(input: KeyboardInput, view: &mut InteractiveView) {
    match input.virtual_keycode {
        Some(VirtualKeyCode::W) => camera_pos_advance(view, 0.1),
        Some(VirtualKeyCode::S) => camera_pos_advance(view, -0.1),
        Some(VirtualKeyCode::A) => camera_pos_strafe(view, -0.1),
        Some(VirtualKeyCode::D) => camera_pos_strafe(view, 0.1),
        _ => {}
    }
}

fn camera_pos_strafe(view: &mut InteractiveView, magnitude: f32) {
    if let Some((pos, target)) = view.renderer.camera_position() {
        *pos += magnitude * Vec3::unit_x();
        *target += magnitude * Vec3::unit_x();
    }
}

fn camera_pos_advance(view: &mut InteractiveView, magnitude: f32) {
    if let Some((pos, target)) = view.renderer.camera_position() {
        let delta = magnitude * Vec3::new(target.x, target.y, 1.0);
        *pos += delta;
        *target += delta;
    }
}

fn mouse_moved(new_pos: PhysicalPosition<f64>, view: &mut InteractiveView) {
    let (x, y) = (new_pos.x as f32, new_pos.y as f32);
    let (old_x, old_y) = view.mouse_pos;
    let delta = (x - old_x, y - old_y);
    view.mouse_pos = (x, y);
    // if lmb pressed, move camera
    if view.lmb_down {
        let pos = view.renderer.camera_position();
        if let Some((_, target)) = pos {
            // TODO: move this '-' into some sort of control settings
            *target += Vec3::<f32>::new(delta.0, -delta.1, 0.0);
        }
    }
    if view.mmb_down {
        let pos = view.renderer.camera_position();
        if let Some((pos, _)) = pos {
            *pos += Vec3::<f32>::new(0.0, delta.1, 0.0);
        }
    }
}
