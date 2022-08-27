use glaze::{parse, PresentInstance, RayTraceRenderer, RealtimeRenderer, RealtimeScene};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::convert::TryFrom;
use std::error::Error;
use std::sync::Arc;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::ui::{draw_ui, UiState};

pub struct InteractiveView {
    window: Window,
    renderer: RealtimeRenderer,
    raytracer: Option<RayTraceRenderer<PresentInstance>>,
    platform: WinitPlatform,
    imgui: imgui::Context,
    mouse_pos: (f32, f32),
    rmb_down: bool,
    mmb_down: bool,
    alt_speed_down: bool,
    state: UiState,
}

impl InteractiveView {
    pub fn new(
        event_loop: &EventLoop<()>,
        scene_path: Option<String>,
    ) -> Result<InteractiveView, Box<dyn Error>> {
        if let Some(monitor) = event_loop.primary_monitor() {
            let monitor_size = monitor.size();
            let default_size = PhysicalSize::new(monitor_size.width / 2, monitor_size.height / 2);
            let window = WindowBuilder::new()
                .with_title(env!("CARGO_PKG_NAME"))
                .with_inner_size(default_size)
                .build(event_loop)
                .unwrap();
            let mut imgui = imgui::Context::create();
            let mut platform = WinitPlatform::init(&mut imgui);
            platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);
            let instance =
                Arc::new(PresentInstance::new(&window).expect("No GPU or window system found"));
            let scene = if let Some(scene_path) = scene_path {
                match parse(&scene_path) {
                    Ok(parsed) => Some(RealtimeScene::new(Arc::clone(&instance), parsed)),
                    Err(e) => {
                        log::error!("Failed to parse scene {}: {}", scene_path, e.to_string());
                        None
                    }
                }
            } else {
                None
            };
            let renderer = RealtimeRenderer::new(
                instance.clone(),
                &mut imgui,
                default_size.width,
                default_size.height,
                1.0,
                scene,
            );
            let raytracer = RayTraceRenderer::<PresentInstance>::try_from(&renderer).ok();
            let state = UiState::new(instance);
            Ok(InteractiveView {
                window,
                renderer,
                raytracer,
                platform,
                imgui,
                mouse_pos: (0.0, 0.0),
                rmb_down: false,
                mmb_down: false,
                alt_speed_down: false,
                state,
            })
        } else {
            Err("No monitor found".into())
        }
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>) -> ! {
        event_loop.run(move |event, _, control_flow| {
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
                        if let Some(raytracer) = &mut self.raytracer {
                            raytracer.change_resolution(
                                (size.width as f32 * scale) as u32,
                                (size.height as f32 * scale) as u32,
                            );
                        }
                    }
                    WindowEvent::KeyboardInput { input, .. } => handle_keyboard(input, &mut self),
                    WindowEvent::CursorMoved { position, .. } => mouse_moved(position, &mut self),
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Right {
                            if state == ElementState::Pressed {
                                self.rmb_down = true;
                            } else {
                                self.rmb_down = false;
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
                    let ui = self.imgui.frame();
                    draw_ui(
                        &ui,
                        &mut self.state,
                        &mut self.window,
                        &mut self.renderer,
                        &mut self.raytracer,
                    );
                    self.platform.prepare_render(&ui, &self.window);
                    let draw_data = ui.render();
                    if self.state.use_raytracer {
                        self.renderer
                            .draw_frame(Some(draw_data), &mut self.raytracer);
                    } else {
                        self.renderer.draw_frame(Some(draw_data), &mut None);
                    }
                }
                Event::LoopDestroyed => self.renderer.wait_idle(),
                _ => (),
            }
        });
    }
}

fn handle_keyboard(input: KeyboardInput, view: &mut InteractiveView) {
    match input.virtual_keycode {
        Some(VirtualKeyCode::W) => camera_pos_advance(view, 1.0),
        Some(VirtualKeyCode::S) => camera_pos_advance(view, -1.0),
        Some(VirtualKeyCode::A) => camera_pos_strafe(view, -1.0),
        Some(VirtualKeyCode::D) => camera_pos_strafe(view, 1.0),
        Some(VirtualKeyCode::LShift) => match input.state {
            ElementState::Pressed => view.alt_speed_down = true,
            ElementState::Released => view.alt_speed_down = false,
        },
        _ => {}
    }
}

fn camera_pos_strafe(view: &mut InteractiveView, direction: f32) {
    if !view.state.movement_lock && !view.imgui.io().want_capture_keyboard {
        let magnitude = view.state.mov_speed;
        let multiplier = if view.alt_speed_down {
            view.state.mov_speed_mul
        } else {
            1.0
        };
        let mut camera = view.renderer.camera();
        camera.strafe(direction * magnitude * multiplier);
        view.renderer.set_camera(camera);
        if let Some(raytracer) = &mut view.raytracer {
            raytracer.update_camera(camera);
        }
    }
}

fn camera_pos_advance(view: &mut InteractiveView, direction: f32) {
    if !view.state.movement_lock && !view.imgui.io().want_capture_keyboard {
        let magnitude = view.state.mov_speed;
        let multiplier = if view.alt_speed_down {
            view.state.mov_speed_mul
        } else {
            1.0
        };
        let mut camera = view.renderer.camera();
        camera.advance(direction * magnitude * multiplier);
        view.renderer.set_camera(camera);
        if let Some(raytracer) = &mut view.raytracer {
            raytracer.update_camera(camera);
        }
    }
}

fn mouse_moved(new_pos: PhysicalPosition<f64>, view: &mut InteractiveView) {
    let (x, y) = (new_pos.x as f32, new_pos.y as f32);
    let (old_x, old_y) = view.mouse_pos;
    let delta = (x - old_x, y - old_y);
    view.mouse_pos = (x, y);
    // if lmb pressed, move camera
    if view.rmb_down && !view.state.movement_lock && !view.imgui.io().want_capture_mouse {
        let magnitude = view.state.mouse_sensitivity;
        let x_dir = if view.state.inverted_mouse_h {
            1.0
        } else {
            -1.0
        };
        let y_dir = if view.state.inverted_mouse_v {
            1.0
        } else {
            -1.0
        };
        let mut camera = view.renderer.camera();
        camera.look_around(
            f32::to_radians(magnitude * x_dir * delta.0),
            f32::to_radians(magnitude * y_dir * delta.1),
        );
        view.renderer.set_camera(camera);
        if let Some(raytracer) = &mut view.raytracer {
            raytracer.update_camera(camera);
        }
    }
    if view.mmb_down && !view.state.movement_lock && !view.imgui.io().want_capture_mouse {
        let magnitude = view.state.vert_speed;
        let direction = if view.state.inverted_vert_mov {
            1.0
        } else {
            -1.0
        };
        let mut camera = view.renderer.camera();
        camera.elevate(direction * magnitude * delta.1);
        view.renderer.set_camera(camera);
        if let Some(raytracer) = &mut view.raytracer {
            raytracer.update_camera(camera);
        }
    }
}
