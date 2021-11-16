use glaze::RealtimeRenderer;
use imgui::Condition;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;

fn main() {
    env_logger::init();
    let mut event_loop = EventLoop::new();
    let render_width = DEFAULT_WIDTH;
    let render_height = DEFAULT_HEIGHT;
    let mut imgui = imgui::Context::create();
    let window = WindowBuilder::new()
        .with_title(env!("CARGO_PKG_NAME"))
        .with_inner_size(winit::dpi::LogicalSize::new(render_width, render_height))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);
    let mut renderer = RealtimeRenderer::create(&window, &mut imgui, render_width, render_height);
    event_loop.run_return(|event, _, control_flow| {
        platform.handle_event(imgui.io_mut(), &window, &event);
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
                platform
                    .prepare_frame(imgui.io_mut(), &window)
                    .expect("Failed to prepare frame");
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                platform
                    .prepare_frame(imgui.io_mut(), &window)
                    .expect("Failed to prepare frame");
                let ui = imgui.frame();
                imgui::Window::new("Hello world")
                    .size([300.0, 100.0], Condition::FirstUseEver)
                    .build(&ui, || {
                        ui.text("Hello world!");
                        ui.separator();
                        let mouse_pos = ui.io().mouse_pos;
                        ui.text(format!(
                            "Mouse Position: ({:.1},{:.1})",
                            mouse_pos[0], mouse_pos[1]
                        ));
                    });
                platform.prepare_render(&ui, &window);
                let draw_data = ui.render();
                renderer.draw_frame(draw_data);
            }
            Event::LoopDestroyed => renderer.wait_idle(),
            _ => (),
        }
    });
    renderer.destroy();
}
