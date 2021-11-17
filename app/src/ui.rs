use glaze::{parse, RealtimeRenderer};
use imgui::{CollapsingHeader, Condition, MenuItem, Slider, Ui};
use nfd2::Response;
use winit::window::Window;

pub struct UiState {
    render_window: bool,
    render_scale_cur: f32,
    render_scale_sel: f32,
    stats_window: bool,
}

impl UiState {
    pub fn new() -> Self {
        UiState {
            render_window: false,
            render_scale_cur: 1.0,
            render_scale_sel: 1.0,
            stats_window: true,
        }
    }
}

pub fn draw_ui(
    ui: &mut Ui,
    state: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if MenuItem::new("Open").shortcut("Ctrl+O").build(ui) {
                open_scene(renderer);
            }
        });
        ui.menu("Window", || {
            ui.checkbox("Render", &mut state.render_window);
            ui.checkbox("Stats", &mut state.stats_window);
        });
    });
    if state.render_window {
        window_render(ui, state, window, renderer);
    }
    if state.stats_window {
        window_stats(ui, state, window, renderer);
    }
}

fn window_render(
    ui: &mut Ui,
    state: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    let mut closed = state.render_window;
    imgui::Window::new("Render")
        .size([400.0, 400.0], Condition::Appearing)
        .opened(&mut closed)
        .save_settings(false)
        .build(ui, || {
            if CollapsingHeader::new("Viewport Options").build(ui) {
                ui.text("Current render scale:");
                ui.text(format!(
                    "Render scale: {}x ({}x{})",
                    state.render_scale_cur,
                    (window.inner_size().width as f32 * state.render_scale_cur) as u32,
                    (window.inner_size().height as f32 * state.render_scale_cur) as u32,
                ));
                ui.separator();
                Slider::new("Render scale", 0.1, 2.5).build(ui, &mut state.render_scale_sel);
                if ui.button("Apply") {
                    let w_size = window.inner_size();
                    renderer.pause();
                    renderer.update_render_size(
                        w_size.width,
                        w_size.height,
                        state.render_scale_sel,
                    );
                    state.render_scale_cur = state.render_scale_sel;
                    renderer.resume();
                }
            }
        });
    state.render_window = closed;
}

fn window_stats(
    ui: &mut Ui,
    _: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    let inner_sz = window.inner_size();
    imgui::Window::new("Stats")
        .size([400.0, 400.0], Condition::Appearing)
        .position([inner_sz.width as f32 - 200.0, 50.0], Condition::Always)
        .draw_background(false)
        .save_settings(false)
        .menu_bar(false)
        .no_decoration()
        .no_inputs()
        .movable(false)
        .build(ui, || {
            let stats = renderer.stats();
            ui.text(format!("FPS: {}", (stats.fps + 0.5) as u32)); // round to nearest integer
            ui.text(format!(
                "Draw calls: {}",
                (stats.avg_draw_calls + 0.5) as u32
            ));
        });
}

fn open_scene(renderer: &mut RealtimeRenderer) {
    let dialog = nfd2::open_file_dialog(None, None);
    match dialog {
        Ok(Response::Okay(path)) => match parse(path) {
            Ok(parsed) => renderer.change_scene(parsed.scene()),
            Err(_) => log::error!("Failed to parse scene file"),
        },
        Ok(Response::Cancel) => (),
        _ => log::error!("Error opening file dialog"),
    }
}
