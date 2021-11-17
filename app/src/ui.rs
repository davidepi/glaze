use glaze::{parse, RealtimeRenderer};
use imgui::{ComboBox, Condition, MenuItem, Selectable, Ui};
use nfd2::Response;
use winit::window::Window;

const VIEWPORT_MASK: u32 = 0x1;

pub struct UiState {
    visible_windows: u32,
    sel_rs: u8,
    cur_rs: u8,
    rs: Vec<(u32, u32)>,
}

impl UiState {
    pub fn new(monitor_size: (u32, u32)) -> Self {
        let multipliers = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.10, 1.15, 1.25, 1.5, 1.75, 2.0,
            2.25, 2.5,
        ];
        let mut available_render_size = Vec::with_capacity(multipliers.len());
        for i in multipliers {
            let size = (
                (monitor_size.0 as f32 * i) as u32,
                (monitor_size.1 as f32 * i) as u32,
            );
            available_render_size.push(size);
        }
        UiState {
            visible_windows: 0,
            sel_rs: 4,
            cur_rs: 4,
            rs: available_render_size,
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
            ui.checkbox_flags("Viewport", &mut state.visible_windows, VIEWPORT_MASK);
        });
    });
    if state.visible_windows != 0 {
        render_windows(ui, state, window, renderer);
    }
}

fn render_windows(
    ui: &mut Ui,
    state: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    if state.visible_windows & VIEWPORT_MASK != 0 {
        imgui::Window::new("Viewport")
            .size([400.0, 300.0], Condition::Appearing)
            .build(ui, || {
                ui.text("Viewport");
                ui.separator();
                ui.text("Current render size:");
                ui.text(format!(
                    "Render size: {}x{}",
                    state.rs[state.cur_rs as usize].0, state.rs[state.cur_rs as usize].1
                ));
                ui.separator();
                let selected_rs = state.rs[state.sel_rs as usize];
                let preview_rs = format!("{}x{}", selected_rs.0, selected_rs.1);
                if let Some(token) = ComboBox::new("Render size")
                    .preview_value(preview_rs)
                    .begin(ui)
                {
                    for (idx, (width, height)) in state.rs.iter().enumerate() {
                        if Selectable::new(format!("{}x{}", width, height)).build(ui) {
                            state.sel_rs = idx as u8;
                        }
                    }
                    token.end();
                };
                if ui.button("Apply") {
                    renderer.pause();
                    renderer.update_render_size(
                        state.rs[state.sel_rs as usize].0,
                        state.rs[state.sel_rs as usize].1,
                    );
                    state.cur_rs = state.sel_rs;
                    renderer.resume();
                }
            });
    }
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
