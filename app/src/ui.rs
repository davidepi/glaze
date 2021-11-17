use glaze::{parse, RealtimeRenderer};
use imgui::{MenuItem, Ui};
use nfd2::Response;

pub struct UiState {}

impl Default for UiState {
    fn default() -> Self {
        Self::new()
    }
}

impl UiState {
    pub fn new() -> Self {
        UiState {}
    }
}

pub fn draw_ui(ui: &mut Ui, state: &mut UiState, renderer: &mut RealtimeRenderer) {
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if MenuItem::new("Open").shortcut("Ctrl+O").build(ui) {
                open_scene(renderer);
            }
        });
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
