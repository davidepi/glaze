use glaze::{parse, Camera, OrthographicCam, PerspectiveCam, RealtimeRenderer};
use imgui::{
    CollapsingHeader, ColorEditFlags, ColorPicker, ComboBox, Condition, Image, MenuItem,
    Selectable, SelectableFlags, Slider, TextureId, Ui,
};
use nfd2::Response;
use winit::window::Window;

pub struct UiState {
    render_window: bool,
    render_scale_cur: f32,
    render_scale_sel: f32,
    textures_window: bool,
    textures_selected: Option<u16>,
    stats_window: bool,
}

impl UiState {
    pub fn new() -> Self {
        UiState {
            render_window: false,
            render_scale_cur: 1.0,
            render_scale_sel: 1.0,
            textures_window: false,
            textures_selected: None,
            stats_window: true,
        }
    }

    pub fn change_scene(&mut self) {
        self.textures_selected = None;
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
                open_scene(renderer, state);
            }
        });
        ui.menu("Window", || {
            ui.checkbox("Render", &mut state.render_window);
            ui.checkbox("Stats", &mut state.stats_window);
            ui.checkbox("Textures", &mut state.textures_window);
        });
    });
    if state.render_window {
        window_render(ui, state, window, renderer);
    }
    if state.textures_window {
        window_textures(ui, state, window, renderer);
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
                ui.separator();
                let mut color = renderer.get_clear_color();
                if ColorPicker::new("Background color", &mut color)
                    .flags(ColorEditFlags::NO_ALPHA)
                    .build(ui)
                {
                    renderer.set_clear_color(color);
                }
                ui.separator();
                let (camera_name, disabled) = {
                    let camera = renderer.camera_mut();
                    let name = match camera {
                        Some(Camera::Perspective(_)) => "Perspective",
                        Some(Camera::Orthographic(_)) => "Orthographic",
                        None => "None",
                    };
                    let disabled = if camera.is_some() {
                        SelectableFlags::empty()
                    } else {
                        SelectableFlags::DISABLED
                    };
                    (name, disabled)
                };
                ui.text(format!("Current camera type: {}", camera_name));
                ComboBox::new("Camera type")
                    .preview_value(camera_name)
                    .build(ui, || {
                        if Selectable::new("Perspective").flags(disabled).build(ui) {
                            let camera = renderer.camera_mut().unwrap();
                            if let Camera::Perspective(_) = camera {
                            } else {
                                *camera = Camera::Perspective(PerspectiveCam {
                                    position: camera.position(),
                                    target: camera.target(),
                                    up: camera.up(),
                                    fovx: 90.0_f32.to_radians(),
                                });
                            }
                        }
                        if Selectable::new("Orthographic").flags(disabled).build(ui) {
                            let camera = renderer.camera_mut().unwrap();
                            if let Camera::Orthographic(_) = camera {
                            } else {
                                *camera = Camera::Orthographic(OrthographicCam {
                                    position: camera.position(),
                                    target: camera.target(),
                                    up: camera.up(),
                                    scale: 5.0,
                                });
                            }
                        }
                    });
                match renderer.camera_mut() {
                    Some(Camera::Perspective(cam)) => {
                        let mut fovx = cam.fovx.to_degrees();
                        Slider::new("Field of View", 1.0, 150.0).build(ui, &mut fovx);
                        cam.fovx = fovx.to_radians();
                    }
                    Some(Camera::Orthographic(cam)) => {
                        Slider::new("Scale", 1.0, 10.0).build(ui, &mut cam.scale);
                    }
                    _ => {}
                }
            }
        });
    state.render_window = closed;
}

fn window_textures(
    ui: &mut Ui,
    state: &mut UiState,
    _: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    let mut closed = &mut state.textures_window;
    let selected = &mut state.textures_selected;
    let scene = renderer.scene();
    let preview = match (&selected, scene) {
        (Some(id), Some(scene)) => {
            let info = scene.texinfo.get(id).unwrap();
            &info.name
        }
        _ => "",
    };
    imgui::Window::new("Textures")
        .opened(&mut closed)
        .size([300.0, 300.0], Condition::Appearing)
        .save_settings(false)
        .build(ui, || {
            ComboBox::new("Texture name")
                .preview_value(preview)
                .build(ui, || {
                    if let Some(scene) = scene {
                        for (id, _) in &scene.textures {
                            let name = &scene.texinfo.get(id).unwrap().name;
                            if Selectable::new(name).build(ui) {
                                *selected = Some(*id);
                            }
                        }
                    }
                });
            if let Some(selected) = selected {
                ui.separator();
                let info = scene.unwrap().texinfo.get(selected).unwrap();
                ui.text(format!("Resolution {}x{}", info.width, info.height));
                ui.text(format!("Format {}", channels_to_string(info.channels)));
                Image::new(TextureId::new(*selected as usize), [256.0, 256.0]).build(ui);
            }
        });
}

fn channels_to_string(colortype: image::ColorType) -> &'static str {
    match colortype {
        image::ColorType::L8 => "L8",
        image::ColorType::Rgb8 => "RGB8",
        image::ColorType::Rgba8 => "RGBA8",
        _ => "Unknown",
    }
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

fn open_scene(renderer: &mut RealtimeRenderer, state: &mut UiState) {
    let dialog = nfd2::open_file_dialog(None, None);
    match dialog {
        Ok(Response::Okay(path)) => match parse(path) {
            Ok(parsed) => {
                renderer.change_scene(parsed.scene());
                state.change_scene();
            }
            Err(_) => log::error!("Failed to parse scene file"),
        },
        Ok(Response::Cancel) => (),
        _ => log::error!("Error opening file dialog"),
    }
}
