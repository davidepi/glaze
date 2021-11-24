use glaze::{
    parse, Camera, OrthographicCam, PerspectiveCam, RealtimeRenderer, ShaderMat, TextureFormat,
    VulkanScene,
};
use imgui::{
    CollapsingHeader, ColorEdit, ColorEditFlags, ColorPicker, ComboBox, Condition, Image, MenuItem,
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
    materials_window: bool,
    materials_selected: Option<u16>,
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
            materials_window: false,
            materials_selected: None,
            stats_window: true,
        }
    }

    pub fn change_scene(&mut self) {
        self.textures_selected = None;
    }
}

pub fn draw_ui(ui: &Ui, state: &mut UiState, window: &mut Window, renderer: &mut RealtimeRenderer) {
    ui.show_demo_window(&mut state.render_window);
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if MenuItem::new("Open").shortcut("Ctrl+O").build(ui) {
                open_scene(renderer, state);
            }
        });
        ui.menu("Window", || {
            ui.checkbox("Render", &mut state.render_window);
            ui.checkbox("Textures", &mut state.textures_window);
            ui.checkbox("Materials", &mut state.materials_window);
            ui.checkbox("Stats", &mut state.stats_window);
        });
    });
    if state.stats_window {
        window_stats(ui, state, window, renderer);
    }
    if state.render_window {
        window_render(ui, state, window, renderer);
    }
    if state.textures_window {
        window_textures(ui, state, window, renderer);
    }
    if state.materials_window {
        window_materials(ui, state, window, renderer);
    }
}

fn window_render(
    ui: &Ui,
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

fn window_textures(ui: &Ui, state: &mut UiState, _: &mut Window, renderer: &mut RealtimeRenderer) {
    let mut closed = &mut state.textures_window;
    let selected = &mut state.textures_selected;
    let scene = renderer.scene();
    let preview = match (&selected, scene) {
        (Some(id), Some(scene)) => {
            let texture = scene.textures.get(id).unwrap();
            &texture.info.name
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
                        let mut all_textures = scene.textures.iter().collect::<Vec<_>>();
                        all_textures.sort_by_key(|(&id, _)| id);
                        all_textures.into_iter().for_each(|(id, texture)| {
                            if Selectable::new(&texture.info.name).build(ui) {
                                *selected = Some(*id);
                            }
                        });
                    }
                });
            if let Some(selected) = selected {
                ui.separator();
                let info = &scene.unwrap().textures.get(selected).unwrap().info;
                ui.text(format!("Resolution {}x{}", info.width, info.height));
                ui.text(format!("Format {}", channels_to_string(info.format)));
                Image::new(TextureId::new(*selected as usize), [256.0, 256.0]).build(ui);
            }
        });
}

fn channels_to_string(colortype: TextureFormat) -> &'static str {
    match colortype {
        TextureFormat::Gray => "Grayscale",
        TextureFormat::Rgb => "RGB",
        TextureFormat::Rgba => "RGBA",
    }
}

fn window_materials(
    ui: &Ui,
    state: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    let mut closed = &mut state.materials_window;
    let selected = &mut state.materials_selected;
    let scene = renderer.scene();
    let preview = match (&selected, scene) {
        (Some(id), Some(scene)) => {
            let (material, _) = scene.materials.get(id).unwrap();
            &material.name
        }
        _ => "",
    };
    if let Some(window) = imgui::Window::new("Materials")
        .opened(&mut closed)
        .save_settings(false)
        .begin(ui)
    {
        if let Some(mat_combo) = ComboBox::new("Material name")
            .preview_value(preview)
            .begin(ui)
        {
            if let Some(scene) = scene {
                let mut all_mats = scene.materials.iter().collect::<Vec<_>>();
                all_mats.sort_by_key(|(&id, _)| id);
                all_mats.into_iter().for_each(|(id, (mat, _))| {
                    if Selectable::new(&mat.name).build(ui) {
                        *selected = Some(*id);
                    }
                });
            }
            mat_combo.end();
        }
        if let (Some(selected), Some(scene)) = (selected, scene) {
            let mut changed = false;
            let mut new_shader = None;
            let mut new_diffuse = None;
            let mut new_diff_mul = None;
            ui.separator();
            let current = &scene.materials.get(selected).unwrap().0;
            if let Some(shader_combo) = ComboBox::new("Type")
                .preview_value(current.shader.name())
                .begin(ui)
            {
                for shader in ShaderMat::all_values() {
                    if Selectable::new(shader.name()).build(ui) {
                        changed = true;
                        new_shader = Some(shader);
                    }
                }
                shader_combo.end();
            }
            let diffuse = texture_selector(&ui, "Diffuse", current.diffuse, &scene);
            if diffuse != current.diffuse {
                changed = true;
                new_diffuse = Some(diffuse);
            }
            let mut color = [
                current.diffuse_mul[0] as f32 / 255.0,
                current.diffuse_mul[1] as f32 / 255.0,
                current.diffuse_mul[2] as f32 / 255.0,
            ];
            if ColorPicker::new("Diffuse multiplier", &mut color)
                .small_preview(true)
                .build(ui)
            {
                changed = true;
                new_diff_mul = Some([
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                ]);
            }
            if changed {
                let mut new_mat = current.clone();
                if let Some(shader) = new_shader {
                    new_mat.shader = shader;
                } else if let Some(new_diffuse) = new_diffuse {
                    new_mat.diffuse = new_diffuse;
                } else if let Some(new_diff_mul) = new_diff_mul {
                    new_mat.diffuse_mul = new_diff_mul;
                }
                renderer.change_material(*selected, new_mat);
            }
        }
        window.end();
    }
}

fn texture_selector(
    ui: &Ui,
    text: &str,
    mut selected: Option<u16>,
    scene: &VulkanScene,
) -> Option<u16> {
    let name = if let Some(id) = selected {
        &scene.textures.get(&id).unwrap().info.name
    } else {
        ""
    };
    if let Some(cb) = ComboBox::new(text).preview_value(name).begin(ui) {
        if Selectable::new("").build(ui) {
            selected = None;
        }
        for (id, texture) in scene.textures.iter() {
            if Selectable::new(&texture.info.name).build(ui) {
                selected = Some(*id);
            }
        }
        cb.end();
    }
    ui.same_line();
    if let Some(selected) = selected {
        Image::new(TextureId::new(selected as usize), [32.0, 32.0]).build(ui);
    }
    selected
}

fn window_stats(ui: &Ui, _: &mut UiState, window: &mut Window, renderer: &mut RealtimeRenderer) {
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
                renderer.change_scene(parsed);
                state.change_scene();
            }
            Err(_) => log::error!("Failed to parse scene file"),
        },
        Ok(Response::Cancel) => (),
        _ => log::error!("Error opening file dialog"),
    }
}
