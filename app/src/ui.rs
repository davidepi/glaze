use std::time::Instant;

use glaze::{
    parse, Camera, OrthographicCam, PerspectiveCam, RealtimeRenderer, ShaderMat, TextureFormat,
    VulkanScene,
};
use imgui::{
    CollapsingHeader, ColorEdit, ComboBox, Condition, Image, ImageButton, MenuItem, PopupModal,
    Selectable, SelectableFlags, Slider, SliderFlags, TextureId, Ui,
};
use nfd2::Response;
use winit::window::Window;

pub struct UiState {
    open_loading_popup: bool,
    current_tick: usize,
    last_tick_time: Instant,
    settings_window: bool,
    render_scale_cur: f32,
    render_scale_sel: f32,
    pub inverted_mouse_h: bool,
    pub inverted_mouse_v: bool,
    pub mouse_sensitivity: f32,
    pub mov_speed: f32,
    pub mov_speed_mul: f32,
    pub vert_speed: f32,
    pub inverted_vert_mov: bool,
    textures_window: bool,
    textures_selected: Option<u16>,
    materials_window: bool,
    materials_selected: Option<u16>,
    stats_window: bool,
}

impl UiState {
    pub fn new() -> Self {
        UiState {
            open_loading_popup: false,
            current_tick: 0,
            last_tick_time: Instant::now(),
            settings_window: false,
            render_scale_cur: 1.0,
            render_scale_sel: 1.0,
            inverted_mouse_h: false,
            inverted_mouse_v: false,
            mouse_sensitivity: 0.05,
            mov_speed: 1.0,
            mov_speed_mul: 2.5,
            vert_speed: 0.1,
            inverted_vert_mov: false,
            textures_window: false,
            textures_selected: None,
            materials_window: false,
            materials_selected: None,
            stats_window: true,
        }
    }

    pub fn change_scene(&mut self) {
        self.textures_selected = None;
        self.materials_selected = None;
        // I need to open the popup in the same scope of the rendering
        self.open_loading_popup = true;
    }
}

pub fn draw_ui(ui: &Ui, state: &mut UiState, window: &mut Window, renderer: &mut RealtimeRenderer) {
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if MenuItem::new("Open").build(ui) {
                open_scene(renderer, state);
            }
        });
        ui.menu("Window", || {
            ui.checkbox("Settings", &mut state.settings_window);
            ui.checkbox("Textures", &mut state.textures_window);
            ui.checkbox("Materials", &mut state.materials_window);
            ui.checkbox("Stats", &mut state.stats_window);
        });
        ui.menu("About", || {
            if MenuItem::new("Help").build(ui) {
                ui.open_popup("Help me");
            }
        });
    });
    if state.stats_window {
        window_stats(ui, state, window, renderer);
    }
    if state.settings_window {
        window_settings(ui, state, window, renderer);
    }
    if state.textures_window {
        window_textures(ui, state, window, renderer);
    }
    if state.materials_window {
        window_materials(ui, state, window, renderer);
    }
    if state.open_loading_popup {
        state.open_loading_popup = false;
        ui.open_popup("Loading scene...");
    }
    if let Some(_token) = PopupModal::new("Loading scene...")
        .always_auto_resize(true)
        .resizable(false)
        .movable(false)
        .begin_popup(ui)
    {
        if let Some(load_msg) = renderer.is_loading() {
            // let spinner_ticks = ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'];
            let spinner_ticks = ['\\', '|', '/', '-'];
            let tick = spinner_ticks[state.current_tick % spinner_ticks.len()];
            if state.last_tick_time.elapsed().as_millis() > 50 {
                state.current_tick += 1;
                state.last_tick_time = Instant::now();
            }
            ui.text(format!("{} {}", tick, load_msg));
        } else {
            ui.close_current_popup();
        };
    }
}

fn window_settings(
    ui: &Ui,
    state: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
) {
    let mut closed = state.settings_window;
    imgui::Window::new("Settings")
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
                    renderer.update_render_size(
                        w_size.width,
                        w_size.height,
                        state.render_scale_sel,
                    );
                    state.render_scale_cur = state.render_scale_sel;
                }
                ui.separator();
                let mut color = renderer.get_clear_color();
                if ColorEdit::new("Background color", &mut color)
                    .inputs(false)
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
                                    near: 0.1,
                                    far: 250.0,
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
                                    near: 0.1,
                                    far: 250.0,
                                });
                            }
                        }
                    });
                match renderer.camera_mut() {
                    Some(Camera::Perspective(cam)) => {
                        Slider::new("Near clipping plane", 0.01, 1.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut cam.near);
                        Slider::new("Far clipping plane", 100.0, 10000.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut cam.far);
                        let mut fovx = cam.fovx.to_degrees();
                        Slider::new("Field of View", 1.0, 150.0).build(ui, &mut fovx);
                        cam.fovx = fovx.to_radians();
                    }
                    Some(Camera::Orthographic(cam)) => {
                        Slider::new("Near clipping plane", 0.01, 1.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut cam.near);
                        Slider::new("Far clipping plane", 100.0, 10000.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut cam.far);
                        Slider::new("Scale", 1.0, 10.0).build(ui, &mut cam.scale);
                    }
                    _ => {}
                }
            }
            if CollapsingHeader::new("Controls").build(ui) {
                ui.text("Mouse");
                ui.checkbox("Invert vertical camera", &mut state.inverted_mouse_v);
                ui.checkbox("Invert horizontal camera ", &mut state.inverted_mouse_h);
                Slider::new("Sensibility", 0.01, 10.0).build(ui, &mut state.mouse_sensitivity);
                ui.checkbox("Invert vertical movement", &mut state.inverted_vert_mov);
                Slider::new("Vertical movement speed", 0.01, 10.0).build(ui, &mut state.vert_speed);
                ui.separator();
                ui.text("Keyboard");
                Slider::new("Movement speed (normal)", 0.01, 10.0).build(ui, &mut state.mov_speed);
                Slider::new("Fast movement multiplier", 1.0, 10.0)
                    .build(ui, &mut state.mov_speed_mul);
            }
        });
    state.settings_window = closed;
}

fn window_textures(ui: &Ui, state: &mut UiState, _: &mut Window, renderer: &mut RealtimeRenderer) {
    let closed = &mut state.textures_window;
    let selected = &mut state.textures_selected;
    let scene = renderer.scene();
    let preview = match (&selected, scene) {
        (Some(id), Some(scene)) => {
            let texture = scene.single_texture(*id).unwrap();
            &texture.info.name
        }
        _ => "",
    };
    imgui::Window::new("Textures")
        .opened(closed)
        .size([300.0, 300.0], Condition::Appearing)
        .save_settings(false)
        .build(ui, || {
            ComboBox::new("Texture name")
                .preview_value(preview)
                .build(ui, || {
                    if let Some(scene) = scene {
                        let mut all_textures = scene.textures().into_iter().collect::<Vec<_>>();
                        all_textures.sort_by_key(|(id, _)| *id);
                        all_textures.into_iter().for_each(|(id, texture)| {
                            if Selectable::new(&texture.info.name).build(ui) {
                                *selected = Some(id);
                            }
                        });
                    }
                });
            if let Some(selected) = selected {
                ui.separator();
                let info = &scene.unwrap().single_texture(*selected).unwrap().info;
                ui.text(format!("Resolution {}x{}", info.width, info.height));
                ui.text(format!("Format {}", channels_to_string(info.format)));
                let tex_w = 512.0;
                let tex_h = 512.0;
                let pos = ui.cursor_screen_pos();
                Image::new(TextureId::new(*selected as usize), [tex_w, tex_h]).build(ui);
                if ui.is_item_hovered() {
                    ui.tooltip(|| {
                        let region_sz = 32.0;
                        let mpos = ui.io().mouse_pos;
                        let mut region_x = mpos[0] - pos[0] - region_sz * 0.5;
                        let mut region_y = mpos[1] - pos[1] - region_sz * 0.5;
                        let zoom = 4.0;
                        region_x = region_x.clamp(0.0, tex_w - region_sz);
                        region_y = region_y.clamp(0.0, tex_h - region_sz);
                        let uv0 = [region_x / tex_w, region_y / tex_h];
                        let uv1 = [
                            (region_x + region_sz) / tex_w,
                            (region_y + region_sz) / tex_h,
                        ];
                        Image::new(
                            TextureId::new(*selected as usize),
                            [region_sz * zoom, region_sz * zoom],
                        )
                        .uv0(uv0)
                        .uv1(uv1)
                        .build(ui);
                    });
                }
            }
        });
}

fn channels_to_string(colortype: TextureFormat) -> &'static str {
    match colortype {
        TextureFormat::Gray => "Grayscale",
        TextureFormat::Rgba => "RGBA",
    }
}

fn window_materials(ui: &Ui, state: &mut UiState, _: &mut Window, renderer: &mut RealtimeRenderer) {
    let closed = &mut state.materials_window;
    let selected = &mut state.materials_selected;
    let scene = renderer.scene();
    let preview = match (&selected, scene) {
        (Some(id), Some(scene)) => {
            let material = scene.single_material(*id).unwrap();
            &material.name
        }
        _ => "",
    };
    if let Some(window) = imgui::Window::new("Materials")
        .opened(closed)
        .size([400.0, 400.0], Condition::Appearing)
        .save_settings(false)
        .begin(ui)
    {
        if let Some(mat_combo) = ComboBox::new("Material name")
            .preview_value(preview)
            .begin(ui)
        {
            if let Some(scene) = scene {
                let mut all_mats = scene.materials();
                all_mats.sort_by_key(|(id, _)| *id);
                all_mats.into_iter().for_each(|(id, mat)| {
                    if Selectable::new(&mat.name).build(ui) {
                        *selected = Some(id);
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
            let mut new_opacity = None;
            ui.separator();
            let current = scene.single_material(*selected).unwrap();
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
            let (diff, diff_clicked) = texture_selector(ui, "Diffuse", current.diffuse, scene);
            if diff != current.diffuse {
                changed = true;
                new_diffuse = Some(diff);
            }
            if diff_clicked {
                state.textures_selected = diff;
                state.textures_window = true;
            }
            let mut color = [
                current.diffuse_mul[0] as f32 / 255.0,
                current.diffuse_mul[1] as f32 / 255.0,
                current.diffuse_mul[2] as f32 / 255.0,
            ];
            if ColorEdit::new("Diffuse multiplier", &mut color)
                .inputs(false)
                .build(ui)
            {
                changed = true;
                new_diff_mul = Some([
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                ]);
            }
            let (opac, opac_clicked) = texture_selector(ui, "Opacity", current.opacity, scene);
            if opac != current.opacity {
                changed = true;
                new_opacity = Some(opac);
            }
            if opac_clicked {
                state.textures_selected = opac;
                state.textures_window = true;
            }
            if changed {
                let mut new_mat = current.clone();
                if let Some(shader) = new_shader {
                    new_mat.shader = shader;
                } else if let Some(new_diffuse) = new_diffuse {
                    new_mat.diffuse = new_diffuse;
                } else if let Some(new_diff_mul) = new_diff_mul {
                    new_mat.diffuse_mul = new_diff_mul;
                } else if let Some(new_opacity) = new_opacity {
                    new_mat.opacity = new_opacity;
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
) -> (Option<u16>, bool) {
    let mut clicked_on_preview = false;
    let name = if let Some(id) = selected {
        &scene.single_texture(id).unwrap().info.name
    } else {
        ""
    };
    if let Some(cb) = ComboBox::new(text).preview_value(name).begin(ui) {
        if Selectable::new("").build(ui) {
            selected = None;
        }
        for (id, texture) in scene.textures().into_iter() {
            if Selectable::new(&texture.info.name).build(ui) {
                selected = Some(id);
            }
            if ui.is_item_hovered() {
                ui.tooltip(|| {
                    Image::new(TextureId::new(id as usize), [128.0, 128.0]).build(ui);
                });
            }
        }
        cb.end();
    }
    if let Some(selected) = selected {
        ui.same_line();
        if ImageButton::new(TextureId::new(selected as usize), [16.0, 16.0])
            .frame_padding(0)
            .build(ui)
        {
            clicked_on_preview = true;
        }
        if ui.is_item_hovered() {
            ui.tooltip(|| {
                ui.text(name);
                Image::new(TextureId::new(selected as usize), [256.0, 256.0]).build(ui);
            });
        }
    }
    (selected, clicked_on_preview)
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
