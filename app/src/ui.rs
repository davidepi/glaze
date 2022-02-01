use glaze::{
    parse, Camera, OrthographicCam, PerspectiveCam, PresentInstance, RealtimeRenderer, ShaderMat,
    TextureFormat, TextureLoaded, VulkanScene,
};
use imgui::{
    CollapsingHeader, ColorEdit, ComboBox, Condition, Image, ImageButton, MenuItem, PopupModal,
    Selectable, SelectableFlags, Slider, SliderFlags, TextureId, Ui,
};
use native_dialog::FileDialog;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Instant;
use winit::window::Window;

const RT_RESULT_TEXTURE_ID: u16 = u16::MAX - 7;

pub struct UiState {
    instance: Arc<PresentInstance>,
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
    info_window: bool,
    render_window: bool,
    rt_width: i32,
    rt_height: i32,
    last_render_w: i32,
    last_render_h: i32,
    rtrenderer: Option<(Receiver<String>, JoinHandle<TextureLoaded>)>,
    rtrenderer_console: String,
    rtrenderer_has_result: bool,
    loading_scene: Option<SceneLoad>,
}

impl UiState {
    pub fn new(instance: Arc<PresentInstance>) -> Self {
        UiState {
            instance,
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
            info_window: false,
            render_window: false,
            rt_width: 1920,
            rt_height: 1080,
            last_render_w: 1920,
            last_render_h: 1080,
            rtrenderer: None,
            rtrenderer_console: String::with_capacity(1024),
            rtrenderer_has_result: false,
            loading_scene: None,
        }
    }

    fn clear_rtrenderer(&mut self) {
        self.rtrenderer = None;
        self.rtrenderer_console.clear();
        self.rtrenderer_has_result = false;
    }
}

/// Contains the loading scene state.
struct SceneLoad {
    /// The channel receaving updates on the loading progress.
    reader: mpsc::Receiver<String>,
    /// Last message extracted from the reader.
    last_message: String,
    /// Thread join handle.
    join_handle: std::thread::JoinHandle<Result<VulkanScene, std::io::Error>>,
}

pub fn draw_ui(ui: &Ui, state: &mut UiState, window: &mut Window, renderer: &mut RealtimeRenderer) {
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if MenuItem::new("Open").build(ui) && state.loading_scene.is_none() {
                open_scene(state);
            }
            if MenuItem::new("Save").build(ui) {
                if let Some(scene) = renderer.scene_mut() {
                    if let Err(error) = scene.save() {
                        log::error!("Failed to save scene: {}", error);
                    }
                }
            }
        });
        ui.menu("Rendering", || {
            ui.checkbox("Render", &mut state.render_window);
            if renderer.instance().supports_raytrace() {
                ui.checkbox("Realtime raytracing", &mut renderer.use_raytracer);
            }
        });
        ui.menu("Window", || {
            ui.checkbox("Settings", &mut state.settings_window);
            ui.checkbox("Textures", &mut state.textures_window);
            ui.checkbox("Materials", &mut state.materials_window);
            ui.checkbox("Stats", &mut state.stats_window);
        });
        ui.menu("Help", || {
            ui.checkbox("Info", &mut state.info_window);
        });
    });
    if state.stats_window {
        window_stats(ui, window, &state.instance, renderer);
    }
    if state.settings_window {
        window_settings(ui, state, window, renderer);
    }
    if state.render_window {
        window_render(ui, window, state, renderer);
    }
    if state.textures_window {
        window_textures(ui, state, renderer);
    }
    if state.materials_window {
        window_materials(ui, state, renderer);
    }
    if state.info_window {
        window_info(ui, state);
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
        // update message if necessary
        let finished = if let Some(loading) = &mut state.loading_scene {
            match loading.reader.try_recv() {
                Ok(msg) => {
                    loading.last_message = msg;
                    false
                }
                Err(TryRecvError::Empty) => false,
                Err(TryRecvError::Disconnected) => true,
            }
        } else {
            true //should never fall here unless some serious bugs exist in the UI
        };
        if finished {
            // swap scene
            let scene = state
                .loading_scene
                .take()
                .unwrap()
                .join_handle
                .join()
                .expect("Failed to wait thread");
            match scene {
                Ok(loaded) => renderer.change_scene(loaded),
                Err(e) => log::error!("Failed to load scene {e}"),
            }
            ui.close_current_popup();
        } else {
            let spinner_ticks = ['\\', '|', '/', '-'];
            let tick = spinner_ticks[state.current_tick % spinner_ticks.len()];
            if state.last_tick_time.elapsed().as_millis() > 50 {
                state.current_tick += 1;
                state.last_tick_time = Instant::now();
            }
            let load_msg = &state.loading_scene.as_ref().unwrap().last_message;
            ui.text(format!("{tick} {load_msg}"));
        }
    }
}

fn window_settings(ui: &Ui, state: &mut UiState, window: &Window, renderer: &mut RealtimeRenderer) {
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
                    let camera = renderer.camera();
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
                            let camera = renderer.camera().unwrap().clone();
                            if let Camera::Perspective(_) = camera {
                            } else {
                                renderer.set_camera(Camera::Perspective(PerspectiveCam {
                                    position: camera.position(),
                                    target: camera.target(),
                                    up: camera.up(),
                                    fovx: 90.0_f32.to_radians(),
                                    near: 0.1,
                                    far: 250.0,
                                }));
                            }
                        }
                        if Selectable::new("Orthographic").flags(disabled).build(ui) {
                            let camera = renderer.camera().unwrap().clone();
                            if let Camera::Orthographic(_) = camera {
                            } else {
                                renderer.set_camera(Camera::Orthographic(OrthographicCam {
                                    position: camera.position(),
                                    target: camera.target(),
                                    up: camera.up(),
                                    scale: 5.0,
                                    near: 0.1,
                                    far: 250.0,
                                }));
                            }
                        }
                    });
                let original_cam = renderer.camera().cloned();
                let new_cam = match &original_cam {
                    Some(Camera::Perspective(cam)) => {
                        let mut near = cam.near;
                        let mut far = cam.far;
                        let mut fovx = cam.fovx.to_degrees();
                        Slider::new("Near clipping plane", 0.01, 1.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut near);
                        Slider::new("Far clipping plane", 100.0, 10000.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut far);
                        Slider::new("Field of View", 1.0, 150.0).build(ui, &mut fovx);
                        let mut new_cam = cam.clone();
                        new_cam.near = near;
                        new_cam.far = far;
                        new_cam.fovx = fovx.to_radians();
                        Some(Camera::Perspective(new_cam))
                    }
                    Some(Camera::Orthographic(cam)) => {
                        let mut near = cam.near;
                        let mut far = cam.far;
                        let mut scale = cam.scale;
                        Slider::new("Near clipping plane", 0.01, 1.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut near);
                        Slider::new("Far clipping plane", 100.0, 10000.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut far);
                        Slider::new("Scale", 1.0, 10.0).build(ui, &mut scale);
                        let mut new_cam = cam.clone();
                        new_cam.near = near;
                        new_cam.far = far;
                        new_cam.scale = scale;
                        Some(Camera::Orthographic(new_cam))
                    }
                    None => None,
                };
                if original_cam != new_cam {
                    renderer.set_camera(new_cam.unwrap());
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

fn window_textures(ui: &Ui, state: &mut UiState, renderer: &RealtimeRenderer) {
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
        .size([600.0, 600.0], Condition::Appearing)
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
                let window_w = ui.window_size()[0];
                let ar = info.width as f32 / info.height as f32;
                let tex_w = f32::min(info.width as f32, window_w);
                let tex_h = tex_w / ar;
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

fn window_materials(ui: &Ui, state: &mut UiState, renderer: &mut RealtimeRenderer) {
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
                state.textures_selected = Some(diff);
                state.textures_window = true;
            }
            let mut color = [
                current.diffuse_mul[0] as f32 / 255.0,
                current.diffuse_mul[1] as f32 / 255.0,
                current.diffuse_mul[2] as f32 / 255.0,
                current.diffuse_mul[3] as f32 / 255.0,
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
                    (color[3] * 255.0) as u8,
                ]);
            }
            let (opac, opac_clicked) = texture_selector(ui, "Opacity", current.opacity, scene);
            if opac != current.opacity {
                changed = true;
                new_opacity = Some(opac);
            }
            if opac_clicked {
                state.textures_selected = Some(opac);
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

fn texture_selector(ui: &Ui, text: &str, mut selected: u16, scene: &VulkanScene) -> (u16, bool) {
    let mut clicked_on_preview = false;
    let name = &scene.single_texture(selected).unwrap().info.name;
    if let Some(cb) = ComboBox::new(text).preview_value(name).begin(ui) {
        for (id, texture) in scene.textures().into_iter() {
            if Selectable::new(&texture.info.name).build(ui) {
                selected = id;
            }
            if ui.is_item_hovered() {
                ui.tooltip(|| {
                    Image::new(TextureId::new(id as usize), [128.0, 128.0]).build(ui);
                });
            }
        }
        cb.end();
    }
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
    (selected, clicked_on_preview)
}

fn window_stats(ui: &Ui, window: &Window, instance: &PresentInstance, renderer: &RealtimeRenderer) {
    let inner_sz = window.inner_size();
    let device = instance.device_properties();
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
            ui.text(format!("[{}]", device.name));
            ui.text(format!("FPS: {}", (stats.fps + 0.5) as u32)); // round to nearest integer
            ui.text(format!(
                "Draw calls: {}",
                (stats.avg_draw_calls + 0.5) as u32
            ));
        });
}

fn window_info(ui: &Ui, state: &mut UiState) {
    let closed = &mut state.info_window;
    let device = state.instance.device_properties();
    if let Some(window) = imgui::Window::new("Info")
        .opened(closed)
        .size([300.0, 200.0], Condition::Appearing)
        .save_settings(false)
        .begin(ui)
    {
        ui.text(format!("Vendor: {}", device.vendor));
        ui.text(format!("Adapter: {}", device.name));
        ui.text(format!("Vulkan version: {}", device.vulkan_api_ver));
        ui.text(format!("Driver version: {}", device.driver_ver));
        ui.separator();
        ui.text("Loaded extensions:");
        for ext in state.instance.loaded_extensions() {
            ui.text(ext);
        }
        window.end()
    }
}

fn window_render(ui: &Ui, window: &Window, state: &mut UiState, renderer: &mut RealtimeRenderer) {
    let closed = &mut state.render_window;
    let sizew = window.inner_size().width as f32 * 0.9;
    let sizeh = window.inner_size().height as f32 * 0.9;
    if let Some(window) = imgui::Window::new("Render")
        .size([sizew, sizeh], Condition::Appearing)
        .opened(closed)
        .begin(ui)
    {
        ui.set_next_item_width(sizew / 4.0);
        imgui::InputInt::new(ui, "Width", &mut state.rt_width).build();
        ui.same_line();
        ui.set_next_item_width(sizew / 4.0);
        imgui::InputInt::new(ui, "Height", &mut state.rt_height).build();
        if let Some((rchan, _)) = &state.rtrenderer {
            match rchan.try_recv() {
                Ok(msg) => {
                    state.rtrenderer_console.push_str(&msg);
                    state.rtrenderer_console.push('\n');
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    let (_, handle) = state.rtrenderer.take().unwrap();
                    if let Ok(res) = handle.join() {
                        renderer.add_texture(RT_RESULT_TEXTURE_ID, res);
                        state.rtrenderer_has_result = true;
                    } else {
                        state.rtrenderer_console.push_str("Render failed");
                    }
                }
            }
            if ui.button("Cancel") {
                state.clear_rtrenderer();
            }
        } else if ui.button("Render") {
            state.clear_rtrenderer();
            state.last_render_w = state.rt_width;
            state.last_render_h = state.rt_height;
            let r = renderer.get_raytrace(state.rt_width as u32, state.rt_height as u32);
            // this block handles "missing scene" and "card not supported"
            match r {
                Ok(r) => {
                    state.rtrenderer_console.clear();
                    state.rtrenderer_console.push_str("Successfully Created\n");
                    let (wchan, rchan) = mpsc::channel();
                    let handle = std::thread::spawn(move || r.draw(wchan));
                    state.rtrenderer = Some((rchan, handle));
                }
                Err(e) => {
                    let msg = e.to_string();
                    state.rtrenderer_console.push_str(&format!("{msg}\n"))
                }
            }
        }
        if state.rtrenderer_has_result {
            if ui.button("Save") {
                let texture = renderer
                    .scene()
                    .unwrap()
                    .single_texture(RT_RESULT_TEXTURE_ID)
                    .unwrap();
                let image = texture.export();
                save_image(image);
            }
            let ar = state.last_render_w as f32 / state.last_render_h as f32;
            let imagew = f32::min(state.last_render_w as f32, ui.window_size()[0]);
            let imageh = imagew / ar;
            Image::new(
                TextureId::new(RT_RESULT_TEXTURE_ID as usize),
                [imagew, imageh],
            )
            .build(ui);
        }
        ui.text("Log");
        ui.text_wrapped(&state.rtrenderer_console);
        window.end()
    }
}

fn open_scene(state: &mut UiState) {
    let dialog = FileDialog::new().show_open_single_file();
    match dialog {
        Ok(Some(path)) => match parse(path) {
            Ok(parsed) => {
                let (wchan, rchan) = mpsc::channel();
                let instance_clone = Arc::clone(&state.instance);
                let thandle =
                    std::thread::spawn(move || VulkanScene::load(instance_clone, parsed, wchan));
                state.loading_scene = Some(SceneLoad {
                    reader: rchan,
                    last_message: "Loading...".to_string(),
                    join_handle: thandle,
                });
                state.textures_selected = None;
                state.materials_selected = None;
                state.open_loading_popup = true;
                state.clear_rtrenderer();
            }
            Err(err) => log::error!("Failed to parse scene file: {}", err),
        },
        Ok(None) => (),
        _ => log::error!("Error opening file dialog"),
    }
}

fn save_image(img: image::RgbaImage) {
    let dialog = FileDialog::new()
        .add_filter("PNG image", &["png"])
        .show_save_single_file();
    match dialog {
        Ok(path) => {
            if let Some(path) = path {
                if let Err(e) = img.save(path) {
                    log::error!("Failed to save image: {e}");
                }
            }
        }
        Err(err) => {
            log::error!("Failed to open save path: {err}");
        }
    }
}
