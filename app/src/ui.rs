// separates better imgui logic (button clicked) from the program logic
#![allow(clippy::collapsible_if)]
use glaze::{
    parse, Camera, ColorRGB, Integrator, Light, LightType, MaterialType, Metal, OrthographicCam,
    PerspectiveCam, PresentInstance, RayTraceRenderer, RayTraceScene, RealtimeRenderer,
    RealtimeScene, Spectrum, Texture, TextureFormat, TextureInfo,
};
use image::GenericImageView;
use imgui::{
    CollapsingHeader, ColorEdit, ComboBox, Condition, Image, ImageButton, InputText, MenuItem,
    PopupModal, Selectable, Slider, SliderFlags, TextureId, Ui,
};
use rfd::FileDialog;
use std::path::PathBuf;
use std::sync::mpsc::TryRecvError;
use std::sync::{mpsc, Arc};
use std::time::Instant;
use winit::window::Window;

pub struct UiState {
    instance: Arc<PresentInstance>,
    pub use_raytracer: bool,
    open_loading_popup: bool,
    current_tick: usize,
    last_tick_time: Instant,
    settings_window: bool,
    render_scale_cur: f32,
    render_scale_sel: f32,
    pub movement_lock: bool,
    pub inverted_mouse_h: bool,
    pub inverted_mouse_v: bool,
    pub mouse_sensitivity: f32,
    pub mov_speed: f32,
    pub mov_speed_mul: f32,
    pub vert_speed: f32,
    pub inverted_vert_mov: bool,
    textures_window: bool,
    textures_selected: Option<u16>,
    texture_path: String,
    texture_format: TextureFormat,
    materials_window: bool,
    materials_selected: Option<u16>,
    lights_window: bool,
    light_selected: Option<usize>,
    spectrum_temperature: bool,
    stats_window: bool,
    info_window: bool,
    loading_scene: Option<SceneLoad>,
}

impl UiState {
    pub fn new(instance: Arc<PresentInstance>) -> Self {
        let use_raytracer = instance.supports_raytrace();
        UiState {
            instance,
            open_loading_popup: false,
            current_tick: 0,
            last_tick_time: Instant::now(),
            settings_window: false,
            render_scale_cur: 1.0,
            render_scale_sel: 1.0,
            movement_lock: false,
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
            lights_window: false,
            light_selected: None,
            spectrum_temperature: true,
            stats_window: true,
            info_window: false,
            loading_scene: None,
            use_raytracer,
            texture_path: "".to_string(),
            texture_format: TextureFormat::RgbaSrgb,
        }
    }
}

/// Contains the loading scene state.
struct SceneLoad {
    /// The channel receaving updates on the loading progress.
    reader: mpsc::Receiver<String>,
    /// Last message extracted from the reader.
    last_message: String,
    /// Thread join handle.
    join_handle: std::thread::JoinHandle<(RealtimeScene, Option<RayTraceScene<PresentInstance>>)>,
}

pub fn draw_ui(
    ui: &Ui,
    state: &mut UiState,
    window: &mut Window,
    renderer: &mut RealtimeRenderer,
    raytracer: &mut Option<RayTraceRenderer<PresentInstance>>,
) {
    ui.main_menu_bar(|| {
        ui.menu("File", || {
            if MenuItem::new("Open scene...").build(ui) && state.loading_scene.is_none() {
                select_and_load_scene(state);
            }
            if MenuItem::new("Save").build(ui) {
                if let Err(error) = renderer.save_scene() {
                    log::error!("Failed to save scene: {}", error);
                }
            }
            if MenuItem::new("Save as...").build(ui) {
                let dialog_pick = FileDialog::new()
                    .set_title("Save As...")
                    .add_filter("Glaze scene", &["glaze"])
                    .save_file();
                if let Some(path) = dialog_pick {
                    if let Err(error) = renderer.scene().save_as(path.as_os_str().to_str().unwrap())
                    {
                        log::error!("Failed to save scene: {}", error);
                    }
                }
            }
        });
        ui.menu("Rendering", || {
            if renderer.instance().supports_raytrace() {
                ui.checkbox("Realtime raytracing", &mut state.use_raytracer);
                ui.menu("Integrator", || {
                    for integrator in Integrator::values() {
                        if imgui::MenuItem::new(integrator.name()).build(ui) {
                            if let Some(raytracer) = raytracer {
                                raytracer.set_integrator(integrator);
                            }
                        }
                    }
                });
            }
        });
        ui.menu("Window", || {
            ui.checkbox("Settings", &mut state.settings_window);
            ui.checkbox("Textures", &mut state.textures_window);
            ui.checkbox("Materials", &mut state.materials_window);
            ui.checkbox("Lights", &mut state.lights_window);
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
        window_settings(ui, state, window, renderer, raytracer);
    }
    if state.textures_window {
        window_textures(ui, state, renderer, raytracer);
    }
    if state.materials_window {
        window_materials(ui, state, renderer, raytracer);
    }
    if state.lights_window {
        window_lights(ui, state, renderer, raytracer);
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
            let (scene, rtscene) = state
                .loading_scene
                .take()
                .unwrap()
                .join_handle
                .join()
                .expect("Failed to wait thread");
            renderer.change_scene(scene);
            if let (Some(raytracer), Some(rtscene)) = (raytracer, rtscene) {
                raytracer.change_scene(rtscene);
            }
            ui.close_current_popup();
            state.movement_lock = false;
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

fn window_settings(
    ui: &Ui,
    state: &mut UiState,
    window: &Window,
    renderer: &mut RealtimeRenderer,
    raytracer: &mut Option<RayTraceRenderer<PresentInstance>>,
) {
    let mut closed = state.settings_window;
    imgui::Window::new("Settings")
        .size([400.0, 400.0], Condition::Appearing)
        .opened(&mut closed)
        .save_settings(false)
        .build(ui, || {
            let w_size = window.inner_size();
            if CollapsingHeader::new("Viewport Options").build(ui) {
                ui.text("Current render scale:");
                ui.text(format!(
                    "Render scale: {}x ({}x{})",
                    state.render_scale_cur,
                    (w_size.width as f32 * state.render_scale_cur) as u32,
                    (w_size.height as f32 * state.render_scale_cur) as u32,
                ));
                ui.separator();
                Slider::new("Render scale", 0.1, 2.5).build(ui, &mut state.render_scale_sel);
                if ui.button("Apply") {
                    renderer.update_render_size(
                        w_size.width,
                        w_size.height,
                        state.render_scale_sel,
                    );
                    state.render_scale_cur = state.render_scale_sel;
                    if let Some(raytracer) = raytracer.as_mut() {
                        raytracer.change_resolution(
                            (w_size.width as f32 * state.render_scale_cur) as u32,
                            (w_size.height as f32 * state.render_scale_cur) as u32,
                        )
                    }
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
                let camera_name = match renderer.camera() {
                    Camera::Perspective(_) => "Perspective",
                    Camera::Orthographic(_) => "Orthographic",
                };
                ui.text(format!("Current camera type: {}", camera_name));
                let old_cam = renderer.camera();
                let mut update_cam = false;
                let mut new_cam = old_cam;
                ComboBox::new("Camera type")
                    .preview_value(camera_name)
                    .build(ui, || {
                        if Selectable::new("Perspective").build(ui) {
                            if let Camera::Perspective(_) = old_cam {
                            } else {
                                update_cam = true;
                                new_cam = Camera::Perspective(PerspectiveCam {
                                    position: old_cam.position(),
                                    target: old_cam.target(),
                                    up: old_cam.up(),
                                    near: old_cam.near_plane(),
                                    far: old_cam.far_plane(),
                                    ..Default::default()
                                });
                            }
                        }
                        if Selectable::new("Orthographic").build(ui) {
                            if let Camera::Orthographic(_) = old_cam {
                            } else {
                                update_cam = true;
                                new_cam = Camera::Orthographic(OrthographicCam {
                                    position: old_cam.position(),
                                    target: old_cam.target(),
                                    up: old_cam.up(),
                                    near: old_cam.near_plane(),
                                    far: old_cam.far_plane(),
                                    ..Default::default()
                                });
                            }
                        }
                    });
                match &mut new_cam {
                    Camera::Perspective(cam) => {
                        update_cam |= Slider::new("Near clipping plane", 0.01, 1.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut cam.near)
                            || Slider::new("Far clipping plane", 100.0, 1E4)
                                .flags(SliderFlags::LOGARITHMIC)
                                .flags(SliderFlags::ALWAYS_CLAMP)
                                .build(ui, &mut cam.far);
                        let mut fovx = cam.fovx.to_degrees();
                        if Slider::new("Field of View", 1.0, 150.0).build(ui, &mut fovx) {
                            cam.fovx = fovx.to_radians();
                            update_cam = true;
                        }
                    }
                    Camera::Orthographic(cam) => {
                        update_cam |= Slider::new("Near clipping plane", 0.01, 1.0)
                            .flags(SliderFlags::ALWAYS_CLAMP)
                            .build(ui, &mut cam.near)
                            || Slider::new("Far clipping plane", 100.0, 1E4)
                                .flags(SliderFlags::ALWAYS_CLAMP)
                                .build(ui, &mut cam.far)
                            || Slider::new("Scale", 0.1, 1E4)
                                .flags(SliderFlags::ALWAYS_CLAMP)
                                .flags(SliderFlags::LOGARITHMIC)
                                .build(ui, &mut cam.scale)
                    }
                };
                if update_cam {
                    renderer.set_camera(new_cam);
                    if let Some(raytracer) = raytracer.as_mut() {
                        raytracer.update_camera(new_cam);
                    }
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
                Slider::new("Movement speed (normal)", 0.01, 100.0)
                    .flags(SliderFlags::LOGARITHMIC)
                    .build(ui, &mut state.mov_speed);
                Slider::new("Fast movement multiplier", 1.0, 1000.0)
                    .flags(SliderFlags::LOGARITHMIC)
                    .build(ui, &mut state.mov_speed_mul);
            }
        });
    state.settings_window = closed;
}

fn window_textures(
    ui: &Ui,
    state: &mut UiState,
    renderer: &mut RealtimeRenderer,
    raytracer: &mut Option<RayTraceRenderer<PresentInstance>>,
) {
    let closed = &mut state.textures_window;
    let texture_path = &mut state.texture_path;
    let format = &mut state.texture_format;
    let selected = &mut state.textures_selected;
    imgui::Window::new("Textures")
        .opened(closed)
        .size([600.0, 600.0], Condition::Appearing)
        .save_settings(false)
        .build(ui, || {
            ui.text("Texture Loader");
            InputText::new(ui, "Load texture", texture_path)
                .hint("Texture path...")
                .build();
            ui.same_line();
            if ui.button("Select...") {
                if let Some(path) = FileDialog::new()
                    .set_title("Select texture...")
                    .add_filter(
                        "Image file",
                        &[
                            "png", "jpeg", "gif", "bmp", "ico", "tiff", "webp", "dds", "tga",
                            "pnm", "pbm", "pgm", "ppm",
                        ],
                    )
                    .pick_file()
                {
                    *texture_path = path.to_str().unwrap().to_string();
                }
            }
            ComboBox::new("Texture format")
                .preview_value(format.str())
                .build(ui, || {
                    TextureFormat::values().for_each(|val| {
                        if Selectable::new(val.str()).build(ui) {
                            *format = val;
                        }
                    });
                });
            if ui.button("Load") {
                if let Ok(data) = image::open(texture_path.clone()) {
                    let name = PathBuf::from(&texture_path)
                        .file_stem()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string();
                    let info = TextureInfo {
                        name,
                        width: data.dimensions().0 as u16,
                        height: data.dimensions().1 as u16,
                        format: *format,
                    };
                    let texture = match *format {
                        TextureFormat::Gray => Texture::new_gray(info, data.to_luma8()),
                        TextureFormat::RgbaNorm | TextureFormat::RgbaSrgb => {
                            Texture::new_rgba(info, data.to_rgba8())
                        }
                    };
                    renderer.add_texture(texture);
                    if let Some(raytracer) = raytracer {
                        raytracer.refresh_binded_textures();
                    }
                } else {
                    //TODO: Handle opening error (maybe a popup?)
                }
            }
            ui.separator();
            ui.text("Texture previewer");
            let preview = match &selected {
                Some(id) => {
                    let texture = renderer.scene().single_texture(*id).unwrap();
                    &texture.info().name
                }
                _ => "",
            };
            ComboBox::new("Texture name")
                .preview_value(preview)
                .build(ui, || {
                    renderer
                        .scene()
                        .textures()
                        .iter()
                        .enumerate()
                        .for_each(|(id, texture)| {
                            if Selectable::new(&texture.info().name).build(ui) {
                                *selected = Some(id as u16);
                            }
                        });
                    // TODO: allow deletion of textures? Not sure if it will switch to default tex
                });
            if let Some(selected) = selected {
                let info = &renderer.scene().single_texture(*selected).unwrap().info();
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
        TextureFormat::RgbaSrgb => "RGBA (sRGB)",
        TextureFormat::RgbaNorm => "RGBA (linear)",
    }
}

fn window_materials(
    ui: &Ui,
    state: &mut UiState,
    renderer: &mut RealtimeRenderer,
    raytracer: &mut Option<RayTraceRenderer<PresentInstance>>,
) {
    let closed = &mut state.materials_window;
    let selected = &mut state.materials_selected;
    let scene = renderer.scene();
    let preview = match &selected {
        Some(id) => {
            let material = scene.single_material(*id).unwrap();
            &material.name
        }
        _ => "",
    };
    if let Some(window) = imgui::Window::new("Materials")
        .opened(closed)
        .size([500.0, 500.0], Condition::Appearing)
        .save_settings(false)
        .begin(ui)
    {
        if let Some(mat_combo) = ComboBox::new("Material name")
            .preview_value(preview)
            .begin(ui)
        {
            scene.materials().iter().enumerate().for_each(|(id, mat)| {
                if Selectable::new(&mat.name).build(ui) {
                    *selected = Some(id as u16);
                }
            });
            mat_combo.end();
        }
        if let Some(selected) = selected {
            let mut new_mat = None;
            ui.separator();
            let current = scene.single_material(*selected).unwrap();
            if let Some(shader_combo) = ComboBox::new("Type")
                .preview_value(current.mtype.name())
                .begin(ui)
            {
                for shader in MaterialType::all_values() {
                    if Selectable::new(shader.name()).build(ui) && current.mtype != shader {
                        let mut new = current.clone();
                        new.mtype = shader;
                        new_mat = Some(new);
                    }
                }
                shader_combo.end();
            }
            if current.mtype.is_fresnel_conductor() {
                let current_metal: Metal = current.metal;
                if let Some(metal_combo) = ComboBox::new("Metal")
                    .preview_value(current_metal.name())
                    .begin(ui)
                {
                    for metal in Metal::all_types() {
                        if Selectable::new(metal.name()).build(ui) && current_metal != metal {
                            let mut new = current.clone();
                            new.metal = metal;
                            new_mat = Some(new);
                        }
                    }
                    metal_combo.end()
                }
            }
            if current.mtype.is_fresnel_dielectric() {
                let mut ior = current.ior;
                if imgui::Slider::new("Dielectric index of refraction", 1.0, 3.0)
                    .flags(SliderFlags::ALWAYS_CLAMP)
                    .build(ui, &mut ior)
                {
                    let mut new = current.clone();
                    new.ior = ior;
                    new_mat = Some(new);
                }
            }
            if current.mtype.has_diffuse() {
                let (diff, diff_clicked) = texture_selector(ui, "Diffuse", current.diffuse, scene);
                if diff != current.diffuse {
                    let mut new = current.clone();
                    new.diffuse = diff;
                    new_mat = Some(new);
                }
                if diff_clicked {
                    state.textures_selected = Some(diff);
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
                    let mut new = current.clone();
                    new.diffuse_mul = [
                        (color[0] * 255.0) as u8,
                        (color[1] * 255.0) as u8,
                        (color[2] * 255.0) as u8,
                    ];
                    new_mat = Some(new);
                }
            }
            if current.mtype.has_roughness() {
                let (rough, rough_clicked) =
                    texture_selector(ui, "Roughness map", current.roughness, scene);
                if rough != current.roughness {
                    let mut new = current.clone();
                    new.roughness = rough;
                    new_mat = Some(new);
                }
                if rough_clicked {
                    state.textures_selected = Some(rough);
                    state.textures_window = true;
                }
                let mut rmul = current.roughness_mul.sqrt();
                if imgui::Slider::new("Roughness multiplier", 0.001, 0.999)
                    .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                    .build(ui, &mut rmul)
                {
                    let mut new = current.clone();
                    new.roughness_mul = rmul * rmul;
                    new_mat = Some(new);
                }
            }
            if current.mtype.has_metalness() {
                let (rough, rough_clicked) =
                    texture_selector(ui, "Metalness map", current.metalness, scene);
                if rough != current.metalness {
                    let mut new = current.clone();
                    new.metalness = rough;
                    new_mat = Some(new);
                }
                if rough_clicked {
                    state.textures_selected = Some(rough);
                    state.textures_window = true;
                }
                let mut mmul = current.metalness_mul;
                if imgui::Slider::new("Metalness multiplier", 0.0, 1.0)
                    .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                    .build(ui, &mut mmul)
                {
                    let mut new = current.clone();
                    new.metalness_mul = mmul;
                    new_mat = Some(new);
                }
            }
            if current.mtype.has_anisotropy() {
                let mut ani = current.anisotropy;
                if imgui::Slider::new("Anisotropy", -0.999, 0.999)
                    .flags(imgui::SliderFlags::ALWAYS_CLAMP)
                    .build(ui, &mut ani)
                {
                    let mut new = current.clone();
                    new.anisotropy = ani;
                    new_mat = Some(new);
                }
            }
            if current.mtype.has_normal() {
                let (norm, norm_clicked) = texture_selector(ui, "Normal", current.normal, scene);
                if norm != current.normal {
                    let mut new = current.clone();
                    new.normal = norm;
                    new_mat = Some(new);
                }
                if norm_clicked {
                    state.textures_selected = Some(norm);
                    state.textures_window = true;
                }
            }
            if current.mtype.has_opacity() {
                let (opac, opac_clicked) = texture_selector(ui, "Opacity", current.opacity, scene);
                if opac != current.opacity {
                    let mut new = current.clone();
                    new.opacity = opac;
                    new_mat = Some(new);
                }
                if opac_clicked {
                    state.textures_selected = Some(opac);
                    state.textures_window = true;
                }
            }
            if current.mtype.has_emission() {
                let mut emissive = current.emissive_col.is_some();
                if ui.checkbox("Emits light", &mut emissive) {
                    let mut new = current.clone();
                    if emissive {
                        new.emissive_col = Some([255, 255, 255]);
                    } else {
                        new.emissive_col = None;
                    }
                    new_mat = Some(new);
                }
                if let Some(col) = current.emissive_col {
                    let mut color = [
                        col[0] as f32 / 255.0,
                        col[1] as f32 / 255.0,
                        col[2] as f32 / 255.0,
                    ];
                    if ColorEdit::new("Emissive color", &mut color)
                        .inputs(false)
                        .build(ui)
                    {
                        let mut new = current.clone();
                        new.emissive_col = Some([
                            (color[0] * 255.0) as u8,
                            (color[1] * 255.0) as u8,
                            (color[2] * 255.0) as u8,
                        ]);
                        new_mat = Some(new);
                    }
                }
            }
            if let Some(new_mat) = new_mat {
                renderer.change_material(*selected, new_mat);
                if let Some(raytracer) = raytracer.as_mut() {
                    let materials = renderer.scene().materials().to_vec();
                    let lights = renderer.scene().lights().to_vec();
                    raytracer.update_materials_and_lights(
                        &materials,
                        &lights,
                        renderer.scene().textures(),
                    );
                }
            }
        }
        window.end();
    }
}

fn texture_selector(ui: &Ui, text: &str, mut selected: u16, scene: &RealtimeScene) -> (u16, bool) {
    let mut clicked_on_preview = false;
    let name = &scene.single_texture(selected).unwrap().info().name;
    if let Some(cb) = ComboBox::new(text).preview_value(name).begin(ui) {
        for (id, texture) in scene.textures().iter().enumerate() {
            if Selectable::new(&texture.info().name).build(ui) {
                selected = id as u16;
            }
            if ui.is_item_hovered() {
                ui.tooltip(|| {
                    Image::new(TextureId::new(id), [128.0, 128.0]).build(ui);
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

fn window_lights(
    ui: &Ui,
    state: &mut UiState,
    renderer: &mut RealtimeRenderer,
    raytracer: &mut Option<RayTraceRenderer<PresentInstance>>,
) {
    let closed = &mut state.lights_window;
    let mut add = None;
    let mut update = None;
    let mut remove = None;
    let mut exposure = renderer.exposure();
    let mut update_exposure = false;
    let mut update_lights = false;
    if let Some(window) = imgui::Window::new("Lights")
        .opened(closed)
        .size([400.0, 400.0], Condition::Appearing)
        .save_settings(false)
        .begin(ui)
    {
        let scene = renderer.scene();
        let mut sky = scene.skydome();
        if imgui::Slider::new("Exposure", 1E-3, 1E3)
            .flags(SliderFlags::ALWAYS_CLAMP | SliderFlags::LOGARITHMIC)
            .build(ui, &mut exposure)
        {
            update_exposure = true;
        }
        if ui.button("Add") {
            let dflt_light = Light {
                ltype: LightType::OMNI,
                name: format!("Light{}", scene.lights().len()),
                ..Default::default()
            };
            update_lights = true;
            add = Some(dflt_light);
        }
        ui.separator();
        ui.spacing();
        // Skylight
        ui.text("Sky Dome");
        let dome_preview = match &sky {
            Some(sky) => {
                let texture = renderer
                    .scene()
                    .single_texture(sky.resource_id as u16)
                    .unwrap();
                &texture.info().name
            }
            _ => "",
        };
        ComboBox::new("Sky Texture")
            .preview_value(dome_preview)
            .build(ui, || {
                if Selectable::new("None").build(ui) {
                    sky = None;
                    update_lights = true;
                }
                renderer
                    .scene()
                    .textures()
                    .iter()
                    .enumerate()
                    .for_each(|(id, texture)| {
                        if Selectable::new(&texture.info().name).build(ui) {
                            match &mut sky {
                                Some(s) => s.resource_id = id as u32,
                                None => {
                                    sky = Some(Light {
                                        ltype: LightType::SKY,
                                        name: "Sky".to_string(),
                                        resource_id: id as u32,
                                        ..Default::default()
                                    });
                                }
                            }
                            update_lights = true;
                        }
                        if ui.is_item_hovered() {
                            ui.tooltip(|| {
                                Image::new(TextureId::new(id), [128.0, 128.0]).build(ui);
                            });
                        }
                    });
            });
        if let Some(sky) = &mut sky {
            ui.same_line();
            if ImageButton::new(TextureId::new(sky.resource_id as usize), [16.0, 16.0])
                .frame_padding(0)
                .build(ui)
            {
                state.textures_selected = Some(sky.resource_id as u16);
                state.textures_window = true;
            }
            if ui.is_item_hovered() {
                ui.tooltip(|| {
                    ui.text(scene.single_texture(sky.resource_id as u16).unwrap().name());
                    Image::new(TextureId::new(sky.resource_id as usize), [256.0, 256.0]).build(ui);
                });
            }
            if Slider::new("Dome Yaw (deg)", 0.0, 360.0).build(ui, &mut sky.yaw_deg)
                || Slider::new("Dome Pitch (deg)", 0.0, 360.0).build(ui, &mut sky.pitch_deg)
                || Slider::new("Dome Roll (deg)", 0.0, 360.0).build(ui, &mut sky.roll_deg)
            {
                update_lights = true;
            }
            if imgui::Slider::new("Intensity", 1E-2, 1E2)
                .flags(SliderFlags::ALWAYS_CLAMP | SliderFlags::LOGARITHMIC)
                .build(ui, &mut sky.intensity)
            {
                update_lights = true;
            }
        }
        ui.separator();
        ui.spacing();
        ui.text("Lights in the scene:");
        let header = [
            imgui::TableColumnSetup::new("Name"),
            imgui::TableColumnSetup::new("Type"),
        ];
        if let Some(table) =
            ui.begin_table_header_with_flags("lighttable", header, imgui::TableFlags::ROW_BG)
        {
            for (light_id, light) in scene.lights().iter().enumerate() {
                ui.table_next_row();
                ui.table_next_column();
                if imgui::Selectable::new(&light.name)
                    .span_all_columns(true)
                    .build(ui)
                {
                    state.light_selected = Some(light_id);
                }
                ui.table_next_column();
                ui.text(light.ltype.name());
                ui.table_next_column();
            }
            table.end();
        }
        if let Some(selected) = state.light_selected {
            // in some cases (area lights removed by the renderer) the selected is not None but
            // points to an invalid light. So i need this double check.
            if let Some(light) = &scene.lights().get(selected) {
                let mut updated_current = false;
                let mut new_name = light.name.to_string();
                let mut new_type = light.ltype;
                let mut new_color = light.color;
                let mut new_pos = light.position;
                let mut new_dir = light.direction;
                let mut new_intensity = light.intensity;
                ui.spacing();
                if imgui::InputText::new(ui, "Light Name", &mut new_name)
                    .enter_returns_true(true)
                    .build()
                {
                    updated_current = true;
                }
                // non-delta lights cannot be modified in type
                if new_type.is_delta() {
                    ComboBox::new("Light type")
                        .preview_value(new_type.name())
                        .build(ui, || {
                            // area light should be set using the emissive material and not the light UI
                            for light_type in
                                LightType::all().iter().filter(|x| x.is_delta()).copied()
                            {
                                if Selectable::new(light_type.name()).build(ui) {
                                    updated_current = true;
                                    new_type = light_type;
                                }
                            }
                        });
                }
                if new_type.has_spectrum() {
                    if state.spectrum_temperature {
                        let mut temperature = 1500.0;
                        if imgui::InputFloat::new(ui, "Temperature (K)", &mut temperature).build() {
                            updated_current = true;
                            new_color = Spectrum::from_blackbody(temperature);
                        }
                    } else {
                        let mut current: [f32; 3] = new_color.to_xyz().to_rgb().into();
                        if imgui::ColorPicker::new("Spectrum color", &mut current)
                            .small_preview(false)
                            .side_preview(false)
                            .build(ui)
                        {
                            updated_current = true;
                            new_color = Spectrum::from_rgb(ColorRGB::from(current), true);
                        }
                    }
                    ui.same_line();
                    if imgui::ColorButton::new(
                        "spectrum_current",
                        new_color.to_xyz().to_rgb().into(),
                    )
                    .tooltip(false)
                    .build(ui)
                    {
                        state.spectrum_temperature = !state.spectrum_temperature;
                    }
                }
                if new_type.has_position() {
                    if imgui::InputFloat3::new(ui, "Position", new_pos.as_mut()).build() {
                        updated_current = true;
                    }
                }
                if new_type.has_direction() {
                    if imgui::InputFloat3::new(ui, "Direction", new_dir.as_mut()).build() {
                        updated_current = true;
                    }
                }
                if new_type.has_intensity() {
                    if imgui::Slider::new("Intensity", 1E-2, 1E2)
                        .flags(SliderFlags::ALWAYS_CLAMP | SliderFlags::LOGARITHMIC)
                        .build(ui, &mut new_intensity)
                    {
                        updated_current = true;
                    }
                }
                // only delta lights can be removed by the user
                if new_type.is_delta() && ui.button("Remove") {
                    remove = Some(selected);
                    update_lights = true;
                } else if updated_current {
                    let new_light = Light {
                        ltype: new_type,
                        name: new_name,
                        position: new_pos,
                        direction: new_dir,
                        color: new_color,
                        intensity: new_intensity,
                        ..Default::default() /* remaining fields are for SKY which should not be
                                              * selectable */
                    };
                    update = Some((selected, new_light));
                    update_lights = true;
                }
            }
        }
        if update_exposure {
            renderer.set_exposure(exposure);
            if let Some(raytracer) = raytracer.as_mut() {
                raytracer.set_exposure(exposure);
            }
        }
        window.end();
        if update_lights {
            let mut lights = renderer.scene().lights().to_vec();
            let materials = renderer.scene().materials().to_vec();
            if let Some(new) = add {
                lights.push(new);
            } else if let Some((old, new)) = update {
                lights[old] = new;
            } else if let Some(old) = remove {
                lights.remove(old);
                state.light_selected = None;
            }
            // add/modify sky
            if let Some(last) = lights.last_mut() {
                if last.ltype == LightType::SKY {
                    //modify existing
                    if let Some(sky) = sky {
                        *last = sky;
                    } else {
                        // remove
                        lights.pop();
                    }
                } else {
                    // push new sky
                    if let Some(sky) = sky {
                        lights.push(sky);
                    }
                }
            } else if let Some(sky) = sky {
                lights.push(sky);
            }
            renderer.update_light(&lights);
            if let Some(raytracer) = raytracer.as_mut() {
                raytracer.update_materials_and_lights(
                    &materials,
                    &lights,
                    renderer.scene().textures(),
                );
            }
        }
    }
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

fn select_and_load_scene(state: &mut UiState) {
    let dialog_pick = FileDialog::new()
        .set_title("Open scene...")
        .add_filter("Glaze scene", &["glaze"])
        .pick_file();
    if let Some(path) = dialog_pick {
        match parse(path) {
            Ok(parsed) => {
                let scene_radius = parsed.meta().unwrap_or_default().scene_radius;
                let (wchan, rchan) = mpsc::channel();
                let instance_clone = Arc::clone(&state.instance);
                let thandle = std::thread::spawn(move || {
                    let instance = instance_clone;
                    wchan.send("Loading realtime scene...".to_string()).ok();
                    let scene = RealtimeScene::new(Arc::clone(&instance), parsed);
                    let rtscene = if instance.supports_raytrace() {
                        wchan.send("Loading raytraced scene...".to_string()).ok();
                        Some(RayTraceScene::<PresentInstance>::from(&scene))
                    } else {
                        None
                    };
                    (scene, rtscene)
                });
                state.loading_scene = Some(SceneLoad {
                    reader: rchan,
                    last_message: "".to_string(),
                    join_handle: thandle,
                });
                state.movement_lock = true;
                state.mov_speed = scene_radius / 100.0;
                state.vert_speed = scene_radius / 1000.0;
                state.textures_selected = None;
                state.materials_selected = None;
                state.light_selected = None;
                state.open_loading_popup = true;
            }
            Err(err) => log::error!("Failed to parse scene file: {}", err),
        }
    }
}
