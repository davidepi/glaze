#![allow(clippy::type_complexity)]
use cgmath::{Point3, Vector2 as Vec2, Vector3 as Vec3};
use clap::{App, Arg};
use console::style;
use glaze::{
    converted_file, parse, serialize, Camera, Material, Mesh, ParserVersion, PerspectiveCam,
    Texture, TextureFormat, TextureInfo, Vertex,
};
use image::io::Reader as ImageReader;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use russimp::material::{MaterialProperty, PropertyTypeInfo};
use russimp::scene::{PostProcess, Scene as RussimpScene};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::thread;
use std::time::Instant;
use tempfile::tempdir;

struct TempScene {
    vertices: Vec<Vertex>,
    meshes: Vec<Mesh>,
    cameras: Vec<Camera>,
    textures: Vec<(u16, Texture)>,
    materials: Vec<(u16, Material)>,
}

macro_rules! error(
    ($msg: expr, $cause: expr) => {
        eprintln!("{}: {}. {}", style("Error").bold().red(), $msg, $cause)
    }
);

fn main() {
    let supported_versions = [ParserVersion::V1]
        .iter()
        .map(ParserVersion::to_str)
        .collect::<Vec<_>>();
    let matches = App::new("glaze-converter")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about("Convert a 3D scene to a format recognizable by glaze")
        .arg(
            Arg::with_name("input")
                .required(true)
                .help("The input file"),
        )
        .arg(
            Arg::with_name("output")
                .required_unless("benchmark")
                .help("The output file"),
        )
        .arg(
            Arg::with_name("benchmark")
                .long("bench")
                .help("Run a benchmark for load and save times of a particular scene"),
        )
        .arg(
            Arg::with_name("file version")
                .short("V")
                .long("file-version")
                .default_value(ParserVersion::V1.to_str())
                .possible_values(&supported_versions),
        )
        .get_matches();
    let input = matches.value_of("input").unwrap();
    let version = ParserVersion::from_str(matches.value_of("file version").unwrap()).unwrap();
    if !matches.is_present("benchmark") {
        let output = matches.value_of("output").unwrap();
        println!("{} Preprocessing input...", style("[1/3]").bold().dim());
        match preprocess_input(input) {
            Ok(scene) => {
                println!("{} Converting scene...", style("[2/3]").bold().dim());
                match convert_input(scene, input) {
                    Ok(scene) => {
                        println!("{} Compressing file...", style("[3/3]").bold().dim());
                        match write_output(scene, version, output) {
                            Ok(_) => println!("{}", style("Done!").bold().green()),
                            Err(e) => error!("Failed to compress file", e),
                        }
                    }
                    Err(e) => error!("Failed to convert scene", e),
                }
            }
            Err(e) => error!("Failed to preprocess input", e),
        }
    }
}

fn preprocess_input<S: AsRef<str>>(input: S) -> Result<RussimpScene, Box<dyn Error>> {
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(120);
    let postprocess = vec![
        PostProcess::Triangulate,
        PostProcess::ValidateDataStructure,
        PostProcess::JoinIdenticalVertices,
        PostProcess::GenerateNormals,
        PostProcess::OptimizeMeshes,
        PostProcess::OptimizeGraph,
        PostProcess::FindInstances,
        PostProcess::FixInfacingNormals,
    ];
    let scene = RussimpScene::from_file(input.as_ref(), postprocess)?;
    pb.finish_and_clear();
    Ok(scene)
}

fn vertex_to_bytes(vert: &Vertex) -> Vec<u8> {
    let vv: [f32; 3] = Vec3::into(vert.vv);
    let vn: [f32; 3] = Vec3::into(vert.vn);
    let vt: [f32; 2] = Vec2::into(vert.vt);
    vv.iter()
        .chain(vn.iter())
        .chain(vt.iter())
        .copied()
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn convert_input(scene: RussimpScene, original_path: &str) -> Result<TempScene, std::io::Error> {
    let mpb = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .progress_chars("#>-");
    let camera_data = scene.cameras;
    let camera_pb = mpb.add(ProgressBar::new(1));
    camera_pb.set_style(style.clone());
    let mesh_data = scene.meshes;
    let mesh_pb = mpb.add(ProgressBar::new(1));
    mesh_pb.set_style(style.clone());
    let material_data = scene.materials;
    let mat_pb = mpb.add(ProgressBar::new(1));
    mat_pb.set_style(style.clone());
    let path = original_path.to_string();
    let camera_thread = thread::spawn(move || convert_cameras(&camera_data, camera_pb));
    let material_thread = thread::spawn(move || convert_materials(&material_data, mat_pb, &path));
    let mesh_thread = thread::spawn(move || convert_meshes(&mesh_data, mesh_pb));
    let cameras = camera_thread.join().unwrap();
    let (materials, textures) = material_thread.join().unwrap()?;
    let mm_pb = mpb.add(ProgressBar::new(1));
    mm_pb.set_style(style);
    let tex_mm_thread = thread::spawn(move || gen_mipmaps(textures, mm_pb));
    let (vertices, meshes) = mesh_thread.join().unwrap()?;
    let textures = tex_mm_thread.join().unwrap();
    mpb.clear().ok();
    Ok(TempScene {
        vertices,
        meshes,
        cameras,
        textures,
        materials,
    })
}

fn gen_mipmaps(mut textures: Vec<(u16, Texture)>, pb: ProgressBar) -> Vec<(u16, Texture)> {
    let effort = textures.len();
    pb.set_length(effort as u64);
    pb.set_message("Generating mipmaps");
    for (_, texture) in textures.iter_mut() {
        texture.gen_mipmaps();
        pb.inc(1);
    }
    textures
}

fn convert_meshes(
    meshes: &[russimp::mesh::Mesh],
    pb: ProgressBar,
) -> Result<(Vec<Vertex>, Vec<Mesh>), std::io::Error> {
    let mut retval_vertices = Vec::new();
    let mut retval_meshes = Vec::new();
    let mut used_vert = HashMap::new();
    let effort = meshes.iter().fold(0, |acc, m| acc + m.faces.len()) as u64;
    pb.set_length(effort);
    pb.set_message("Converting meshes");
    for mesh in meshes {
        let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
        if mesh.uv_components[0] < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported UV components in mesh",
            ));
        }
        for face in &mesh.faces {
            if face.0.len() != 3 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Only triangles are supported",
                ));
            }
            for index in &face.0 {
                let vv = mesh.vertices[*index as usize];
                let vn = mesh.normals[*index as usize];
                let vt = mesh.texture_coords[0].as_ref().unwrap()[*index as usize];
                let vertex = Vertex {
                    vv: Vec3::new(vv.x, vv.y, vv.z),
                    vn: Vec3::new(vn.x, vn.y, vn.z),
                    vt: Vec2::new(vt.x, 1.0 - vt.y), // flip y for vulkan
                };
                let vertex_bytes = vertex_to_bytes(&vertex);
                let vertex_index = if let Some(v) = used_vert.get(&vertex_bytes) {
                    *v
                } else {
                    let index = retval_vertices.len() as u32;
                    used_vert.insert(vertex_bytes, index);
                    retval_vertices.push(vertex);
                    index
                };
                indices.push(vertex_index);
            }
            pb.inc(1);
        }
        let mesh = Mesh {
            indices,
            material: mesh.material_index as u16,
            instances: Vec::new(),
        };
        retval_meshes.push(mesh);
    }
    pb.finish();
    Ok((retval_vertices, retval_meshes))
}

fn convert_cameras(cameras: &[russimp::camera::Camera], pb: ProgressBar) -> Vec<Camera> {
    let effort = std::cmp::max(cameras.len(), 1);
    pb.set_length(effort as u64);
    pb.set_message("Converting cameras");
    let mut retval_cameras = Vec::with_capacity(effort);
    for camera in cameras {
        retval_cameras.push(Camera::Perspective(PerspectiveCam {
            position: Point3::new(camera.position.x, camera.position.y, camera.position.z),
            target: Point3::new(camera.look_at.x, camera.look_at.y, camera.look_at.z),
            up: Vec3::new(camera.up.x, camera.up.y, camera.up.z),
            fovx: camera.horizontal_fov,
            near: camera.clip_plane_near,
            far: camera.clip_plane_far,
        }));
        pb.inc(1);
    }
    if retval_cameras.is_empty() {
        retval_cameras.push(Camera::Perspective(PerspectiveCam {
            position: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 100.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fovx: f32::to_radians(90.0),
            near: 0.1,
            far: 250.0,
        }));
        pb.inc(1);
    }
    pb.finish();
    retval_cameras
}

fn convert_materials(
    materials: &[russimp::material::Material],
    pb: ProgressBar,
    original_path: &str,
) -> Result<(Vec<(u16, Material)>, Vec<(u16, Texture)>), std::io::Error> {
    let effort = materials.len();
    pb.set_length(effort as u64);
    pb.set_message("Converting materials and textures");
    let mut used_textures = HashMap::new();
    let mut retval_textures = Vec::new();
    let mut retval_materials = Vec::new();
    for material in materials {
        for (texture_type, textures) in &material.textures {
            let texture = textures.first().unwrap(); // support single textures only
                                                     // replace \ with / and hopes UNIX path do not use strange names
                                                     //TODO: add support for embedded textures
            let mut path = PathBuf::from(texture.path.clone().replace("\\", "/"));
            let tex_name = texture.path.as_ref();
            if path.is_relative() {
                path = PathBuf::from(original_path).parent().unwrap().join(path);
            }
            match texture_type {
                russimp::texture::TextureType::Diffuse => {
                    convert_texture(
                        tex_name,
                        path,
                        &mut used_textures,
                        TextureFormat::Rgba,
                        &mut retval_textures,
                    )?;
                }
                russimp::texture::TextureType::Opacity => {
                    convert_texture(
                        tex_name,
                        path,
                        &mut used_textures,
                        TextureFormat::Gray,
                        &mut retval_textures,
                    )?;
                }
                _ => {} //unsupported, do nothing
            }
        }
        // build material
        let material = convert_material(&material.properties, &used_textures);
        let mat_id = retval_materials.len() as u16;
        retval_materials.push((mat_id, material));
        pb.inc(1);
    }
    pb.finish();
    Ok((retval_materials, retval_textures))
}

fn convert_texture(
    name: &str,
    path: PathBuf,
    used: &mut HashMap<String, u16>,
    format: TextureFormat,
    ret: &mut Vec<(u16, Texture)>,
) -> Result<(), std::io::Error> {
    let used_name = used_name(name, format);
    if !used.contains_key(&used_name) {
        let data = ImageReader::open(path)?.decode();
        if let Ok(data) = data {
            let texture = match format {
                TextureFormat::Gray => {
                    let img_raw = data.to_luma8();
                    let info = TextureInfo {
                        // this is the displayed name, I want to retain this the same as the one on the file
                        name: name.to_string(),
                        width: img_raw.width() as u16,
                        height: img_raw.height() as u16,
                        format: TextureFormat::Gray,
                    };
                    Texture::new_gray(info, img_raw)
                }
                TextureFormat::Rgba => {
                    let img_raw = data.to_rgba8();
                    let info = TextureInfo {
                        // this is the displayed name, I want to retain this the same as the one on the file
                        name: name.to_string(),
                        width: img_raw.width() as u16,
                        height: img_raw.height() as u16,
                        format: TextureFormat::Rgba,
                    };
                    Texture::new_rgba(info, img_raw)
                }
            };
            let id = ret.len() as u16;
            ret.push((id, texture));
            used.insert(used_name, id);
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Could not read texture format",
            ))
        }
    } else {
        Ok(())
    }
}

fn convert_material(props: &[MaterialProperty], used_textures: &HashMap<String, u16>) -> Material {
    let mut retval = Material::default();
    for property in props {
        match property.key.as_str() {
            "?mat.name" => retval.name = matprop_to_str(property),
            "$clr.diffuse" => retval.diffuse_mul = fcol_to_ucol(matprop_to_fvec(property)),
            "$tex.file" => {
                let prop_name = matprop_to_str(property);
                let format = match property.semantic {
                    russimp::texture::TextureType::Diffuse => TextureFormat::Rgba,
                    russimp::texture::TextureType::Opacity => TextureFormat::Gray,
                    _ => TextureFormat::Rgba,
                };
                let tex_name = used_name(&prop_name, format);
                let texture = used_textures.get(&tex_name);
                match property.semantic {
                    russimp::texture::TextureType::Diffuse => retval.diffuse = texture.copied(),
                    russimp::texture::TextureType::Opacity => retval.opacity = texture.copied(),
                    _ => {}
                }
            }
            _ => {} // super ugly...
        }
    }
    retval
}

fn used_name(name: &str, format: TextureFormat) -> String {
    // assign the format as part of the name. The same texture will be forcibly duplicated when
    // used with different formats
    let format_str = match format {
        TextureFormat::Gray => "(R)",
        TextureFormat::Rgba => "(RGBA)",
    };
    format!("{}{}", name, format_str)
}

fn fcol_to_ucol(col: [f32; 3]) -> [u8; 3] {
    [
        (col[0] * 255.0) as u8,
        (col[1] * 255.0) as u8,
        (col[2] * 255.0) as u8,
    ]
}

fn matprop_to_str(property: &MaterialProperty) -> String {
    match &property.data {
        PropertyTypeInfo::String(s) => s.clone(),
        _ => "".to_string(),
    }
}

fn matprop_to_ivec(property: &MaterialProperty) -> [i32; 3] {
    match &property.data {
        PropertyTypeInfo::IntegerArray(a) => [a[0], a[1], a[2]],
        _ => [0, 0, 0],
    }
}

fn matprop_to_fvec(property: &MaterialProperty) -> [f32; 3] {
    match &property.data {
        PropertyTypeInfo::FloatArray(a) => [a[0], a[1], a[2]],
        _ => [0.0, 0.0, 0.0],
    }
}

fn write_output<P: AsRef<Path>>(
    scene: TempScene,
    version: ParserVersion,
    output: P,
) -> Result<(), Box<dyn Error>> {
    Ok(serialize(
        output,
        version,
        &scene.vertices,
        &scene.meshes,
        &scene.cameras,
        &scene.textures,
        &scene.materials,
    )?)
}

pub fn benchmark(input: &str, version: ParserVersion) -> Result<(), Box<dyn std::error::Error>> {
    // the benchmark is simple on purpose. This method will take seconds if not minutes to run.
    let dir = tempdir()?;
    let file;
    let conversion_time;
    if !converted_file(input) {
        let mut conv_results = String::new();
        file = dir.path().join("benchmark.bin");
        let preprocess_start = Instant::now();
        let preprocessed = preprocess_input(input)?;
        let preprocess_end = Instant::now();
        let conversion = convert_input(preprocessed, input)?;
        let conversion_end = Instant::now();
        let _ = write_output(conversion, version, &file)?;
        let compression_end = Instant::now();
        writeln!(&mut conv_results, "--- Writing ---")?;
        writeln!(
            &mut conv_results,
            "Preprocessing: {}s",
            (preprocess_end - preprocess_start).as_secs_f32()
        )?;
        writeln!(
            &mut conv_results,
            "Conversion: {}s",
            (conversion_end - preprocess_end).as_secs_f32()
        )?;
        writeln!(
            &mut conv_results,
            "Compressing + Writing: {}s",
            (compression_end - conversion_end).as_secs_f32()
        )?;
        conversion_time = Some(conv_results);
    } else {
        conversion_time = None;
        file = PathBuf::from(input);
    }
    let mut parsed = parse(&file)?;
    let vert_start = Instant::now();
    let vertices = parsed.vertices()?;
    let vert_end = Instant::now();
    let _ = parsed.meshes()?;
    let mesh_end = Instant::now();
    let textures = parsed.textures()?;
    let texture_end = Instant::now();
    let _ = parsed.materials()?;
    let material_end = Instant::now();
    //  Results //
    println!("Reading and writing results for {}", input);
    println!("Total vertices: {}", vertices.len());
    println!("Total textures: {}", textures.len());
    if let Some(results) = conversion_time {
        println!("{}", results);
    }
    println!("--- Reading ---");
    println!("Vertices: {}s", (vert_end - vert_start).as_secs_f32());
    println!("Meshes: {}s", (mesh_end - vert_end).as_secs_f32());
    println!("Textures: {}s", (texture_end - mesh_end).as_secs_f32());
    println!("Materials: {}s", (material_end - texture_end).as_secs_f32());
    // would be nice to add also the compression factor...
    Ok(())
}

#[cfg(test)]
mod tests {
    use glaze::{parse, serialize};
    use std::error::Error;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_working_conversion() -> Result<(), Box<dyn Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("cube.obj");
        let scene = super::preprocess_input(path.to_str().unwrap()).unwrap();
        let scene = super::convert_input(scene, path.to_str().unwrap()).unwrap();
        let dir = tempdir()?;
        let file = dir.path().join("serializer.bin");
        assert!(serialize(
            &file,
            glaze::ParserVersion::V1,
            &scene.vertices,
            &scene.meshes,
            &scene.cameras,
            &scene.textures,
            &scene.materials
        )
        .is_ok());
        let parsed = parse(&file);
        assert!(parsed.is_ok());
        if let Ok(mut parsed) = parsed {
            assert_eq!(parsed.meshes()?.len(), 1);
            assert_eq!(parsed.cameras()?.len(), 1);
            assert_eq!(parsed.materials()?.len(), 2);
            let textures = parsed.textures()?;
            assert_eq!(textures[0].1.mipmap_levels(), 10);
            assert_eq!(parsed.vertices()?.len(), 24);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }
}
