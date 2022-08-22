#![allow(clippy::type_complexity)]
use cgmath::{
    Matrix, Matrix4, MetricSpace, Point2, Point3, SquareMatrix, Transform as CgmathTransform,
    Vector3 as Vec3,
};
use clap::Parser;
use console::style;
use glaze::{
    converted_file, parse, Camera, ColorRGB, Light, Material, Mesh, MeshInstance, Meta,
    ParserVersion, PerspectiveCam, Serializer, Spectrum, Texture, TextureFormat, TextureInfo,
    Transform, Vertex, DEFAULT_TEXTURE_ID,
};
use image::io::Reader as ImageReader;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use russimp::material::{MaterialProperty, PropertyTypeInfo};
use russimp::scene::{PostProcess, Scene as RussimpScene};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt::Write;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempdir;

struct TempScene {
    vertices: Vec<Vertex>,
    meshes: Vec<Mesh>,
    cameras: Vec<Camera>,
    textures: Vec<Texture>,
    materials: Vec<Material>,
    transforms: Vec<Transform>,
    instances: Vec<MeshInstance>,
    lights: Vec<Light>,
    meta: Meta,
}

macro_rules! error(
    ($msg: expr, $cause: expr) => {
        eprintln!("{}: {}. {}", style("Error").bold().red(), $msg, $cause)
    }
);

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input scene
    input: String,
    /// Converted scene
    #[clap(required_unless_present = "benchmark")]
    output: Option<String>,
    /// Perform a reading benchmark on the input scene
    #[clap(short, long)]
    benchmark: bool,
    /// Calculate and store the mip-maps inside the scene file
    #[clap(long = "gen-mipmaps")]
    gen_mipmaps: bool,
}

fn main() {
    let args = Args::parse();
    if !args.benchmark {
        let output = args.output.unwrap();
        println!("{} Preprocessing input...", style("[1/3]").bold().dim());
        match preprocess_input(&args.input) {
            Ok(scene) => {
                println!("{} Converting scene...", style("[2/3]").bold().dim());
                match convert_input(scene, &args.input, args.gen_mipmaps) {
                    Ok(scene) => {
                        println!("{} Compressing file...", style("[3/3]").bold().dim());
                        match write_output(scene, ParserVersion::V1, &output) {
                            Ok(_) => println!("{}", style("Done!").bold().green()),
                            Err(e) => error!("Failed to compress file", e),
                        }
                    }
                    Err(e) => error!("Failed to convert scene", e),
                }
            }
            Err(e) => error!("Failed to preprocess input", e),
        }
    } else if let Err(error) = benchmark(&args.input, ParserVersion::V1, args.gen_mipmaps) {
        error!("Failed to benchmark scene", error);
    };
}

fn preprocess_input<S: AsRef<str>>(input: S) -> Result<RussimpScene, Box<dyn Error>> {
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(Duration::from_millis(120));
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
    let vv: [f32; 3] = Point3::into(vert.vv);
    let vn: [f32; 3] = Vec3::into(vert.vn);
    let vt: [f32; 2] = Point2::into(vert.vt);
    vv.iter()
        .chain(vn.iter())
        .chain(vt.iter())
        .copied()
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn convert_input(
    scene: RussimpScene,
    original_path: &str,
    gen_mm: bool,
) -> Result<TempScene, std::io::Error> {
    let mpb = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .progress_chars("#>-")
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap();
    let camera_data = scene.cameras;
    let camera_pb = mpb.add(ProgressBar::new(1));
    camera_pb.set_style(style.clone());
    let light_data = scene.lights;
    let light_pb = mpb.add(ProgressBar::new(1));
    light_pb.set_style(style.clone());
    let mesh_data = scene.meshes;
    let mesh_pb = mpb.add(ProgressBar::new(1));
    mesh_pb.set_style(style.clone());
    let material_data = scene.materials;
    let mat_pb = mpb.add(ProgressBar::new(1));
    mat_pb.set_style(style.clone());
    let path = original_path.to_string();
    let lights_thread = thread::spawn(move || convert_lights(&light_data, light_pb));
    let material_thread = thread::spawn(move || convert_materials(&material_data, mat_pb, &path));
    let mesh_thread = thread::spawn(move || convert_meshes(&mesh_data, mesh_pb));
    let mut lights = lights_thread.join().unwrap();
    let (materials, mut textures, mut area_lights) = material_thread.join().unwrap()?;
    lights.append(&mut area_lights);
    if gen_mm {
        let mm_pb = mpb.add(ProgressBar::new(1));
        mm_pb.set_style(style);
        let tex_mm_thread = thread::spawn(move || gen_mipmaps(textures, mm_pb));
        textures = tex_mm_thread.join().unwrap();
    }
    let (vertices, meshes) = mesh_thread.join().unwrap()?;
    let (transforms, instances) = if let Some(root) = scene.root {
        convert_transforms_and_instances(&root)
    } else {
        // no scene structure, puts each mesh in the scene with an identity matrix.
        // this should never happen, even for simple files assimp should generate a single node.
        let instances = meshes
            .iter()
            .map(|m| MeshInstance {
                mesh_id: m.id,
                transform_id: 0,
            })
            .collect();
        (vec![Transform::identity()], instances)
    };
    let (centre, radius) = calc_scene_centre_radius(&vertices, &meshes, &instances, &transforms);
    let camera_thread = thread::spawn(move || convert_cameras(&camera_data, camera_pb, radius));
    let cameras = camera_thread.join().unwrap();
    let meta = Meta {
        scene_radius: radius,
        exposure: 1.0,
        scene_centre: centre.into(),
    };
    mpb.clear().ok();
    Ok(TempScene {
        vertices,
        meshes,
        cameras,
        textures,
        materials,
        transforms,
        instances,
        lights,
        meta,
    })
}

fn calc_scene_centre_radius(
    verts: &[Vertex],
    meshes: &[Mesh],
    instns: &[MeshInstance],
    trans: &[Transform],
) -> (Point3<f32>, f32) {
    let mut pmin = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut pmax = Point3::new(f32::MIN, f32::MIN, f32::MIN);
    for instance in instns {
        let mesh = &meshes[instance.mesh_id as usize];
        let tran = &trans[instance.transform_id as usize];
        for index in &mesh.indices {
            let vert = &verts[*index as usize];
            let vert_world = tran.inner().transform_point(vert.vv);
            pmin.x = f32::min(pmin.x, vert_world.x);
            pmin.y = f32::min(pmin.y, vert_world.y);
            pmin.z = f32::min(pmin.z, vert_world.z);
            pmax.x = f32::max(pmax.x, vert_world.x);
            pmax.y = f32::max(pmax.y, vert_world.y);
            pmax.z = f32::max(pmax.z, vert_world.z);
        }
    }
    (pmin + (pmax - pmin) * 0.5, pmax.distance(pmin) / 2.0)
}

fn russimp_to_cgmath_matrix(mat: russimp::Matrix4x4) -> Matrix4<f32> {
    // russimp uses assimp Matrix4 (row-major) and cgmath expects a column_major
    // so the matrix have to be transposed
    let matrix = Matrix4::new(
        mat.a1, mat.a2, mat.a3, mat.a4, mat.b1, mat.b2, mat.b3, mat.b4, mat.c1, mat.c2, mat.c3,
        mat.c4, mat.d1, mat.d2, mat.d3, mat.d4,
    );
    matrix.transpose()
}

fn convert_transforms_and_instances(
    root: &RefCell<russimp::node::Node>,
) -> (Vec<Transform>, Vec<MeshInstance>) {
    let mut transforms = HashMap::new();
    //insert the identity transform with index 0
    transforms.insert(Transform::identity().to_bytes(), 0);
    let mut instances = Vec::new();
    conv_trans_inst_rec(root, Matrix4::identity(), &mut transforms, &mut instances);
    // sort transforms
    let sorted_transforms = transforms
        .into_iter()
        .map(|(trans, id)| (id, Transform::from_bytes(trans)))
        .collect::<BTreeMap<_, _>>();
    (sorted_transforms.into_values().collect(), instances)
}

fn conv_trans_inst_rec(
    root: &RefCell<russimp::node::Node>,
    cur_trans: Matrix4<f32>,
    transforms: &mut HashMap<[u8; 64], u16>,
    instances: &mut Vec<MeshInstance>,
) {
    let node = root.borrow();
    let cur_trans = russimp_to_cgmath_matrix(node.transformation) * cur_trans;
    if !node.meshes.is_empty() {
        let next_id = transforms.len() as u16;
        // this will probably never report duplicates due to how fp values works...
        let tid = transforms
            .entry(Transform::from(cur_trans).to_bytes())
            .or_insert(next_id);
        for mesh in &node.meshes {
            instances.push(MeshInstance {
                mesh_id: *mesh as u16,
                transform_id: *tid,
            });
        }
    }
    for child in &node.children {
        conv_trans_inst_rec(child, cur_trans, transforms, instances);
    }
}

fn gen_mipmaps(mut textures: Vec<Texture>, pb: ProgressBar) -> Vec<Texture> {
    let effort = textures.len();
    pb.set_length(effort as u64);
    pb.set_message("Generating mipmaps");
    for texture in textures.iter_mut() {
        texture.gen_mipmaps();
        pb.inc(1);
    }
    textures
}

fn convert_lights(lights: &[russimp::light::Light], pb: ProgressBar) -> Vec<Light> {
    let effort = lights.len();
    pb.set_length(effort as u64);
    pb.set_message("Converting lights");
    let mut retval_lights = Vec::with_capacity(effort);
    for light in lights {
        let color = ColorRGB::new(
            light.color_diffuse.r,
            light.color_diffuse.g,
            light.color_diffuse.b,
        );
        let spectrum = Spectrum::from_rgb(color, true);
        let mut new_light = Light {
            ltype: glaze::LightType::OMNI,
            name: light.name.clone(),
            color: spectrum,
            position: russimp_vec_to_point(light.pos),
            direction: russimp_vec_to_vec(light.direction),
            intensity: light.attenuation_linear,
            ..Default::default()
        };
        match light.light_source_type {
            russimp::light::LightSourceType::Point => new_light.ltype = glaze::LightType::OMNI,
            russimp::light::LightSourceType::Directional => new_light.ltype = glaze::LightType::SUN,
            _ => panic!("Unsupported light type"),
        };
        retval_lights.push(new_light);
        pb.inc(1);
    }
    pb.finish();
    retval_lights
}

fn russimp_vec_to_vec(vec: russimp::Vector3D) -> Vec3<f32> {
    Vec3::new(vec.x, vec.y, vec.z)
}

fn russimp_vec_to_point(vec: russimp::Vector3D) -> Point3<f32> {
    Point3::new(vec.x, vec.y, vec.z)
}

fn convert_meshes(
    meshes: &[russimp::mesh::Mesh],
    pb: ProgressBar,
) -> Result<(Vec<Vertex>, Vec<Mesh>), std::io::Error> {
    const DEFAULT_TEXCOORD: [Point2<f32>; 3] = [
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
    ];
    let mut retval_vertices = Vec::new();
    let mut retval_meshes = Vec::new();
    let mut used_vert = HashMap::new();
    let effort = meshes.iter().fold(0, |acc, m| acc + m.faces.len()) as u64;
    pb.set_length(effort);
    pb.set_message("Converting meshes");
    for (id, mesh) in meshes.iter().enumerate() {
        let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
        let use_default_uvs = mesh.uv_components[0] < 2;
        for face in &mesh.faces {
            if face.0.len() != 3 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Only triangles are supported",
                ));
            }
            for (i, index) in face.0.iter().enumerate() {
                let vv = mesh.vertices[*index as usize];
                let vn = mesh.normals[*index as usize];
                let vt = if use_default_uvs {
                    DEFAULT_TEXCOORD[i]
                } else {
                    let tmp = mesh.texture_coords[0].as_ref().unwrap()[*index as usize];
                    Point2::new(tmp.x, tmp.y)
                };
                let vertex = Vertex {
                    vv: Point3::new(vv.x, vv.y, vv.z),
                    vn: Vec3::new(vn.x, vn.y, vn.z),
                    vt: Point2::new(vt.x, 1.0 - vt.y), // flip y for vulkan
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
            id: id as u16,
            indices,
            material: mesh.material_index as u16 + 1, // +1 because 0 is the default material
        };
        retval_meshes.push(mesh);
    }
    pb.finish();
    Ok((retval_vertices, retval_meshes))
}

fn convert_cameras(cams: &[russimp::camera::Camera], pb: ProgressBar, radius: f32) -> Vec<Camera> {
    let effort = std::cmp::max(cams.len(), 1);
    pb.set_length(effort as u64);
    pb.set_message("Converting cameras");
    let mut retval_cameras = Vec::with_capacity(effort);
    for camera in cams {
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
            near: f32::max(1E-3, radius * 2.0 * 1E-5),
            far: f32::max(100.0, radius * 2.0),
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
) -> Result<(Vec<Material>, Vec<Texture>, Vec<Light>), std::io::Error> {
    let effort = materials.len();
    pb.set_length(effort as u64);
    pb.set_message("Converting materials and textures");
    let mut used_textures = HashMap::new();
    let mut retval_textures = Vec::new();
    let mut retval_materials = Vec::new();
    let mut retval_lights = Vec::new();
    // add default texture
    retval_textures.push(Texture::default());
    retval_materials.push(Material::default());
    for material in materials {
        for (texture_type, textures) in &material.textures {
            let texture = textures.first().unwrap(); // support single textures only
                                                     // replace \ with / and hopes UNIX path do not use strange names
                                                     //TODO: add support for embedded textures
            let mut path = PathBuf::from(texture.path.clone().replace('\\', "/"));
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
                        TextureFormat::RgbaSrgb,
                        &mut retval_textures,
                    )?;
                }
                russimp::texture::TextureType::Normals => {
                    convert_texture(
                        tex_name,
                        path,
                        &mut used_textures,
                        TextureFormat::RgbaNorm,
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
        let (material, light) = convert_material(
            &material.properties,
            &used_textures,
            retval_materials.len() as u16,
        );
        retval_materials.push(material);
        if let Some(light) = light {
            retval_lights.push(light);
        }
        pb.inc(1);
    }
    pb.finish();
    Ok((retval_materials, retval_textures, retval_lights))
}

fn convert_texture(
    name: &str,
    path: PathBuf,
    used: &mut HashMap<String, u16>,
    format: TextureFormat,
    ret: &mut Vec<Texture>,
) -> Result<(), std::io::Error> {
    let used_name = used_name(name, format);
    #[allow(clippy::map_entry)] // this block is very long, I don't want to have a single `match`
    if !used.contains_key(&used_name) {
        let data = ImageReader::open(path)?.decode();
        if let Ok(data) = data {
            let mut info = TextureInfo {
                // this is the displayed name, I want to retain this the same as the one on the file
                name: name.to_string(),
                width: data.width() as u16,
                height: data.height() as u16,
                format: TextureFormat::RgbaSrgb,
            };
            let texture = match format {
                TextureFormat::Gray => {
                    let img_raw = data.to_luma8();
                    info.format = TextureFormat::Gray;
                    Texture::new_gray(info, img_raw)
                }
                TextureFormat::RgbaSrgb => {
                    let img_raw = data.to_rgba8();
                    info.format = TextureFormat::RgbaSrgb;
                    Texture::new_rgba(info, img_raw)
                }
                TextureFormat::RgbaNorm => {
                    let img_raw = data.to_rgba8();
                    info.format = TextureFormat::RgbaNorm;
                    Texture::new_rgba(info, img_raw)
                }
            };
            // this works as long as the default ID is 0
            let id = ret.len() as u16;
            ret.push(texture);
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

fn convert_material(
    props: &[MaterialProperty],
    used_textures: &HashMap<String, u16>,
    mat_id: u16,
) -> (Material, Option<Light>) {
    let mut retval = Material::default();
    for property in props {
        match property.key.as_str() {
            "?mat.name" => retval.name = matprop_to_str(property),
            "$clr.diffuse" => retval.diffuse_mul = fcol_to_ucol(matprop_to_fvec(property)),
            "$clr.emissive" => {
                let color = fcol_to_ucol(matprop_to_fvec(property));
                if color[0] > 0 || color[1] > 0 || color[2] > 0 {
                    retval.emissive_col = Some(color);
                }
            }
            "$tex.file" => {
                let prop_name = matprop_to_str(property);
                let format = match property.semantic {
                    russimp::texture::TextureType::Diffuse => TextureFormat::RgbaSrgb,
                    russimp::texture::TextureType::Normals => TextureFormat::RgbaNorm,
                    russimp::texture::TextureType::Opacity => TextureFormat::Gray,
                    _ => TextureFormat::RgbaSrgb,
                };
                let tex_name = used_name(&prop_name, format);
                let texture = if let Some(texture_id) = used_textures.get(&tex_name) {
                    *texture_id
                } else {
                    DEFAULT_TEXTURE_ID
                };
                match property.semantic {
                    russimp::texture::TextureType::Diffuse => retval.diffuse = texture,
                    russimp::texture::TextureType::Normals => retval.normal = texture,
                    russimp::texture::TextureType::Opacity => retval.opacity = texture,
                    _ => {}
                }
            }
            _ => {} // super ugly...
        }
    }
    if retval.emissive_col.is_some() {
        let light = Light {
            ltype: glaze::LightType::AREA,
            name: retval.name.clone(),
            resource_id: mat_id as u32,
            ..Default::default()
        };
        (retval, Some(light))
    } else {
        (retval, None)
    }
}

fn used_name(name: &str, format: TextureFormat) -> String {
    // assign the format as part of the name. The same texture will be forcibly duplicated when
    // used with different formats
    let format_str = match format {
        TextureFormat::Gray => "(R)",
        TextureFormat::RgbaSrgb => "(sRGBA)",
        TextureFormat::RgbaNorm => "(lRGBA)",
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

fn matprop_to_fvec(property: &MaterialProperty) -> [f32; 3] {
    match &property.data {
        PropertyTypeInfo::FloatArray(a) => [a[0], a[1], a[2]],
        _ => [0.0, 0.0, 0.0],
    }
}

fn write_output(
    scene: TempScene,
    version: ParserVersion,
    output: &str,
) -> Result<(), std::io::Error> {
    Serializer::new(output, version)
        .with_vertices(&scene.vertices)
        .with_meshes(&scene.meshes)
        .with_transforms(&scene.transforms)
        .with_instances(&scene.instances)
        .with_cameras(&scene.cameras)
        .with_textures(&scene.textures)
        .with_materials(&scene.materials)
        .with_lights(&scene.lights)
        .with_metadata(&scene.meta)
        .serialize()
}

fn benchmark(
    input: &str,
    version: ParserVersion,
    gen_mm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
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
        let conversion = convert_input(preprocessed, input, gen_mm)?;
        let conversion_end = Instant::now();
        write_output(conversion, version, file.to_str().unwrap())?;
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
    let parsed = parse(&file)?;
    let vert_start = Instant::now();
    let vertices = parsed.vertices()?;
    let vert_end = Instant::now();
    let meshes = parsed.meshes()?;
    let mesh_end = Instant::now();
    let textures = parsed.textures()?;
    let texture_end = Instant::now();
    let materials = parsed.materials()?;
    let material_end = Instant::now();
    //  Results //
    println!("Reading and writing results for {}", input);
    println!("Total vertices: {}", vertices.len());
    println!("Total meshes: {}", meshes.len());
    println!("Total textures: {}", textures.len());
    println!("Total materials: {}", materials.len());
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
    use glaze::{parse, Serializer};
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
        let scene = super::convert_input(scene, path.to_str().unwrap(), false).unwrap();
        let dir = tempdir()?;
        let file = dir.path().join("serializer.bin");
        assert!(
            Serializer::new(file.to_str().unwrap(), glaze::ParserVersion::V1)
                .with_vertices(&scene.vertices)
                .with_meshes(&scene.meshes)
                .with_transforms(&scene.transforms)
                .with_instances(&scene.instances)
                .with_cameras(&scene.cameras)
                .with_textures(&scene.textures)
                .with_materials(&scene.materials)
                .serialize()
                .is_ok()
        );
        let parsed = parse(&file);
        assert!(parsed.is_ok());
        if let Ok(parsed) = parsed {
            assert_eq!(parsed.meshes()?.len(), 1);
            assert_eq!(parsed.transforms()?.len(), 1);
            assert_eq!(parsed.instances()?.len(), 1);
            assert_eq!(parsed.cameras()?.len(), 1);
            assert_eq!(parsed.materials()?.len(), 3);
            assert_eq!(parsed.textures()?.len(), 2);
            assert_eq!(parsed.vertices()?.len(), 24);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }

    #[test]
    fn test_mipmap_generation() -> Result<(), Box<dyn Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("cube.obj");
        let scene = super::preprocess_input(path.to_str().unwrap()).unwrap();
        let scene = super::convert_input(scene, path.to_str().unwrap(), true).unwrap();
        let dir = tempdir()?;
        let file = dir.path().join("serializer_with_mipmap.bin");
        assert!(
            Serializer::new(file.to_str().unwrap(), glaze::ParserVersion::V1)
                .with_textures(&scene.textures)
                .serialize()
                .is_ok()
        );
        let parsed = parse(&file);
        assert!(parsed.is_ok());
        if let Ok(parsed) = parsed {
            let textures = parsed.textures()?;
            assert!(textures[1].has_mipmaps());
            assert_eq!(textures[1].mipmap_levels(), 10);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }

    #[test]
    fn test_mipmap_skip() -> Result<(), Box<dyn Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("cube.obj");
        let scene = super::preprocess_input(path.to_str().unwrap()).unwrap();
        let scene = super::convert_input(scene, path.to_str().unwrap(), false).unwrap();
        let dir = tempdir()?;
        let file = dir.path().join("serializer_no_mipmap.bin");
        assert!(
            Serializer::new(file.to_str().unwrap(), glaze::ParserVersion::V1)
                .with_textures(&scene.textures)
                .serialize()
                .is_ok()
        );
        let parsed = parse(&file);
        assert!(parsed.is_ok());
        if let Ok(parsed) = parsed {
            let textures = parsed.textures()?;
            assert!(!textures[1].has_mipmaps());
            assert_eq!(textures[1].mipmap_levels(), 1);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }

    #[test]
    fn test_mesh_instances() -> Result<(), Box<dyn Error>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("test.fbx");
        let scene = super::preprocess_input(path.to_str().unwrap()).unwrap();
        let scene = super::convert_input(scene, path.to_str().unwrap(), false).unwrap();
        let dir = tempdir()?;
        let file = dir.path().join("instances.bin");
        assert!(
            Serializer::new(file.to_str().unwrap(), glaze::ParserVersion::V1)
                .with_vertices(&scene.vertices)
                .with_meshes(&scene.meshes)
                .with_transforms(&scene.transforms)
                .with_instances(&scene.instances)
                .with_cameras(&scene.cameras)
                .with_textures(&scene.textures)
                .with_materials(&scene.materials)
                .serialize()
                .is_ok()
        );
        let parsed = parse(&file);
        assert!(parsed.is_ok());
        if let Ok(parsed) = parsed {
            assert_eq!(parsed.meshes()?.len(), 1);
            assert_eq!(parsed.instances()?.len(), 5);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }
}
