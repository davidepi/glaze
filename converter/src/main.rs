use cgmath::{Point3, Vector2 as Vec2, Vector3 as Vec3};
use clap::{App, Arg};
use console::style;
use glaze::{
    converted_file, parse, serialize, Camera, Material, Mesh, ParserVersion, PerspectiveCam,
    ShaderMat, Texture, TextureFormat, TextureInfo, Vertex,
};
use image::io::Reader as ImageReader;
use indicatif::{MultiProgress, ProgressBar};
use russimp::scene::{PostProcess, Scene as RussimpScene};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;
use tempfile::tempdir;

struct TempScene {
    vertices: Vec<Vertex>,
    meshes: Vec<Mesh>,
    cameras: Vec<Camera>,
    textures: Vec<(u16, Texture)>,
    materials: Vec<(u16, Material)>,
}

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
        if let Ok(scene) = preprocess_input(input) {
            println!("{} Converting scene...", style("[2/3]").bold().dim());
            if let Ok(scene) = convert_input(scene, input) {
                println!("{} Saving file...", style("[3/3]").bold().dim());
                if write_output(scene, version, output).is_ok() {
                    println!("{}", style("Done!").bold().green());
                } else {
                    eprint!("Failed to save file");
                }
            } else {
                eprintln!("Failed to convert scene");
            }
        } else {
            eprintln!("Failed to preprocess input file");
        }
    } else if benchmark(input, version).is_err() {
            eprintln!("Failed to benchmark scene");
        }
}

fn preprocess_input<S: AsRef<str>>(input: S) -> Result<RussimpScene, Box<dyn Error>> {
    let pb = ProgressBar::new_spinner();
    let postprocess = vec![
        PostProcess::Triangulate,
        PostProcess::ValidateDataStructure,
        PostProcess::JoinIdenticalVertices,
        PostProcess::GenerateUVCoords,
        PostProcess::GenerateNormals,
        PostProcess::OptimizeMeshes,
        PostProcess::OptimizeGraph,
        PostProcess::FindInstances,
        PostProcess::FixInfacingNormals,
        PostProcess::RemoveRedundantMaterials,
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

fn convert_input<S: AsRef<str>>(
    scene: RussimpScene,
    original_path: S,
) -> Result<TempScene, Box<dyn Error>> {
    let mpb = MultiProgress::new();
    let cameras = convert_cameras(&scene, &mpb);
    let (materials, textures) = convert_materials(&scene.materials, &mpb, original_path.as_ref())?;
    let (vertices, meshes) = convert_meshes(&scene.meshes, &mpb)?;
    Ok(TempScene {
        vertices,
        meshes,
        cameras,
        textures,
        materials,
    })
}

fn convert_meshes(
    meshes: &Vec<russimp::mesh::Mesh>,
    mpb: &MultiProgress,
) -> Result<(Vec<Vertex>, Vec<Mesh>), std::io::Error> {
    let mut retval_vertices = Vec::new();
    let mut retval_meshes = Vec::new();
    let mut used_vert = HashMap::new();
    let effort = meshes.iter().fold(0, |acc, m| acc + m.faces.len());
    let pb = mpb.add(ProgressBar::new(effort as u64).with_message("Converting Mesh faces"));
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
                    vt: Vec2::new(vt.x, vt.y),
                };
                let vertex_index = *used_vert
                    .entry(vertex_to_bytes(&vertex))
                    .or_insert(retval_vertices.len() as u32);
                indices.push(vertex_index);
                retval_vertices.push(vertex);
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

fn convert_cameras(scene: &RussimpScene, mpb: &MultiProgress) -> Vec<Camera> {
    let effort = std::cmp::min(scene.cameras.len(), 1);
    let pb = mpb.add(ProgressBar::new(effort as u64).with_message("Converting cameras"));
    let mut retval_cameras = Vec::with_capacity(effort);
    for camera in &scene.cameras {
        retval_cameras.push(Camera::Perspective(PerspectiveCam {
            position: Point3::new(camera.position.x, camera.position.y, camera.position.z),
            target: Point3::new(camera.look_at.x, camera.look_at.y, camera.look_at.z),
            up: Vec3::new(camera.up.x, camera.up.y, camera.up.z),
            fovx: camera.horizontal_fov,
        }));
        pb.inc(1);
    }
    if retval_cameras.is_empty() {
        retval_cameras.push(Camera::Perspective(PerspectiveCam {
            position: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 100.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fovx: f32::to_radians(90.0),
        }));
        pb.inc(1);
    }
    pb.finish();
    retval_cameras
}

fn convert_materials<S: AsRef<str>>(
    materials: &Vec<russimp::material::Material>,
    mpb: &MultiProgress,
    original_path: S,
) -> Result<(Vec<(u16, Material)>, Vec<(u16, Texture)>), std::io::Error> {
    let effort = materials.len();
    let pb =
        mpb.add(ProgressBar::new(effort as u64).with_message("Converting materials and textures"));
    let mut used_textures = HashMap::new();
    let mut retval_textures = Vec::new();
    let mut retval_materials = Vec::new();
    for material in materials {
        let mut diffuse = None;
        for (texture_type, textures) in &material.textures {
            match texture_type {
                russimp::texture::TextureType::Diffuse => {
                    let texture = textures.first().unwrap(); // support single textures only
                                                             // replaces \ with / and hopes UNIX path do not use strange names
                    let tex_name = texture.path.clone().replace("\\", "/");
                    //TODO: add support for embedded textures
                    let mut path = PathBuf::from(&tex_name);
                    if path.is_relative() {
                        path = PathBuf::from(original_path.as_ref())
                            .parent()
                            .unwrap()
                            .join(path);
                    }
                    let id =
                        convert_texture(&tex_name, path, &mut used_textures, &mut retval_textures);
                    diffuse = Some(id);
                }
                _ => {} //unsupported, do nothing
            }
        }
        let properties = &material.properties[..];
        let glaze_material = Material {
            name: format!("Material#{}", retval_materials.len()),
            shader: ShaderMat::Test,
            diffuse,
            diffuse_mul: [255, 255, 255],
        };
        let mat_id = retval_materials.len() as u16;
        retval_materials.push((mat_id, glaze_material));
        pb.inc(1);
    }
    pb.finish();
    Ok((retval_materials, retval_textures))
}

fn convert_texture(
    name: &str,
    path: PathBuf,
    used: &mut HashMap<&str, u16>,
    ret: &mut Vec<(u16, Texture)>,
) -> u16 {
    if let Some(id) = used.get(name) {
        *id
    } else {
        let data = ImageReader::open(path)
            .expect(&format!("Failed to find image {}", name))
            .decode()
            .expect(&format!("Incompatible image {}", name))
            .to_rgba8();
        let info = TextureInfo {
            name: name.to_string(),
            width: data.width() as u16,
            height: data.height() as u16,
            format: TextureFormat::Rgba,
        };
        let texture = Texture::new_rgba(info, data);
        let id = ret.len() as u16;
        ret.push((id, texture));
        id
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
            assert_eq!(parsed.materials()?.len(), 1);
            assert_eq!(parsed.textures()?.len(), 1);
            assert_eq!(parsed.vertices()?.len(), 36);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }
}
