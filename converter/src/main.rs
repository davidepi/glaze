use cgmath::{Point3, Vector2 as Vec2, Vector3 as Vec3};
use clap::{App, Arg};
use console::style;
use glaze::{
    serialize, Camera, Library, Material, Mesh, ParserVersion, PerspectiveCam, Scene, ShaderMat,
    Texture, Vertex,
};
use image::io::Reader as ImageReader;
use indicatif::{MultiProgress, ProgressBar};
use russimp::scene::{PostProcess, Scene as RussimpScene};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::str::FromStr;

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
                .required(true)
                .help("The output file"),
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
    let output = matches.value_of("output").unwrap();
    let version = ParserVersion::from_str(matches.value_of("file version").unwrap()).unwrap();
    println!("{} Preprocessing input...", style("[1/3]").bold().dim());
    if let Ok(scene) = preprocess_input(input) {
        println!("{} Converting scene...", style("[2/3]").bold().dim());
        if let Ok(scene) = convert_input(scene, input) {
            println!("{} Saving file...", style("[3/3]").bold().dim());
            if write_output(&scene, version, output).is_ok() {
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
}

fn preprocess_input(input: &str) -> Result<RussimpScene, Box<dyn Error>> {
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
    let scene = RussimpScene::from_file(input, postprocess)?;
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

fn convert_input(scene: RussimpScene, original_path: &str) -> Result<Scene, Box<dyn Error>> {
    let mpb = MultiProgress::new();
    let cameras = convert_cameras(&scene, &mpb);
    let (materials, textures) = convert_materials(&scene.materials, &mpb, original_path)?;
    let (vertices, meshes) = convert_meshes(&scene.meshes, &mpb)?;
    Ok(Scene {
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

fn convert_materials(
    materials: &Vec<russimp::material::Material>,
    mpb: &MultiProgress,
    original_path: &str,
) -> Result<(Library<Material>, Library<Texture>), std::io::Error> {
    let effort = materials.len();
    let pb =
        mpb.add(ProgressBar::new(effort as u64).with_message("Converting materials and textures"));
    let mut used_textures = HashMap::new();
    let mut retval_textures = Library::new();
    let mut retval_materials = Library::new();
    for material in materials {
        let mut diffuse = None;
        for (texture_type, textures) in &material.textures {
            match texture_type {
                russimp::texture::TextureType::Diffuse => {
                    let texture = textures.first().unwrap(); // support single textures only
                                                             // replaces \ with / and hopes UNIX path do not use strange names
                    let texture_name = texture.path.clone().replace("\\", "/");
                    //TODO: add support for embedded textures
                    let mut path = PathBuf::from(&texture_name);
                    if path.is_relative() {
                        path = PathBuf::from(original_path).parent().unwrap().join(path);
                    }
                    let id = *used_textures
                        .entry(texture_name)
                        //TODO: better error handling
                        .or_insert_with_key(|name| {
                            retval_textures.insert(
                                name,
                                ImageReader::open(path)
                                    .expect(&format!("Failed to find image {}", name))
                                    .decode()
                                    .expect(&format!("Incompatible image {}", name))
                                    .to_rgba8(),
                            )
                        });
                    diffuse = Some(id);
                }
                _ => {} //unsupported, do nothing
            }
        }
        let glaze_material = Material {
            shader_id: ShaderMat::Test.id(),
            diffuse,
        };
        retval_materials.insert(
            &format!("Material#{}", retval_materials.len()),
            glaze_material,
        );
        pb.inc(1);
    }
    pb.finish();
    Ok((retval_materials, retval_textures))
}

fn write_output(scene: &Scene, version: ParserVersion, output: &str) -> Result<(), Box<dyn Error>> {
    Ok(serialize(output, version, scene)?)
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
        assert!(serialize(&file, glaze::ParserVersion::V1, &scene).is_ok());
        let parsed = parse(&file);
        assert!(parsed.is_ok());
        if let Ok(parsed) = parsed {
            assert_eq!(parsed.meshes().len(), 1);
            assert_eq!(parsed.cameras().len(), 1);
            assert_eq!(parsed.materials().len(), 1);
            assert_eq!(parsed.textures().len(), 1);
            assert_eq!(parsed.vertices().len(), 36);
        } else {
            panic!("Failed to parse back scene")
        }
        Ok(())
    }
}
