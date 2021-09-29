use super::filehasher::FileHasher;
use super::{ParsedContent, HEADER_LEN};
use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Scene, Vertex};
use crate::materials::{Library, Texture};
use cgmath::{Matrix4, Point3, Vector2 as Vec2, Vector3 as Vec3};
use std::convert::TryInto;
use std::hash::Hasher;
use std::io::{Error, Read, Seek, SeekFrom};
use twox_hash::XxHash64;

const CONTENT_LIST_SIZE: usize = std::mem::size_of::<Offsets>();
const HASHER_SEED: u64 = 0x368262AAA1DEB64D;
const HASH_SIZE: usize = std::mem::size_of::<u64>();
fn get_hasher() -> impl Hasher {
    XxHash64::with_seed(HASHER_SEED)
}

#[repr(packed)]
struct Offsets {
    vert_len: u64, // amount of data in bytes
    mesh_len: u64,
    cam_len: u16,
    tex_len: u64,
}

impl Offsets {
    fn seek_and_parse<R: Read + Seek>(file: &mut R) -> Result<Offsets, Error> {
        file.seek(SeekFrom::Start((HEADER_LEN + HASH_SIZE) as u64))?;
        let mut cl_data = [0; CONTENT_LIST_SIZE];
        file.read_exact(&mut cl_data)?;
        let vert_len = u64::from_le_bytes(cl_data[0..8].try_into().unwrap());
        let mesh_len = u64::from_le_bytes(cl_data[8..16].try_into().unwrap());
        let cam_len = u16::from_le_bytes(cl_data[16..18].try_into().unwrap());
        let tex_len = u64::from_le_bytes(cl_data[18..26].try_into().unwrap());
        Ok(Offsets {
            vert_len,
            mesh_len,
            cam_len,
            tex_len,
        })
    }
}

pub(super) struct ContentV1 {
    vertices: Vec<Vertex>,
    meshes: Vec<Mesh>,
    cameras: Vec<Camera>,
    textures: Vec<(u16, String, Texture)>,
}

impl ContentV1 {
    pub(super) fn parse<R: Read + Seek>(file: &mut R) -> Result<Self, Error> {
        let mut hash_buf = [0; HASH_SIZE];
        file.read_exact(&mut hash_buf)?;
        let expected_hash = u64::from_le_bytes(hash_buf);
        let hasher = get_hasher();
        let actual_hash = FileHasher::new(hasher).hash(file)?;
        if expected_hash == actual_hash {
            let offsets = Offsets::seek_and_parse(file)?;
            let mut vert_data = Vec::with_capacity(offsets.vert_len as usize);
            file.take(offsets.vert_len).read_to_end(&mut vert_data)?; // avoid zeroing vec
            let mut mesh_data = Vec::with_capacity(offsets.mesh_len as usize);
            file.take(offsets.mesh_len).read_to_end(&mut mesh_data)?; // avoid zeroing vec
            let mut cam_data = Vec::with_capacity(offsets.cam_len as usize);
            file.take(offsets.cam_len as u64)
                .read_to_end(&mut cam_data)?;
            let mut tex_data = Vec::with_capacity(offsets.tex_len as usize);
            file.take(offsets.tex_len).read_to_end(&mut tex_data)?; // avoid zeroing vec
            let vert_chunk = VertexChunk::decode(vert_data);
            let mesh_chunk = MeshChunk::decode(mesh_data);
            let cam_chunk = CameraChunk::decode(cam_data);
            let tex_chunk = TextureChunk::decode(tex_data);
            Ok(ContentV1 {
                vertices: vert_chunk.elements(),
                meshes: mesh_chunk.elements(),
                cameras: cam_chunk.elements(),
                textures: tex_chunk.elements(),
            })
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted file",
            ))
        }
    }

    pub(super) fn serialize(scene: &Scene) -> Vec<u8> {
        let vert_chunk = VertexChunk::encode(&scene.vertices);
        let mesh_chunk = MeshChunk::encode(&scene.meshes);
        let cam_chunk = CameraChunk::encode(&scene.cameras);
        let tex_chunk = TextureChunk::encode(&scene.textures.iter());
        let vert_size = vert_chunk.size_bytes();
        let mesh_size = mesh_chunk.size_bytes();
        let cam_size = cam_chunk.size_bytes();
        let tex_size = tex_chunk.size_bytes();
        let total_size =
            HASH_SIZE + CONTENT_LIST_SIZE + vert_size + mesh_size + cam_size + tex_size;
        let mut retval = Vec::with_capacity(total_size);
        retval.extend([0; HASH_SIZE]);
        retval.extend_from_slice(&u64::to_le_bytes(vert_size as u64));
        retval.extend_from_slice(&u64::to_le_bytes(mesh_size as u64));
        retval.extend_from_slice(&u16::to_le_bytes(cam_size as u16));
        retval.extend_from_slice(&u64::to_le_bytes(tex_size as u64));
        retval.extend(vert_chunk.data());
        retval.extend(mesh_chunk.data());
        retval.extend(cam_chunk.data());
        retval.extend(tex_chunk.data());
        let mut hasher = get_hasher();
        hasher.write(&retval[HASH_SIZE..]);
        let hash = hasher.finish().to_le_bytes();
        retval[0..HASH_SIZE].copy_from_slice(&hash);
        retval
    }
}

impl ParsedContent for ContentV1 {
    fn scene(&self) -> Scene {
        // super inefficient for V1, other versions should load on demand from file to decrease
        // memory footprint
        Scene {
            vertices: self.vertices.clone(),
            meshes: self.meshes.clone(),
            cameras: self.cameras.clone(),
            textures: self.textures.iter().cloned().collect(),
        }
    }

    fn vertices(&self) -> Vec<Vertex> {
        self.vertices.clone()
    }

    fn meshes(&self) -> Vec<Mesh> {
        self.meshes.clone()
    }

    fn cameras(&self) -> Vec<Camera> {
        self.cameras.clone()
    }

    fn textures(&self) -> Library<Texture> {
        self.textures.iter().cloned().collect()
    }
}

trait ParsedChunk {
    type Item;
    fn encode(item: &[Self::Item]) -> Self;
    fn decode(data: Vec<u8>) -> Self;
    fn size_bytes(&self) -> usize;
    fn elements(self) -> Vec<Self::Item>;
    fn data(self) -> Vec<u8>;
}

struct VertexChunk {
    data: Vec<u8>,
}

impl ParsedChunk for VertexChunk {
    type Item = Vertex;

    fn encode(item: &[Self::Item]) -> Self {
        let data = item.into_iter().flat_map(vertex_to_bytes).collect();
        VertexChunk { data }
    }

    fn decode(data: Vec<u8>) -> Self {
        VertexChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Vec<Self::Item> {
        self.data.chunks_exact(32).map(bytes_to_vertex).collect()
    }

    fn data(self) -> Vec<u8> {
        self.data
    }
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

fn bytes_to_vertex(data: &[u8]) -> Vertex {
    let vv = Vec3::new(
        f32::from_le_bytes(data[0..4].try_into().unwrap()),
        f32::from_le_bytes(data[4..8].try_into().unwrap()),
        f32::from_le_bytes(data[8..12].try_into().unwrap()),
    );
    let vn = Vec3::new(
        f32::from_le_bytes(data[12..16].try_into().unwrap()),
        f32::from_le_bytes(data[16..20].try_into().unwrap()),
        f32::from_le_bytes(data[20..24].try_into().unwrap()),
    );
    let vt = Vec2::new(
        f32::from_le_bytes(data[24..28].try_into().unwrap()),
        f32::from_le_bytes(data[28..32].try_into().unwrap()),
    );
    Vertex { vv, vn, vt }
}

struct MeshChunk {
    data: Vec<u8>,
}

impl ParsedChunk for MeshChunk {
    type Item = Mesh;

    fn encode(item: &[Self::Item]) -> Self {
        let len = item.len() as u32;
        let mut data = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
        for mesh in item {
            let encoded = mesh_to_bytes(mesh);
            let encoded_len = encoded.len() as u32;
            data.extend(encoded_len.to_le_bytes().iter());
            data.extend(encoded);
        }
        MeshChunk { data }
    }

    fn decode(data: Vec<u8>) -> Self {
        MeshChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Vec<Self::Item> {
        let len = u32::from_le_bytes(self.data[0..4].try_into().unwrap());
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = 4;
        while index < self.size_bytes() {
            let encoded_len =
                u32::from_le_bytes(self.data[index..index + 4].try_into().unwrap()) as usize;
            index += 4;
            let mesh = bytes_to_mesh(&self.data[index..index + encoded_len]);
            index += encoded_len;
            retval.push(mesh);
        }
        retval
    }

    fn data(self) -> Vec<u8> {
        self.data
    }
}

fn mesh_to_bytes(mesh: &Mesh) -> Vec<u8> {
    let faces_no = u32::to_le_bytes(mesh.indices.len() as u32);
    let instances_no = u32::to_le_bytes(mesh.instances.len() as u32);
    let material = u16::to_le_bytes(mesh.material);
    let mut retval = faces_no
        .iter()
        .chain(instances_no.iter())
        .chain(material.iter())
        .copied()
        .collect::<Vec<_>>();
    retval.extend(mesh.indices.iter().copied().flat_map(u32::to_le_bytes));
    for matrix in &mesh.instances {
        let iter = <Matrix4<f32> as AsRef<[f32; 16]>>::as_ref(matrix)
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes);
        retval.extend(iter);
    }
    retval
}

fn bytes_to_mesh(data: &[u8]) -> Mesh {
    let faces_no = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let instances_no = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let face_end = (10 + faces_no * 4) as usize;
    let material = u16::from_le_bytes(data[8..10].try_into().unwrap());
    let indices = data[10..face_end]
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut instances = Vec::with_capacity(instances_no as usize);
    for instance_data in data[face_end..].chunks_exact(16 * 4) {
        let m00 = f32::from_le_bytes(instance_data[0..4].try_into().unwrap());
        let m01 = f32::from_le_bytes(instance_data[4..8].try_into().unwrap());
        let m02 = f32::from_le_bytes(instance_data[8..12].try_into().unwrap());
        let m03 = f32::from_le_bytes(instance_data[12..16].try_into().unwrap());
        let m10 = f32::from_le_bytes(instance_data[16..20].try_into().unwrap());
        let m11 = f32::from_le_bytes(instance_data[20..24].try_into().unwrap());
        let m12 = f32::from_le_bytes(instance_data[24..28].try_into().unwrap());
        let m13 = f32::from_le_bytes(instance_data[28..32].try_into().unwrap());
        let m20 = f32::from_le_bytes(instance_data[32..36].try_into().unwrap());
        let m21 = f32::from_le_bytes(instance_data[36..40].try_into().unwrap());
        let m22 = f32::from_le_bytes(instance_data[40..44].try_into().unwrap());
        let m23 = f32::from_le_bytes(instance_data[44..48].try_into().unwrap());
        let m30 = f32::from_le_bytes(instance_data[48..52].try_into().unwrap());
        let m31 = f32::from_le_bytes(instance_data[52..56].try_into().unwrap());
        let m32 = f32::from_le_bytes(instance_data[56..60].try_into().unwrap());
        let m33 = f32::from_le_bytes(instance_data[60..64].try_into().unwrap());
        let matrix = Matrix4::new(
            m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33,
        );
        instances.push(matrix);
    }
    Mesh {
        indices,
        material,
        instances,
    }
}

struct CameraChunk {
    data: Vec<u8>,
}

impl ParsedChunk for CameraChunk {
    type Item = Camera;

    fn encode(item: &[Self::Item]) -> Self {
        let mut data = vec![item.len() as u8];
        for camera in item {
            let encoded = camera_to_bytes(camera);
            let encoded_len = encoded.len() as u8;
            data.push(encoded_len);
            data.extend(encoded);
        }
        CameraChunk { data }
    }

    fn decode(data: Vec<u8>) -> Self {
        CameraChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Vec<Self::Item> {
        let len = self.data[0];
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = 1;
        while index < self.size_bytes() {
            let encoded_len = self.data[index] as usize;
            index += 1;
            let camera = bytes_to_camera(&self.data[index..index + encoded_len]);
            index += encoded_len;
            retval.push(camera);
        }
        retval
    }

    fn data(self) -> Vec<u8> {
        self.data
    }
}

struct TextureChunk {
    data: Vec<u8>,
}

impl ParsedChunk for TextureChunk {
    type Item = (u16, String, Texture);

    fn encode(item: &[Self::Item]) -> Self {
        let len = item.len() as u16;
        let mut data = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
        for texture in item {
            let encoded = texture_to_bytes(texture);
            let encoded_len = encoded.len() as u32;
            data.extend(encoded_len.to_le_bytes().iter());
            data.extend(encoded);
        }
        TextureChunk { data }
    }

    fn decode(data: Vec<u8>) -> Self {
        TextureChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Vec<Self::Item> {
        let len = u16::from_le_bytes(self.data[0..2].try_into().unwrap());
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = std::mem::size_of::<u16>();
        while index < self.size_bytes() {
            let encoded_len =
                u32::from_le_bytes(self.data[index..index + 4].try_into().unwrap()) as usize;
            index += std::mem::size_of::<u32>();
            let texture = bytes_to_texture(&self.data[index..index + encoded_len]);
            index += encoded_len;
            retval.push(texture);
        }
        retval
    }

    fn data(self) -> Vec<u8> {
        self.data
    }
}

fn texture_to_bytes((index, name, texture): &(u16, String, Texture)) -> Vec<u8> {
    let str_len = name.bytes().len();
    assert!(str_len < 256);
    let tex_width = texture.width();
    let tex_height = texture.height();
    let tex_data = texture.as_raw();
    let tex_len = tex_data.len();
    let total_len = std::mem::size_of::<u16>()
        + std::mem::size_of::<u8>()
        + str_len
        + 2 * std::mem::size_of::<u32>()
        + tex_len;
    let mut retval = Vec::with_capacity(total_len);
    retval.extend(index.to_le_bytes());
    retval.push(str_len as u8);
    retval.extend(name.bytes());
    retval.extend(tex_width.to_le_bytes());
    retval.extend(tex_height.to_le_bytes());
    retval.extend(tex_data);
    retval
}

fn bytes_to_texture(data: &[u8]) -> (u16, String, Texture) {
    let tex_index = u16::from_le_bytes(data[0..2].try_into().unwrap());
    let str_len = data[2] as usize;
    let mut index = 3;
    let name = String::from_utf8(data[index..index + str_len].to_vec()).unwrap();
    index += str_len;
    let width = u32::from_le_bytes(data[index..index + 4].try_into().unwrap());
    index += 4;
    let height = u32::from_le_bytes(data[index..index + 4].try_into().unwrap());
    index += 4;
    let texture =
        Texture::from_raw(width, height, data[index..].to_vec()).expect("Corrupted image");
    (tex_index, name, texture)
}

fn camera_to_bytes(camera: &Camera) -> Vec<u8> {
    let camera_type;
    let position;
    let target;
    let up;
    let other_arg;
    match camera {
        Camera::Perspective(cam) => {
            camera_type = &[0];
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.fovx;
        }
        Camera::Orthographic(cam) => {
            camera_type = &[1];
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.scale;
        }
    }
    let pos: [f32; 3] = Point3::into(position);
    let tgt: [f32; 3] = Point3::into(target);
    let upp: [f32; 3] = Vec3::into(up);
    let oth = f32::to_le_bytes(other_arg);
    let cam_data_iter = pos
        .iter()
        .chain(tgt.iter())
        .chain(upp.iter())
        .copied()
        .flat_map(f32::to_le_bytes)
        .chain(oth.iter().copied());
    camera_type.iter().copied().chain(cam_data_iter).collect()
}

fn bytes_to_camera(data: &[u8]) -> Camera {
    let cam_type = data[0];
    let position = Point3::new(
        f32::from_le_bytes(data[1..5].try_into().unwrap()),
        f32::from_le_bytes(data[5..9].try_into().unwrap()),
        f32::from_le_bytes(data[9..13].try_into().unwrap()),
    );
    let target = Point3::new(
        f32::from_le_bytes(data[13..17].try_into().unwrap()),
        f32::from_le_bytes(data[17..21].try_into().unwrap()),
        f32::from_le_bytes(data[21..25].try_into().unwrap()),
    );
    let up = Vec3::new(
        f32::from_le_bytes(data[25..29].try_into().unwrap()),
        f32::from_le_bytes(data[29..33].try_into().unwrap()),
        f32::from_le_bytes(data[33..37].try_into().unwrap()),
    );
    let other_val = f32::from_le_bytes(data[37..41].try_into().unwrap());
    match cam_type {
        0 => Camera::Perspective(PerspectiveCam {
            position,
            target,
            up,
            fovx: other_val,
        }),
        1 => Camera::Orthographic(OrthographicCam {
            position,
            target,
            up,
            scale: other_val,
        }),
        _ => {
            panic!("Unexpected cam type")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_to_camera, bytes_to_mesh, bytes_to_texture, bytes_to_vertex, camera_to_bytes,
        mesh_to_bytes, texture_to_bytes, vertex_to_bytes,
    };
    use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Scene, Vertex};
    use crate::materials::{Library, Texture};
    use crate::parser::{parse, serialize, ParserVersion};
    use cgmath::{Matrix4, Point3, Vector2 as Vec2, Vector3 as Vec3};
    use rand::distributions::Alphanumeric;
    use rand::prelude::*;
    use rand::Rng;
    use rand_xoshiro::Xoshiro128StarStar;
    use std::fs::{remove_file, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::tempdir;

    fn gen_vertices(count: u32, seed: u64) -> Vec<Vertex> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut buffer = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let vv = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let vn = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let vt = Vec2::<f32>::new(rng.gen(), rng.gen());
            let vertex = Vertex { vv, vn, vt };
            buffer.push(vertex);
        }
        buffer
    }

    fn gen_meshes(count: u32, seed: u64) -> Vec<Mesh> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut buffer = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let high_range = rng.gen_bool(0.1);
            let has_instances = rng.gen_bool(0.3);
            let indices_no: u32 = if high_range {
                rng.gen_range(0..100000)
            } else {
                rng.gen_range(0..1000)
            };
            let material = rng.gen();
            let instances_no: u8 = if has_instances { rng.gen() } else { 0 };
            let indices = (0..indices_no).map(|_| rng.gen()).collect::<Vec<_>>();
            let mut instances = Vec::with_capacity(instances_no as usize);
            for _ in 0..instances_no {
                let matrix = Matrix4::<f32>::new(
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                );
                instances.push(matrix);
            }
            let mesh = Mesh {
                indices,
                material,
                instances,
            };
            buffer.push(mesh);
        }
        buffer
    }

    fn gen_cameras(count: u8, seed: u64) -> Vec<Camera> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut buffer = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let cam_type = rng.gen_range(1..=2);
            let position = Point3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let target = Point3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let up = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let other_arg = rng.gen();
            let cam = match cam_type {
                1 => Camera::Perspective(PerspectiveCam {
                    position,
                    target,
                    up,
                    fovx: other_arg,
                }),
                2 => Camera::Orthographic(OrthographicCam {
                    position,
                    target,
                    up,
                    scale: other_arg,
                }),
                _ => {
                    panic!("Non existing variant");
                }
            };
            buffer.push(cam);
        }
        buffer
    }

    fn gen_textures(count: u16, seed: u64) -> Library<Texture> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let data = include_bytes!("../../../resources/checker.jpg");
        let image = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg).unwrap();
        let mut data = Vec::with_capacity(count as usize);
        for i in 0..count {
            let cur_img = image.clone();
            if rng.gen_bool(0.5) {
                cur_img.flipv();
            }
            if rng.gen_bool(0.5) {
                cur_img.fliph();
            }
            if rng.gen_bool(0.5) {
                cur_img.rotate90();
            }
            let name = Xoshiro128StarStar::seed_from_u64(rng.gen())
                .sample_iter(&Alphanumeric)
                .take(64)
                .map(char::from)
                .collect::<String>();
            data.push((i, name, cur_img.into_rgba8()));
        }
        data.into_iter().collect()
    }

    #[test]
    fn encode_decode_vertex() {
        let vertices = gen_vertices(32, 0xC2B4D5A5A9E49945);
        for vertex in vertices {
            let data = vertex_to_bytes(&vertex);
            let decoded = bytes_to_vertex(&data);
            assert_eq!(decoded, vertex);
        }
    }

    #[test]
    fn encode_decode_mesh() {
        let meshes = gen_meshes(32, 0x7BDF6FF7246CABDF);
        for mesh in meshes {
            let data = mesh_to_bytes(&mesh);
            let decoded = bytes_to_mesh(&data);
            assert_eq!(decoded, mesh);
        }
    }

    #[test]
    fn encode_decode_cameras() {
        let cameras = gen_cameras(128, 0xB983667AE564853E);
        for camera in cameras {
            let data = camera_to_bytes(&camera);
            let decoded = bytes_to_camera(&data);
            assert_eq!(decoded, camera);
        }
    }

    #[test]
    fn encode_decode_textures() {
        let textures = gen_textures(16, 0xE59CCF79D9AD3162);
        for texture in textures {
            let data = texture_to_bytes(&texture);
            let decoded = bytes_to_texture(&data);
            assert_eq!(decoded, texture);
        }
    }

    #[test]
    fn corrupted() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000, 0x62A9F273AF56253C);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted.bin");
        let mut scene = Scene::default();
        scene.vertices = vertices;
        serialize(file.as_path(), ParserVersion::V1, &scene)?;
        let read_ok = parse(file.as_path());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start(32000))?;
            file.write(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let read_corrupted = parse(file.as_path());
        remove_file(file.as_path())?;
        assert!(read_ok.is_ok());
        assert!(read_corrupted.is_err());
        Ok(())
    }

    #[test]
    fn write_and_read_only_vert() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000, 0xBE8AE7F7E3A5248E);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_vertices.bin");
        let mut scene = Scene::default();
        scene.vertices = vertices;
        serialize(file.as_path(), ParserVersion::V1, &scene)?;
        let read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_vertices = read.vertices();
        assert_eq!(read_vertices.len(), scene.vertices.len());
        for i in 0..read_vertices.len() {
            let val = &read_vertices[i];
            let expected = &scene.vertices[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_meshes() -> Result<(), std::io::Error> {
        let meshes = gen_meshes(128, 0xDD219155A3536881);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_meshes.bin");
        let mut scene = Scene::default();
        scene.meshes = meshes;
        serialize(file.as_path(), ParserVersion::V1, &scene)?;
        let read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_meshes = read.meshes();
        assert_eq!(read_meshes.len(), scene.meshes.len());
        for i in 0..read_meshes.len() {
            let val = &read_meshes[i];
            let expected = &scene.meshes[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_cameras() -> Result<(), std::io::Error> {
        let cameras = gen_cameras(32, 0xCC6AD9820F396116);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_cameras.bin");
        let mut scene = Scene::default();
        scene.cameras = cameras;
        serialize(file.as_path(), ParserVersion::V1, &scene)?;
        let read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_cameras = read.cameras();
        assert_eq!(read_cameras.len(), scene.cameras.len());
        for i in 0..read_cameras.len() {
            let val = &read_cameras[i];
            let expected = &scene.cameras[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_textures() -> Result<(), std::io::Error> {
        let textures = gen_textures(4, 0x50DFC0EA9BF6E9BE);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_textures.bin");
        let mut scene = Scene::default();
        scene.textures = textures;
        serialize(file.as_path(), ParserVersion::V1, &scene)?;
        let read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_textures = read.textures();
        assert_eq!(read_textures.len(), scene.textures.len());
        for i in 0..read_textures.len() {
            let val = &read_textures.get(i).unwrap();
            let expected = &scene.textures.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }
}
