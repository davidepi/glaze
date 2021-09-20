use super::ParsedContent;
use super::HEADER_LEN;
use crate::geometry::Camera;
use crate::geometry::Mesh;
use crate::geometry::OrthographicCam;
use crate::geometry::PerspectiveCam;
use crate::geometry::Scene;
use crate::geometry::Vertex;
use cgmath::Matrix4;
use cgmath::Point3;
use cgmath::Vector2 as Vec2;
use cgmath::Vector3 as Vec3;
use std::convert::TryInto;
use std::fs::File;
use std::io::Error;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;

const CONTENT_LIST_SIZE: usize = std::mem::size_of::<Offsets>();

#[repr(packed)]
struct Offsets {
    vert_len: u64, // amount of data in bytes
    mesh_len: u64,
    cam_len: u16,
}

impl Offsets {
    fn parse(file: &mut File) -> Result<Offsets, Error> {
        file.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        let mut cl_data = [0; CONTENT_LIST_SIZE];
        file.read_exact(&mut cl_data)?;
        let vert_len = u64::from_le_bytes(cl_data[0..8].try_into().unwrap());
        let mesh_len = u64::from_le_bytes(cl_data[8..16].try_into().unwrap());
        let cam_len = u16::from_le_bytes(cl_data[16..18].try_into().unwrap());
        Ok(Offsets {
            vert_len,
            mesh_len,
            cam_len,
        })
    }
}

pub(super) struct ContentV1 {
    vertices: Vec<Vertex>,
    meshes: Vec<Mesh>,
    cameras: Vec<Camera>,
}

impl ContentV1 {
    pub(super) fn parse(file: &mut File) -> Result<Self, Error> {
        let offsets = Offsets::parse(file)?;
        let vert_seek = (HEADER_LEN + CONTENT_LIST_SIZE) as u64;
        file.seek(SeekFrom::Start(vert_seek))?;
        let mut vert_data = Vec::with_capacity(offsets.vert_len as usize);
        file.take(offsets.vert_len).read_to_end(&mut vert_data)?; // avoid zeroing vec
        let mut mesh_data = Vec::with_capacity(offsets.mesh_len as usize);
        file.take(offsets.mesh_len).read_to_end(&mut mesh_data)?; // avoid zeroing vec
        let mut cam_data = Vec::with_capacity(offsets.cam_len as usize);
        file.take(offsets.cam_len as u64)
            .read_to_end(&mut cam_data)?; // avoid zeroing vec
        let vert_chunk = VertexChunk::decode(vert_data);
        let mesh_chunk = MeshChunk::decode(mesh_data);
        let cam_chunk = CameraChunk::decode(cam_data);
        Ok(ContentV1 {
            vertices: vert_chunk.elements(),
            meshes: mesh_chunk.elements(),
            cameras: cam_chunk.elements(),
        })
    }

    pub(super) fn serialize(scene: &Scene) -> Vec<u8> {
        let vert_chunk = VertexChunk::encode(&scene.vertices);
        let mesh_chunk = MeshChunk::encode(&scene.meshes);
        let cam_chunk = CameraChunk::encode(&scene.cameras);
        let vert_size = vert_chunk.size_bytes();
        let mesh_size = mesh_chunk.size_bytes();
        let cam_size = cam_chunk.size_bytes();
        let total_size = CONTENT_LIST_SIZE + vert_size + mesh_size + cam_size;
        let mut retval = Vec::with_capacity(total_size);
        retval.extend_from_slice(&u64::to_le_bytes(vert_size as u64));
        retval.extend_from_slice(&u64::to_le_bytes(mesh_size as u64));
        retval.extend_from_slice(&u16::to_le_bytes(cam_size as u16));
        retval.extend(vert_chunk.data());
        retval.extend(mesh_chunk.data());
        retval.extend(cam_chunk.data());
        retval
    }
}

impl ParsedContent for ContentV1 {
    fn scene(self) -> Scene {
        Scene {
            vertices: self.vertices,
            meshes: self.meshes,
            cameras: self.cameras,
        }
    }

    fn vertices(&self) -> &Vec<Vertex> {
        &self.vertices
    }

    fn meshes(&self) -> &Vec<Mesh> {
        &self.meshes
    }

    fn cameras(&self) -> &Vec<Camera> {
        &self.cameras
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

fn camera_to_bytes(camera: &Camera) -> Vec<u8> {
    let camera_type;
    let position;
    let target;
    let up;
    let other_arg;
    let data = match camera {
        Camera::Perspective(cam) => {
            camera_type = &[0];
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.fov;
        }
        Camera::Orthographic(cam) => {
            camera_type = &[1];
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.scale;
        }
    };
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
            fov: other_val,
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
    use super::bytes_to_camera;
    use super::bytes_to_mesh;
    use super::bytes_to_vertex;
    use super::camera_to_bytes;
    use super::mesh_to_bytes;
    use super::vertex_to_bytes;
    use crate::geometry::Camera;
    use crate::geometry::Mesh;
    use crate::geometry::OrthographicCam;
    use crate::geometry::PerspectiveCam;
    use crate::geometry::Scene;
    use crate::geometry::Vertex;
    use crate::parser::parse;
    use crate::parser::serialize;
    use crate::parser::ParserVersion;
    use cgmath::Matrix4;
    use cgmath::Point3;
    use cgmath::Vector2 as Vec2;
    use cgmath::Vector3 as Vec3;
    use rand::prelude::*;
    use rand::Rng;
    use rand_xoshiro::Xoshiro128StarStar;
    use std::fs::remove_file;
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
                    fov: other_arg,
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
        let file = dir.path().join("write_and_read_meshes.bin");
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
}
