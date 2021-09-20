use super::ParsedContent;
use super::HEADER_LEN;
use crate::geometry::Mesh;
use crate::geometry::Vertex;
use cgmath::Matrix4;
use cgmath::Vector2 as Vec2;
use cgmath::Vector3 as Vec3;
use std::convert::TryInto;
use std::fs::File;
use std::io::Error;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;

const CONTENT_LIST_SIZE: usize = 16;

struct Offsets {
    vert_len: u64, // amount of data in bytes
    mesh_len: u64,
}

impl Offsets {
    fn parse(file: &mut File) -> Result<Offsets, Error> {
        file.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        let mut cl_data = [0; CONTENT_LIST_SIZE];
        file.read_exact(&mut cl_data)?;
        let vert_len = u64::from_le_bytes(cl_data[0..8].try_into().unwrap());
        let mesh_len = u64::from_le_bytes(cl_data[8..16].try_into().unwrap());
        Ok(Offsets { vert_len, mesh_len })
    }
}

pub(super) struct ContentV1 {
    vertices: Vec<Vertex>,
    meshes: Vec<Mesh>,
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
        let vert_chunk = VertexChunk::decode(vert_data);
        let mesh_chunk = MeshChunk::decode(mesh_data);
        Ok(ContentV1 {
            vertices: vert_chunk.elements(),
            meshes: mesh_chunk.elements(),
        })
    }

    pub(super) fn serialize(vert: &[Vertex], meshes: &[Mesh]) -> Vec<u8> {
        let vert_chunk = VertexChunk::encode(vert);
        let mesh_chunk = MeshChunk::encode(meshes);
        let vert_size = vert_chunk.size_bytes();
        let mesh_size = mesh_chunk.size_bytes();
        let total_size = CONTENT_LIST_SIZE + vert_size;
        let mut retval = Vec::with_capacity(total_size);
        retval.extend_from_slice(&u64::to_le_bytes(vert_size as u64));
        retval.extend_from_slice(&u64::to_le_bytes(mesh_size as u64)); // mesh size
        retval.extend(vert_chunk.data());
        retval.extend(mesh_chunk.data());
        retval
    }
}

impl ParsedContent for ContentV1 {
    fn vertices(&self) -> &Vec<Vertex> {
        &self.vertices
    }

    fn meshes(&self) -> &Vec<Mesh> {
        &self.meshes
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

#[cfg(test)]
mod tests {
    use super::bytes_to_mesh;
    use super::bytes_to_vertex;
    use super::mesh_to_bytes;
    use super::vertex_to_bytes;
    use crate::geometry::Mesh;
    use crate::geometry::Vertex;
    use crate::parser::parse;
    use crate::parser::serialize;
    use crate::parser::ParserVersion;
    use cgmath::Matrix4;
    use cgmath::Vector2 as Vec2;
    use cgmath::Vector3 as Vec3;
    use rand::prelude::*;
    use rand::Rng;
    use rand_xoshiro::Xoshiro128StarStar;
    use std::fs::remove_file;
    use tempfile::tempdir;

    fn gen_vertices(len: usize, seed: u64) -> Vec<Vertex> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut buffer = Vec::with_capacity(len);
        for _ in 0..len {
            let vv = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let vn = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let vt = Vec2::<f32>::new(rng.gen(), rng.gen());
            let vertex = Vertex { vv, vn, vt };
            buffer.push(vertex);
        }
        buffer
    }

    fn gen_meshes(len: usize, seed: u64) -> Vec<Mesh> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut buffer = Vec::with_capacity(len);
        for _ in 0..len {
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
    fn write_and_read_only_vert() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000, 0xBE8AE7F7E3A5248E);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_vertices.bin");
        serialize(file.as_path(), ParserVersion::V1, &vertices, &[])?;
        let read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_vertices = read.vertices();
        assert_eq!(read_vertices.len(), vertices.len());
        for i in 0..read_vertices.len() {
            let val = &read_vertices[i];
            let expected = &vertices[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_meshes() -> Result<(), std::io::Error> {
        let meshes = gen_meshes(128, 0xDD219155A3536881);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_meshes.bin");
        serialize(file.as_path(), ParserVersion::V1, &[], &meshes)?;
        let read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_meshes = read.meshes();
        assert_eq!(read_meshes.len(), meshes.len());
        for i in 0..read_meshes.len() {
            let val = &read_meshes[i];
            let expected = &meshes[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }
}
