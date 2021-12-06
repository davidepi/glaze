use super::{ReadParsed, HEADER_LEN};
use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
use crate::materials::{TextureFormat, TextureInfo};
use crate::{Material, Texture};
use cgmath::{Matrix4, Point3, Vector2 as Vec2, Vector3 as Vec3};
use image::png::{PngDecoder, PngEncoder};
use image::{GrayImage, ImageDecoder, RgbaImage};
use std::convert::TryInto;
use std::hash::Hasher;
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use twox_hash::XxHash64;
use xz2::read::{XzDecoder, XzEncoder};

const CONTENT_LIST_SIZE: usize = std::mem::size_of::<Offsets>();
const HASHER_SEED: u64 = 0x368262AAA1DEB64D;
const HASH_SIZE: usize = std::mem::size_of::<u64>();
const OFFSET_SIZE: usize = std::mem::size_of::<Offsets>();

fn get_hasher() -> impl Hasher {
    XxHash64::with_seed(HASHER_SEED)
}

fn compress(data: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    let mut encoder = XzEncoder::new(data, 9);
    encoder
        .read_to_end(&mut compressed)
        .expect("Failed to compress data");
    compressed
}

fn decompress(data: &[u8]) -> Vec<u8> {
    let mut decoder = XzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .expect("Failed to decompress data");
    decompressed
}

struct Offsets {
    // in bytes from the very beginning of the file (including the common header)
    vert_off: u64,
    vert_len: u64,
    vert_hash: u64,
    mesh_off: u64,
    mesh_len: u64,
    mesh_hash: u64,
    cam_off: u64,
    cam_len: u64,
    cam_hash: u64,
    tex_off: u64,
    tex_len: u64,
    tex_hash: u64,
    mat_off: u64,
    mat_len: u64,
    mat_hash: u64,
    // currently unused
    _light_len: u64,
    _light_off: u64,
    _next: u64,
}

impl Offsets {
    fn seek_and_parse<R: Read + Seek>(file: &mut R) -> Result<Offsets, Error> {
        file.seek(SeekFrom::Start((HEADER_LEN + HASH_SIZE) as u64))?;
        let mut cl_data = [0; CONTENT_LIST_SIZE];
        file.read_exact(&mut cl_data)?;
        let vert_off = u64::from_le_bytes(cl_data[0..8].try_into().unwrap());
        let vert_len = u64::from_le_bytes(cl_data[8..16].try_into().unwrap());
        let vert_hash = u64::from_le_bytes(cl_data[16..24].try_into().unwrap());
        let mesh_off = u64::from_le_bytes(cl_data[24..32].try_into().unwrap());
        let mesh_len = u64::from_le_bytes(cl_data[32..40].try_into().unwrap());
        let mesh_hash = u64::from_le_bytes(cl_data[40..48].try_into().unwrap());
        let cam_off = u64::from_le_bytes(cl_data[48..56].try_into().unwrap());
        let cam_len = u64::from_le_bytes(cl_data[56..64].try_into().unwrap());
        let cam_hash = u64::from_le_bytes(cl_data[64..72].try_into().unwrap());
        let tex_off = u64::from_le_bytes(cl_data[72..80].try_into().unwrap());
        let tex_len = u64::from_le_bytes(cl_data[80..88].try_into().unwrap());
        let tex_hash = u64::from_le_bytes(cl_data[88..96].try_into().unwrap());
        let mat_off = u64::from_le_bytes(cl_data[96..104].try_into().unwrap());
        let mat_len = u64::from_le_bytes(cl_data[104..112].try_into().unwrap());
        let mat_hash = u64::from_le_bytes(cl_data[112..120].try_into().unwrap());
        Ok(Offsets {
            vert_off,
            vert_len,
            vert_hash,
            mesh_off,
            mesh_len,
            mesh_hash,
            cam_off,
            cam_len,
            cam_hash,
            tex_off,
            tex_len,
            tex_hash,
            mat_off,
            mat_len,
            mat_hash,
            _light_len: 0,
            _light_off: 0,
            _next: 0,
        })
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(120);
        bytes.extend(&self.vert_off.to_le_bytes());
        bytes.extend(&self.vert_len.to_le_bytes());
        bytes.extend(&self.vert_hash.to_le_bytes());
        bytes.extend(&self.mesh_off.to_le_bytes());
        bytes.extend(&self.mesh_len.to_le_bytes());
        bytes.extend(&self.mesh_hash.to_le_bytes());
        bytes.extend(&self.cam_off.to_le_bytes());
        bytes.extend(&self.cam_len.to_le_bytes());
        bytes.extend(&self.cam_hash.to_le_bytes());
        bytes.extend(&self.tex_off.to_le_bytes());
        bytes.extend(&self.tex_len.to_le_bytes());
        bytes.extend(&self.tex_hash.to_le_bytes());
        bytes.extend(&self.mat_off.to_le_bytes());
        bytes.extend(&self.mat_len.to_le_bytes());
        bytes.extend(&self.mat_hash.to_le_bytes());
        bytes
    }
}

pub(super) struct ContentV1<R: Read + Seek> {
    file: R,
    offsets: Offsets,
}

impl<R: Read + Seek> ContentV1<R> {
    pub(super) fn parse(mut file: R) -> Result<Self, Error> {
        let mut offset_hash = [0; HASH_SIZE];
        file.read_exact(&mut offset_hash)?;
        let expected_hash = u64::from_le_bytes(offset_hash);
        let mut hasher = get_hasher();
        let offsets = Offsets::seek_and_parse(&mut file)?;
        hasher.write(&offsets.as_bytes());
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            let offsets = Offsets::seek_and_parse(&mut file)?;
            Ok(ContentV1 { file, offsets })
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted file structure",
            ))
        }
    }

    pub(super) fn serialize<W: Write + Seek>(
        mut fout: W,
        vertices: &[Vertex],
        meshes: &[Mesh],
        cameras: &[Camera],
        texture: &[(u16, Texture)],
        materials: &[(u16, Material)],
    ) -> Result<(), Error> {
        let base_offset = HEADER_LEN + HASH_SIZE + OFFSET_SIZE;
        fout.seek(SeekFrom::Start(base_offset as u64))?;
        // vertices
        let mut hasher = get_hasher();
        let chunk = VertexChunk::encode(vertices);
        let vert_size = chunk.size_bytes();
        let data = chunk.data();
        hasher.write(&data);
        #[allow(clippy::erasing_op)] // yes, I want to multiply by 0 for consistency
        let vert_offset = base_offset + 0 * HASH_SIZE;
        let vert_hash = hasher.finish();
        fout.write_all(vert_hash.to_le_bytes().as_ref())?;
        fout.write_all(&data)?;
        // meshes
        let mut hasher = get_hasher();
        let chunk = MeshChunk::encode(meshes);
        let mesh_size = chunk.size_bytes();
        let data = chunk.data();
        hasher.write(&data);
        #[allow(clippy::identity_op)] // shut up clippy, it helps me when I need to modify it
        let mesh_offset = base_offset + vert_size + 1 * HASH_SIZE;
        let mesh_hash = hasher.finish();
        fout.write_all(mesh_hash.to_le_bytes().as_ref())?;
        fout.write_all(&data)?;
        // cameras
        let mut hasher = get_hasher();
        let chunk = CameraChunk::encode(cameras);
        let cam_size = chunk.size_bytes();
        let data = chunk.data();
        hasher.write(&data);
        let cam_offset = base_offset + vert_size + mesh_size + 2 * HASH_SIZE;
        let cam_hash = hasher.finish();
        fout.write_all(cam_hash.to_le_bytes().as_ref())?;
        fout.write_all(&data)?;
        // textures
        let mut hasher = get_hasher();
        let chunk = TextureChunk::encode(texture);
        let tex_size = chunk.size_bytes();
        let data = chunk.data();
        hasher.write(&data);
        let tex_offset = base_offset + vert_size + mesh_size + cam_size + 3 * HASH_SIZE;
        let tex_hash = hasher.finish();
        fout.write_all(tex_hash.to_le_bytes().as_ref())?;
        fout.write_all(&data)?;
        // materials
        let mut hasher = get_hasher();
        let chunk = MaterialChunk::encode(materials);
        let mat_size = chunk.size_bytes();
        let data = chunk.data();
        hasher.write(&data);
        let mat_offset = base_offset + vert_size + mesh_size + cam_size + tex_size + 4 * HASH_SIZE;
        let mat_hash = hasher.finish();
        fout.write_all(mat_hash.to_le_bytes().as_ref())?;
        fout.write_all(&data)?;
        // offsets
        let mut hasher = get_hasher();
        let offsets = Offsets {
            vert_off: vert_offset as u64,
            vert_len: vert_size as u64,
            vert_hash,
            mesh_off: mesh_offset as u64,
            mesh_len: mesh_size as u64,
            mesh_hash,
            cam_off: cam_offset as u64,
            cam_len: cam_size as u64,
            cam_hash,
            tex_off: tex_offset as u64,
            tex_len: tex_size as u64,
            tex_hash,
            mat_off: mat_offset as u64,
            mat_len: mat_size as u64,
            mat_hash,
            _light_len: 0,
            _light_off: 0,
            _next: 0,
        }
        .as_bytes();
        hasher.write(&offsets);
        let off_hash = hasher.finish();
        fout.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        fout.write_all(off_hash.to_le_bytes().as_ref())?;
        fout.write_all(&offsets)?;
        Ok(())
    }
}

impl<R: Read + Seek> ReadParsed for ContentV1<R> {
    fn vertices(&mut self) -> Result<Vec<Vertex>, Error> {
        let mut read_hash = [0; HASH_SIZE];
        let mut vert_data = Vec::with_capacity(self.offsets.vert_len as usize);
        self.file.seek(SeekFrom::Start(self.offsets.vert_off))?;
        self.file.read_exact(&mut read_hash)?;
        (&mut self.file)
            .take(self.offsets.vert_len)
            .read_to_end(&mut vert_data)?;
        let mut hasher = get_hasher();
        hasher.write(&vert_data);
        let expected_hash = u64::from_le_bytes(read_hash);
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            Ok(VertexChunk::decode(vert_data).elements()?)
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted vertices",
            ))
        }
    }

    fn meshes(&mut self) -> Result<Vec<Mesh>, Error> {
        let mut read_hash = [0; HASH_SIZE];
        let mut mesh_data = Vec::with_capacity(self.offsets.mesh_len as usize);
        self.file.seek(SeekFrom::Start(self.offsets.mesh_off))?;
        self.file.read_exact(&mut read_hash)?;
        (&mut self.file)
            .take(self.offsets.mesh_len)
            .read_to_end(&mut mesh_data)?;
        let mut hasher = get_hasher();
        hasher.write(&mesh_data);
        let expected_hash = u64::from_le_bytes(read_hash);
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            Ok(MeshChunk::decode(mesh_data).elements()?)
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted meshes",
            ))
        }
    }

    fn cameras(&mut self) -> Result<Vec<Camera>, Error> {
        let mut read_hash = [0; HASH_SIZE];
        let mut cam_data = Vec::with_capacity(self.offsets.cam_len as usize);
        self.file.seek(SeekFrom::Start(self.offsets.cam_off))?;
        self.file.read_exact(&mut read_hash)?;
        (&mut self.file)
            .take(self.offsets.cam_len)
            .read_to_end(&mut cam_data)?;
        let mut hasher = get_hasher();
        hasher.write(&cam_data);
        let expected_hash = u64::from_le_bytes(read_hash);
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            Ok(CameraChunk::decode(cam_data).elements()?)
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted cameras",
            ))
        }
    }

    fn textures(&mut self) -> Result<Vec<(u16, Texture)>, Error> {
        let mut read_hash = [0; HASH_SIZE];
        let mut tex_data = Vec::with_capacity(self.offsets.tex_len as usize);
        self.file.seek(SeekFrom::Start(self.offsets.tex_off))?;
        self.file.read_exact(&mut read_hash)?;
        (&mut self.file)
            .take(self.offsets.tex_len)
            .read_to_end(&mut tex_data)?;
        let mut hasher = get_hasher();
        hasher.write(&tex_data);
        let expected_hash = u64::from_le_bytes(read_hash);
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            Ok(TextureChunk::decode(tex_data).elements()?)
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted textures",
            ))
        }
    }

    fn materials(&mut self) -> Result<Vec<(u16, Material)>, Error> {
        let mut read_hash = [0; HASH_SIZE];
        let mut mat_data = Vec::with_capacity(self.offsets.mat_len as usize);
        self.file.seek(SeekFrom::Start(self.offsets.mat_off))?;
        self.file.read_exact(&mut read_hash)?;
        (&mut self.file)
            .take(self.offsets.mat_len)
            .read_to_end(&mut mat_data)?;
        let mut hasher = get_hasher();
        hasher.write(&mat_data);
        let expected_hash = u64::from_le_bytes(read_hash);
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            Ok(MaterialChunk::decode(mat_data).elements()?)
        } else {
            Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "Corrupted materials",
            ))
        }
    }
}

trait ParsedChunk {
    type Item;
    fn encode(item: &[Self::Item]) -> Self;
    fn decode(data: Vec<u8>) -> Self;
    fn size_bytes(&self) -> usize;
    fn elements(self) -> Result<Vec<Self::Item>, Error>;
    fn data(self) -> Vec<u8>;
}

struct VertexChunk {
    data: Vec<u8>,
}

impl ParsedChunk for VertexChunk {
    type Item = Vertex;

    fn encode(item: &[Self::Item]) -> Self {
        let data = item.iter().flat_map(vertex_to_bytes).collect::<Vec<_>>();
        VertexChunk {
            data: compress(&data),
        }
    }

    fn decode(data: Vec<u8>) -> Self {
        VertexChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Result<Vec<Self::Item>, Error> {
        let decompressed = decompress(&self.data);
        Ok(decompressed.chunks_exact(32).map(bytes_to_vertex).collect())
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
        MeshChunk {
            data: compress(&data),
        }
    }

    fn decode(data: Vec<u8>) -> Self {
        MeshChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Result<Vec<Self::Item>, Error> {
        let decompressed = decompress(&self.data);
        let len = u32::from_le_bytes(decompressed[0..4].try_into().unwrap());
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = 4;
        while index < decompressed.len() {
            let encoded_len =
                u32::from_le_bytes(decompressed[index..index + 4].try_into().unwrap()) as usize;
            index += 4;
            let mesh = bytes_to_mesh(&decompressed[index..index + encoded_len]);
            index += encoded_len;
            retval.push(mesh);
        }
        Ok(retval)
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
        CameraChunk {
            data: compress(&data),
        }
    }

    fn decode(data: Vec<u8>) -> Self {
        CameraChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Result<Vec<Self::Item>, Error> {
        let decompressed = decompress(&self.data);
        let len = decompressed[0];
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = 1;
        while index < decompressed.len() {
            let encoded_len = decompressed[index] as usize;
            index += 1;
            let camera = bytes_to_camera(&decompressed[index..index + encoded_len]);
            index += encoded_len;
            retval.push(camera);
        }
        Ok(retval)
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
    let near_plane;
    let far_plane;
    let other_arg;
    match camera {
        Camera::Perspective(cam) => {
            camera_type = &[0];
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.fovx;
            near_plane = cam.near;
            far_plane = cam.far;
        }
        Camera::Orthographic(cam) => {
            camera_type = &[1];
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.scale;
            near_plane = cam.near;
            far_plane = cam.far;
        }
    }
    let pos: [f32; 3] = Point3::into(position);
    let tgt: [f32; 3] = Point3::into(target);
    let upp: [f32; 3] = Vec3::into(up);
    let oth = f32::to_le_bytes(other_arg);
    let near = f32::to_le_bytes(near_plane);
    let far = f32::to_le_bytes(far_plane);
    let cam_data_iter = pos
        .iter()
        .chain(tgt.iter())
        .chain(upp.iter())
        .copied()
        .flat_map(f32::to_le_bytes)
        .chain(oth.iter().copied())
        .chain(near.iter().copied())
        .chain(far.iter().copied());
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
    let near = f32::from_le_bytes(data[41..45].try_into().unwrap());
    let far = f32::from_le_bytes(data[45..49].try_into().unwrap());
    match cam_type {
        0 => Camera::Perspective(PerspectiveCam {
            position,
            target,
            up,
            fovx: other_val,
            near,
            far,
        }),
        1 => Camera::Orthographic(OrthographicCam {
            position,
            target,
            up,
            scale: other_val,
            near,
            far,
        }),
        _ => {
            panic!("Unexpected cam type")
        }
    }
}

struct TextureChunk {
    data: Vec<u8>,
}

impl ParsedChunk for TextureChunk {
    type Item = (u16, Texture);

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

    fn elements(self) -> Result<Vec<Self::Item>, Error> {
        let len = u16::from_le_bytes(self.data[0..2].try_into().unwrap());
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = std::mem::size_of::<u16>();
        while index < self.size_bytes() {
            let encoded_len =
                u32::from_le_bytes(self.data[index..index + 4].try_into().unwrap()) as usize;
            index += std::mem::size_of::<u32>();
            let texture = bytes_to_texture(&self.data[index..index + encoded_len])?;
            index += encoded_len;
            retval.push(texture);
        }
        Ok(retval)
    }

    fn data(self) -> Vec<u8> {
        self.data
    }
}

fn texture_to_bytes((index, texture): &(u16, Texture)) -> Vec<u8> {
    let name = texture.name();
    let str_len = name.bytes().len();
    assert!(str_len < 256);
    let (width, height) = texture.dimensions();
    let miplvls = texture.mipmap_levels();
    let (mut w_mip, mut h_mip) = (width, height);
    let mut tex_data = Vec::new();
    for level in 0..miplvls {
        let mut mip_data = Vec::new();
        let mip = texture.raw(level);
        // ad-hoc compression surely better than general purpose one
        PngEncoder::new_with_quality(
            &mut mip_data,
            image::png::CompressionType::Fast,
            image::png::FilterType::Up,
        )
        .encode(
            mip,
            w_mip as u32,
            h_mip as u32,
            texture.format().to_color_type(),
        )
        .expect("Failed to encode texture");
        w_mip >>= 1;
        h_mip >>= 1;
        tex_data.extend((mip_data.len() as u32).to_le_bytes());
        tex_data.extend(mip_data);
    }
    let tex_len = tex_data.len();
    let total_len = std::mem::size_of::<u16>() + 3 * std::mem::size_of::<u8>() + str_len + tex_len;
    let mut retval = Vec::with_capacity(total_len);
    retval.extend(index.to_le_bytes());
    retval.push(format_to_u8(texture.format()));
    retval.push(str_len as u8);
    retval.extend(name.bytes());
    retval.push(miplvls as u8);
    retval.extend(tex_data);
    retval
}

fn format_to_u8(format: TextureFormat) -> u8 {
    match format {
        TextureFormat::Gray => 1,
        TextureFormat::Rgba => 2,
    }
}

fn u8_to_format(format: u8) -> Result<TextureFormat, Error> {
    match format {
        1 => Ok(TextureFormat::Gray),
        2 => Ok(TextureFormat::Rgba),
        _ => panic!("Texture format unexpected"),
    }
}

fn bytes_to_texture(data: &[u8]) -> Result<(u16, Texture), Error> {
    let tex_index = u16::from_le_bytes(data[0..2].try_into().unwrap());
    let format = u8_to_format(data[2])?;
    let str_len = data[3] as usize;
    let mut index = 4;
    let name = String::from_utf8(data[index..index + str_len].to_vec()).unwrap();
    index += str_len;
    let miplvls = data[index] as usize;
    index += 1;
    let mut dimensions = Vec::with_capacity(miplvls);
    let mut mipmaps = Vec::with_capacity(miplvls);
    for _ in 0..miplvls {
        let miplen = u32::from_le_bytes(data[index..index + 4].try_into().unwrap()) as usize;
        index += 4;
        let decoder = PngDecoder::new(&data[index..index + miplen]).expect("Corrupted image");
        index += miplen;
        dimensions.push(decoder.dimensions());
        let mut decoded = vec![0; decoder.total_bytes() as usize];
        decoder
            .read_image(&mut decoded)
            .expect("Failed to decode image");
        mipmaps.push(decoded);
    }
    let info = TextureInfo {
        name,
        format,
        width: dimensions[0].0 as u16,
        height: dimensions[0].1 as u16,
    };
    let texture = match format {
        TextureFormat::Gray => {
            let mut data = Vec::with_capacity(miplvls);
            for (mip, (w, h)) in mipmaps.into_iter().zip(dimensions.into_iter()) {
                let conversion_error = Error::new(
                    ErrorKind::InvalidData,
                    "Failed to parse data into grayscale image",
                );
                let tex = GrayImage::from_raw(w, h, mip).ok_or(conversion_error)?;
                data.push(tex);
            }
            Texture::new_gray_with_mipmaps(info, data)
        }
        TextureFormat::Rgba => {
            let mut data = Vec::with_capacity(miplvls);
            for (mip, (w, h)) in mipmaps.into_iter().zip(dimensions.into_iter()) {
                let conversion_error = Error::new(
                    ErrorKind::InvalidData,
                    "Failed to parse data into rgba image",
                );
                let tex = RgbaImage::from_raw(w, h, mip).ok_or(conversion_error)?;
                data.push(tex);
            }
            Texture::new_rgba_with_mipmaps(info, data)
        }
    };
    Ok((tex_index, texture))
}

struct MaterialChunk {
    data: Vec<u8>,
}

impl ParsedChunk for MaterialChunk {
    type Item = (u16, Material);

    fn encode(item: &[Self::Item]) -> Self {
        let len = item.len() as u16;
        let mut data = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
        for material in item {
            let encoded = material_to_bytes(material);
            let encoded_len = encoded.len() as u32;
            data.extend(encoded_len.to_le_bytes().iter());
            data.extend(encoded);
        }
        MaterialChunk {
            data: compress(&data),
        }
    }

    fn decode(data: Vec<u8>) -> Self {
        MaterialChunk { data }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }

    fn elements(self) -> Result<Vec<Self::Item>, Error> {
        let decompressed = decompress(&self.data);
        let len = u16::from_le_bytes(decompressed[0..2].try_into().unwrap());
        let mut retval = Vec::with_capacity(len as usize);
        let mut index = std::mem::size_of::<u16>();
        while index < decompressed.len() {
            let encoded_len =
                u32::from_le_bytes(decompressed[index..index + 4].try_into().unwrap()) as usize;
            index += std::mem::size_of::<u32>();
            let material = bytes_to_material(&decompressed[index..index + encoded_len]);
            index += encoded_len;
            retval.push(material);
        }
        Ok(retval)
    }

    fn data(self) -> Vec<u8> {
        self.data
    }
}

fn material_to_bytes((index, material): &(u16, Material)) -> Vec<u8> {
    let str_len = material.name.bytes().len();
    assert!(str_len < 256);
    let total_len = std::mem::size_of::<u16>() // index
        + std::mem::size_of::<u8>() // str_len
        + str_len //actual string
        + std::mem::size_of::<u8>() // shader id
        + std::mem::size_of::<u16>() // diffuse texture id
        + 3 * std::mem::size_of::<u8>() // diffuse multiplier
        + std::mem::size_of::<u16>(); // opacity texture id
    let mut retval = Vec::with_capacity(total_len);
    retval.extend(index.to_le_bytes());
    retval.push(str_len as u8);
    retval.extend(material.name.bytes());
    retval.push(material.shader.into());
    if let Some(diffuse) = material.diffuse {
        retval.extend(diffuse.to_le_bytes());
    } else {
        retval.extend(u16::MAX.to_le_bytes());
    }
    retval.extend(&material.diffuse_mul[0..3]);
    if let Some(opacity) = material.opacity {
        retval.extend(opacity.to_le_bytes());
    } else {
        retval.extend(u16::MAX.to_le_bytes());
    }
    retval
}

fn bytes_to_material(data: &[u8]) -> (u16, Material) {
    let mat_index = u16::from_le_bytes(data[0..2].try_into().unwrap());
    let str_len = data[2] as usize;
    let mut index = 3;
    let name = String::from_utf8(data[index..index + str_len].to_vec()).unwrap();
    index += str_len;
    let shader_id = data[index];
    index += 1;
    let diffuse_id = u16::from_le_bytes(data[index..index + 2].try_into().unwrap());
    index += 2;
    let diffuse = if diffuse_id != u16::MAX {
        Some(diffuse_id)
    } else {
        None
    };
    let diffuse_mul = [data[index], data[index + 1], data[index + 2]];
    index += 3;
    let opacity_id = u16::from_le_bytes(data[index..index + 2].try_into().unwrap());
    let opacity = if opacity_id != u16::MAX {
        Some(opacity_id)
    } else {
        None
    };
    let material = Material {
        name,
        shader: shader_id.into(),
        diffuse,
        diffuse_mul,
        opacity,
    };
    (mat_index, material)
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_to_camera, bytes_to_mesh, bytes_to_texture, bytes_to_vertex, camera_to_bytes,
        compress, decompress, mesh_to_bytes, texture_to_bytes, vertex_to_bytes,
    };
    use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
    use crate::materials::{TextureFormat, TextureInfo};
    use crate::parser::v1::{bytes_to_material, material_to_bytes, HASH_SIZE};
    use crate::parser::{parse, ParserVersion, HEADER_LEN};
    use crate::{serialize, Material, ShaderMat, Texture};
    use cgmath::{Matrix4, Point3, Vector2 as Vec2, Vector3 as Vec3};
    use image::GenericImageView;
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
            let near = rng.gen_range(0.001..1.0);
            let far = rng.gen_range(10.0..10000.0);
            let cam = match cam_type {
                1 => Camera::Perspective(PerspectiveCam {
                    position,
                    target,
                    up,
                    fovx: other_arg,
                    near,
                    far,
                }),
                2 => Camera::Orthographic(OrthographicCam {
                    position,
                    target,
                    up,
                    scale: other_arg,
                    near,
                    far,
                }),
                _ => {
                    panic!("Non existing variant");
                }
            };
            buffer.push(cam);
        }
        buffer
    }

    fn gen_textures(count: u16, seed: u64) -> Vec<(u16, Texture)> {
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
            let format = if rng.gen_bool(0.5) {
                TextureFormat::Gray
            } else {
                TextureFormat::Rgba
            };
            let name = Xoshiro128StarStar::seed_from_u64(rng.gen())
                .sample_iter(&Alphanumeric)
                .take(rng.gen_range(0..255))
                .map(char::from)
                .collect::<String>();
            let info = TextureInfo {
                name,
                width: cur_img.width() as u16,
                height: cur_img.height() as u16,
                format,
            };
            let texture = match format {
                TextureFormat::Gray => Texture::new_gray(info, cur_img.into_luma8()),
                TextureFormat::Rgba => Texture::new_rgba(info, cur_img.into_rgba8()),
            };
            data.push((i, texture));
        }
        data.into_iter().collect()
    }

    fn gen_materials(count: u16, seed: u64) -> Vec<(u16, Material)> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut data = Vec::with_capacity(count as usize);
        for i in 0..count {
            let shaders = ShaderMat::all_values();
            let shader = shaders[rng.gen_range(0..shaders.len())];
            let diffuse = if rng.gen_bool(0.1) {
                Some(rng.gen_range(0..u16::MAX - 1))
            } else {
                None
            };
            let name = Xoshiro128StarStar::seed_from_u64(rng.gen())
                .sample_iter(&Alphanumeric)
                .take(rng.gen_range(0..255))
                .map(char::from)
                .collect::<String>();
            let diffuse_mul = [
                rng.gen_range(0..255),
                rng.gen_range(0..255),
                rng.gen_range(0..255),
            ];
            let opacity = if rng.gen_bool(0.01) {
                Some(rng.gen_range(0..u16::MAX - 1))
            } else {
                None
            };
            let material = Material {
                name,
                shader,
                diffuse,
                diffuse_mul,
                opacity,
            };
            data.push((i, material));
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
        let textures = gen_textures(4, 0xE59CCF79D9AD3162);
        for texture in textures {
            let data = texture_to_bytes(&texture);
            let decoded = bytes_to_texture(&data).unwrap();
            assert_eq!(decoded, texture);
        }
    }

    #[test]
    fn encode_decode_materials() {
        let materials = gen_materials(2048, 0xC7F8CE22512B15FB);
        for material in materials {
            let data = material_to_bytes(&material);
            let decoded = bytes_to_material(&data);
            assert_eq!(decoded, material);
        }
    }

    #[test]
    fn write_and_read_only_vert() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000, 0xBE8AE7F7E3A5248E);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_vertices.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &vertices,
            &[],
            &[],
            &[],
            &[],
        )?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_vertices = read.vertices()?;
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
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &meshes,
            &[],
            &[],
            &[],
        )?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_meshes = read.meshes()?;
        assert_eq!(read_meshes.len(), meshes.len());
        for i in 0..read_meshes.len() {
            let val = &read_meshes[i];
            let expected = &meshes[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_cameras() -> Result<(), std::io::Error> {
        let cameras = gen_cameras(32, 0xCC6AD9820F396116);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_cameras.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &[],
            &cameras,
            &[],
            &[],
        )?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_cameras = read.cameras()?;
        assert_eq!(read_cameras.len(), cameras.len());
        for i in 0..read_cameras.len() {
            let val = &read_cameras[i];
            let expected = &cameras[i];
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_textures() -> Result<(), std::io::Error> {
        let textures = gen_textures(2, 0x50DFC0EA9BF6E9BE);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_textures.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &[],
            &[],
            &textures,
            &[],
        )?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_textures = read.textures()?;
        assert_eq!(read_textures.len(), textures.len());
        for i in 0..read_textures.len() {
            let val = &read_textures.get(i).unwrap();
            let expected = &textures.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_materials() -> Result<(), std::io::Error> {
        let materials = gen_materials(512, 0xCEF1587343C7486C);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_materials.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &[],
            &[],
            &[],
            &materials,
        )?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_materials = read.materials()?;
        assert_eq!(read_materials.len(), materials.len());
        for i in 0..read_materials.len() {
            let val = &read_materials.get(i).unwrap();
            let expected = &materials.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn corrupted_offset() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000, 0x62A9F273AF56253C);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_off.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &vertices,
            &[],
            &[],
            &[],
            &[],
        )?;
        let read_ok = parse(file.as_path());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start((HEADER_LEN + HASH_SIZE + 4) as u64))?;
            file.write(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let read_corrupted = parse(file.as_path());
        remove_file(file.as_path())?;
        assert!(read_ok.is_ok());
        assert!(read_corrupted.is_err());
        Ok(())
    }

    #[test]
    fn compress_decompress() {
        let data = "The quick brown fox jumps over the lazy dog";
        let compressed = compress(data.as_bytes());
        let decompressed = decompress(&compressed);
        assert_ne!(&compressed, &decompressed);
        let result = String::from_utf8(decompressed).unwrap();
        assert_eq!(result, data);
    }
}
