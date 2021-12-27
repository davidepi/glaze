use super::{ParsedScene, HEADER_LEN};
use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
use crate::materials::{TextureFormat, TextureInfo};
use crate::{Material, Texture};
use cgmath::{Matrix4, Point3, Vector2 as Vec2, Vector3 as Vec3};
use fnv::FnvHashMap;
use image::png::{CompressionType, FilterType, PngDecoder, PngEncoder};
use image::{GrayImage, ImageDecoder, RgbaImage};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use std::convert::TryInto;
use std::hash::Hasher;
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use twox_hash::XxHash64;
use xz2::read::{XzDecoder, XzEncoder};

/*-------------------------------------- FILE STRUCTURE ----------------------------------------.
| The initial header is handled by the parse() function and ignored by this module.             |
| After the header, a HASH_SIZE byte hash is read. This is the hash of the OffsetsTable         |
| structure.                                                                                    |
|                                                                                               |
| The OffsetsTable structure has a dynamic size. After its hash of HASH_SIZE bytes, there is a  |
| single byte stating how many chunks there are in the file. Then for each chunk in the file    |
| 17 bytes shows its position in the file. The first byte is the ChunkID. Then 8 bytes for its  |
| offset from the beginning of the file, then 8 bytes for its length (in bytes).                |
| This concludes the OffsetsTable.                                                              |
|                                                                                               |
| Now, each chunk can then be read independently. The offset found for this specific chunk      |
| points to the beginning of the chunk. Each chunk is composed of HASH_SIZE bytes of hash       |
| followed by the chunk data itself. The HASH_SIZE bytes are included in the chunk len count    |
|                                                                                               |
| If an unrecognized chunk is encountered, no error is emitted. This is the expected behaviour  |
\.---------------------------------------------------------------------------------------------*/

/// Seed for the hasher used in this file format.
const HASHER_SEED: u64 = 0x368262AAA1DEB64D;
/// Length of each hash used in this file format.
const HASH_SIZE: usize = std::mem::size_of::<u64>();

/// Returns the hasher used by this file format.
fn get_hasher() -> impl Hasher {
    XxHash64::with_seed(HASHER_SEED)
}

/// Compress some data using lzma with compression level 9
fn compress<R: Read>(data: R) -> Vec<u8> {
    let mut compressed = Vec::new();
    let mut encoder = XzEncoder::new(data, 9);
    encoder
        .read_to_end(&mut compressed)
        .expect("Failed to compress data");
    compressed
}

/// Decompress some data using lzma
fn decompress<R: Read>(data: R) -> Vec<u8> {
    let mut decoder = XzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .expect("Failed to decompress data");
    decompressed
}

/// Assigns an unique ID to the chunk contained in the OffsetsTable
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ChunkID {
    Vertex = 0,
    Mesh = 1,
    Camera = 2,
    Texture = 3,
    Material = 4,
}

impl TryFrom<u8> for ChunkID {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ChunkID::Vertex),
            1 => Ok(ChunkID::Mesh),
            2 => Ok(ChunkID::Camera),
            3 => Ok(ChunkID::Texture),
            4 => Ok(ChunkID::Material),
            _ => Err(Error::new(ErrorKind::Unsupported, "Unsupported chunk")),
        }
    }
}

impl From<ChunkID> for u8 {
    fn from(ci: ChunkID) -> Self {
        match ci {
            ChunkID::Vertex => 0,
            ChunkID::Mesh => 1,
            ChunkID::Camera => 2,
            ChunkID::Texture => 3,
            ChunkID::Material => 4,
        }
    }
}

/// Stores the list of chunks contained in this file, along with their position in the file.
///WARNING: offsets in this structure are relative to the offset table position. Their actual
///WARNING: offset is adjusted and made absolute in the file upon calling OffsetsTable::as_bytes()
#[derive(Default)]
struct OffsetsTable {
    /// Next available offset in the file
    next_chunk: u64,
    /// Chunks in the file <ChunkID, (Offset relative to the end of this struct, length in bytes)>
    chunks: FnvHashMap<ChunkID, (u64, u64)>,
}

impl OffsetsTable {
    /// Amount of bytes for each serialized chunk
    const SERIALIZED_CHUNK_LEN: usize = 17;

    /// Reads the offsets structure from the file and parse it.
    fn seek_and_parse<R: Read + Seek>(file: &mut R) -> Result<OffsetsTable, Error> {
        file.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        let mut expected_hash = [0; HASH_SIZE];
        file.read_exact(&mut expected_hash)?;
        let expected_hash = u64::from_le_bytes(expected_hash);
        let mut chunks_no = [0];
        file.read_exact(&mut chunks_no[..])?;
        let chunks_no = chunks_no[0] as usize;
        file.seek(SeekFrom::Current(-1))?;
        let bytes_size = 1 + chunks_no * Self::SERIALIZED_CHUNK_LEN; // chunkid(1) + offset(8) + len(8)
        let mut buffer = Vec::with_capacity(bytes_size);
        file.take(bytes_size as u64).read_to_end(&mut buffer)?;
        let mut hasher = get_hasher();
        hasher.write(&buffer);
        let actual_hash = hasher.finish();
        if expected_hash == actual_hash {
            let mut map = FnvHashMap::with_capacity_and_hasher(chunks_no, Default::default());
            let mut next_chunk = 0;
            for chunk in buffer[1..].chunks_exact(Self::SERIALIZED_CHUNK_LEN) {
                if let Ok(id) = ChunkID::try_from(chunk[0]) {
                    let offset = u64::from_le_bytes(chunk[1..9].try_into().unwrap());
                    let len = u64::from_le_bytes(chunk[9..].try_into().unwrap());
                    //TODO: what to do in case of duplicates? :thinking_emoji:
                    next_chunk = offset + len;
                    map.insert(id, (offset, len));
                } else {
                    log::warn!("Found an unsupported chunk. ID was {}.", chunk[0]);
                }
            }
            let retval = OffsetsTable {
                next_chunk,
                chunks: map,
            };
            Ok(retval)
        } else {
            Err(Error::new(
                ErrorKind::InvalidData,
                "Corrupted file structure",
            ))
        }
    }

    /// Converts the offsets structure into an array of bytes.
    fn as_bytes(&self) -> Vec<u8> {
        let chunks_no = self.chunks.len();
        let mylen = (HASH_SIZE + 1 + chunks_no * Self::SERIALIZED_CHUNK_LEN) as u64;
        let mut bytes = Vec::with_capacity(mylen as usize);
        bytes.extend([0; HASH_SIZE]);
        bytes.push(chunks_no as u8);
        for (id, (offset, len)) in self.chunks.iter() {
            bytes.push(u8::from(*id));
            let adjusted_offset = HEADER_LEN as u64 + mylen + *offset;
            bytes.extend(u64::to_le_bytes(adjusted_offset));
            bytes.extend(u64::to_le_bytes(*len));
        }
        let mut hasher = get_hasher();
        hasher.write(&bytes[HASH_SIZE..]);
        let hash = hasher.finish();
        bytes.splice(..HASH_SIZE, hash.to_le_bytes());
        bytes
    }

    /// Writes the offset of a chunk WITHOUT considering the offset table len
    fn set_offset(&mut self, id: ChunkID, len: u64) {
        if len > 0 {
            let offset = self.next_chunk;
            self.chunks.insert(id, (offset, len));
            self.next_chunk = offset + len;
        }
    }
}

/// Parser for this file format.
pub(super) struct ContentV1<R: Read + Seek> {
    /// Handle to the Reader
    file: R,
    /// Offsets of this particular file.
    offsets: OffsetsTable,
}

impl<R: Read + Seek> ContentV1<R> {
    /// Initializes the parser for this particular file format.
    pub(super) fn parse(mut file: R) -> Result<Self, Error> {
        let offsets = OffsetsTable::seek_and_parse(&mut file)?;
        Ok(ContentV1 { file, offsets })
    }

    /// Writes the scene structures into the file handled by this parser.
    pub(super) fn serialize<W: Write + Seek>(
        mut fout: W,
        vertices: &[Vertex],
        meshes: &[Mesh],
        cameras: &[Camera],
        textures: &[(u16, Texture)],
        materials: &[(u16, Material)],
    ) -> Result<(), Error> {
        let vert = Chunk::from_vertices(vertices);
        let mesh = Chunk::from_meshes(meshes);
        let cams = Chunk::from_cameras(cameras);
        let texs = Chunk::from_textures(textures);
        let mats = Chunk::from_materials(materials);
        let mut offsets = OffsetsTable::default();
        offsets.set_offset(ChunkID::Vertex, vert.data.len() as u64);
        offsets.set_offset(ChunkID::Mesh, mesh.data.len() as u64);
        offsets.set_offset(ChunkID::Camera, cams.data.len() as u64);
        offsets.set_offset(ChunkID::Texture, texs.data.len() as u64);
        offsets.set_offset(ChunkID::Material, mats.data.len() as u64);
        fout.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        fout.write_all(&offsets.as_bytes())?;
        fout.write_all(&vert.data)?;
        fout.write_all(&mesh.data)?;
        fout.write_all(&cams.data)?;
        fout.write_all(&texs.data)?;
        fout.write_all(&mats.data)?;
        Ok(())
    }

    /// Reads a chunk with a given ID from the file.
    fn read_chunk(&mut self, id: ChunkID) -> Result<Chunk, Error> {
        let chunk = if let Some((offset, len)) = self.offsets.chunks.get(&id) {
            self.file.seek(SeekFrom::Start(*offset))?;
            let mut read = Vec::with_capacity(*len as usize);
            (&mut self.file).take(*len).read_to_end(&mut read)?;
            Chunk { data: read }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        };
        Ok(chunk)
    }
}

impl<R: Read + Seek> ParsedScene for ContentV1<R> {
    fn vertices(&mut self) -> Result<Vec<Vertex>, Error> {
        self.read_chunk(ChunkID::Vertex)?.to_vertices()
    }

    fn meshes(&mut self) -> Result<Vec<Mesh>, Error> {
        self.read_chunk(ChunkID::Mesh)?.to_meshes()
    }

    fn cameras(&mut self) -> Result<Vec<Camera>, Error> {
        self.read_chunk(ChunkID::Camera)?.to_cameras()
    }

    fn textures(&mut self) -> Result<Vec<(u16, Texture)>, Error> {
        self.read_chunk(ChunkID::Texture)?.to_textures()
    }

    fn materials(&mut self) -> Result<Vec<(u16, Material)>, Error> {
        self.read_chunk(ChunkID::Material)?.to_materials()
    }
}

/// Calculates the hash of a given amount of bytes and preprends to them.
fn prepend_hash(mut input_data: Vec<u8>) -> Vec<u8> {
    let mut hasher = get_hasher();
    hasher.write(&input_data);
    let hash = hasher.finish();
    input_data.splice(..0, hash.to_le_bytes());
    input_data
}

/// Verifies that the first HASH_SIZE bytes of the given input data correspond to the hash of the
/// remaining data.
/// If this is true, returns the input data WITHOUT the hash, otherwise returns None.
fn verify_hash(input_data: Vec<u8>) -> Option<Vec<u8>> {
    let bytes = input_data[0..HASH_SIZE].try_into().unwrap();
    let remaining = &input_data[HASH_SIZE..];
    let expected_hash = u64::from_le_bytes(bytes);
    let mut hasher = get_hasher();
    hasher.write(remaining);
    let actual_hash = hasher.finish();
    if expected_hash == actual_hash {
        Some(remaining.to_vec())
    } else {
        None
    }
}

/// A Chunk containing some data (may be vertices, meshes, cameras, who knows).
struct Chunk {
    data: Vec<u8>,
}

impl Chunk {
    /// Creates a chunk from a slice of vertices.
    fn from_vertices(items: &[Vertex]) -> Self {
        if !items.is_empty() {
            let uncompressed = items.iter().flat_map(vertex_to_bytes).collect::<Vec<u8>>();
            let compressed = compress(&uncompressed[..]);
            Chunk {
                data: prepend_hash(compressed),
            }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        }
    }

    /// Reinterprets the bytes of this chunks as a vector of vertices.
    #[allow(clippy::wrong_self_convention)] // this is private, who cares
    fn to_vertices(self) -> Result<Vec<Vertex>, Error> {
        if !self.data.is_empty() {
            if let Some(verified_data) = verify_hash(self.data) {
                let decompressed = decompress(&verified_data[..]);
                Ok(decompressed.chunks_exact(32).map(bytes_to_vertex).collect())
            } else {
                Err(Error::new(ErrorKind::InvalidData, "Corrupted vertices"))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Creates a chunk from a slice of meshes.
    fn from_meshes(items: &[Mesh]) -> Self {
        if !items.is_empty() {
            let len = items.len() as u32;
            let mut uncompressed = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
            for mesh in items {
                let encoded = mesh_to_bytes(mesh);
                let encoded_len = encoded.len() as u32;
                uncompressed.extend(encoded_len.to_le_bytes().iter());
                uncompressed.extend(encoded);
            }
            let compressed = compress(&uncompressed[..]);
            Chunk {
                data: prepend_hash(compressed),
            }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        }
    }

    /// Reinterprets the bytes of this chunks as a vector of meshes.
    #[allow(clippy::wrong_self_convention)] // this is private, who cares
    fn to_meshes(self) -> Result<Vec<Mesh>, Error> {
        if !self.data.is_empty() {
            if let Some(verified_data) = verify_hash(self.data) {
                let decompressed = decompress(&verified_data[..]);
                let len = u32::from_le_bytes(decompressed[0..4].try_into().unwrap());
                let mut mesh_bytes = Vec::with_capacity(len as usize);
                let mut index = 4;
                while index < decompressed.len() {
                    let encoded_len =
                        u32::from_le_bytes(decompressed[index..index + 4].try_into().unwrap())
                            as usize;
                    index += 4;
                    let mesh = &decompressed[index..index + encoded_len];
                    index += encoded_len;
                    mesh_bytes.push(mesh);
                }
                let retval = mesh_bytes.into_par_iter().map(bytes_to_mesh).collect();
                Ok(retval)
            } else {
                Err(Error::new(ErrorKind::InvalidData, "Corrupted meshes"))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Creates a chunk from a slice of cameras.
    fn from_cameras(items: &[Camera]) -> Self {
        if !items.is_empty() {
            let mut uncompressed = Vec::with_capacity(items.len());
            uncompressed.push(items.len() as u8);
            for camera in items {
                let encoded = camera_to_bytes(camera);
                let encoded_len = encoded.len() as u8;
                uncompressed.push(encoded_len);
                uncompressed.extend(encoded);
            }
            let compressed = compress(&uncompressed[..]);
            Chunk {
                data: prepend_hash(compressed),
            }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        }
    }

    /// Reinterprets the bytes of this chunks as a vector of cameras.
    #[allow(clippy::wrong_self_convention)] // this is private, who cares
    fn to_cameras(self) -> Result<Vec<Camera>, Error> {
        if !self.data.is_empty() {
            if let Some(verified) = verify_hash(self.data) {
                let decompressed = decompress(&verified[..]);
                let len = decompressed[0];
                let mut retval = Vec::with_capacity(len as usize);
                let mut index = 1;
                while index < decompressed.len() {
                    let encoded_len = decompressed[index] as usize;
                    index += 1;
                    let camera = bytes_to_camera(&decompressed[index..(index + encoded_len)]);
                    index += encoded_len;
                    retval.push(camera);
                }
                Ok(retval)
            } else {
                Err(Error::new(ErrorKind::InvalidData, "Corrupted cameras"))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Creates a chunk from a slice of textures.
    fn from_textures(items: &[(u16, Texture)]) -> Self {
        if !items.is_empty() {
            let len = items.len() as u16;
            let mut uncompressed = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
            for texture in items {
                let encoded = texture_to_bytes(texture);
                let encoded_len = encoded.len() as u32;
                uncompressed.extend(encoded_len.to_le_bytes().iter());
                uncompressed.extend(encoded);
            }
            // textures are already compressed as part of the texture_to_bytes procedure
            Chunk {
                data: prepend_hash(uncompressed),
            }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        }
    }

    /// Reinterprets the bytes of this chunks as a vector of textures.
    #[allow(clippy::wrong_self_convention)] // this is private, who cares
    fn to_textures(self) -> Result<Vec<(u16, Texture)>, Error> {
        if !self.data.is_empty() {
            if let Some(verified) = verify_hash(self.data) {
                let len = u16::from_le_bytes(verified[0..2].try_into().unwrap());
                let mut tex_bytes = Vec::with_capacity(len as usize);
                let mut index = std::mem::size_of::<u16>();
                while index < verified.len() {
                    let encoded_len =
                        u32::from_le_bytes(verified[index..index + 4].try_into().unwrap()) as usize;
                    index += std::mem::size_of::<u32>();
                    let bytes = &verified[index..index + encoded_len];
                    index += encoded_len;
                    tex_bytes.push(bytes);
                }
                let retval = tex_bytes
                    .into_par_iter()
                    .map(bytes_to_texture)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(retval)
            } else {
                Err(Error::new(ErrorKind::InvalidData, "Corrupted textures"))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Creates a chunk from a slice of materials.
    fn from_materials(items: &[(u16, Material)]) -> Self {
        if !items.is_empty() {
            let len = items.len() as u16;
            let mut uncompressed = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
            for material in items {
                let encoded = material_to_bytes(material);
                let encoded_len = encoded.len() as u32;
                uncompressed.extend(encoded_len.to_le_bytes().iter());
                uncompressed.extend(encoded);
            }
            let compressed = compress(&uncompressed[..]);
            Chunk {
                data: prepend_hash(compressed),
            }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        }
    }

    /// Reinterprets the bytes of this chunks as a vector of materials.
    #[allow(clippy::wrong_self_convention)] // this is private, who cares
    fn to_materials(self) -> Result<Vec<(u16, Material)>, Error> {
        if !self.data.is_empty() {
            if let Some(verified) = verify_hash(self.data) {
                let decompressed = decompress(&verified[..]);
                let len = u16::from_le_bytes(decompressed[0..2].try_into().unwrap());
                let mut retval = Vec::with_capacity(len as usize);
                let mut index = std::mem::size_of::<u16>();
                while index < decompressed.len() {
                    let encoded_len =
                        u32::from_le_bytes(decompressed[index..index + 4].try_into().unwrap())
                            as usize;
                    index += std::mem::size_of::<u32>();
                    let material = bytes_to_material(&decompressed[index..index + encoded_len]);
                    index += encoded_len;
                    retval.push(material);
                }
                Ok(retval)
            } else {
                Err(Error::new(ErrorKind::InvalidData, "Corrupted materials"))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }
}

/// Converts a Vertex to a vector of bytes.
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

/// Converts a vector of bytes to a Vertex.
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

/// Converts a Mesh to a vector of bytes.
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

/// Converts a vector of bytes to a Mesh.
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

/// Converts a Camera to a vector of bytes.
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

/// Converts a vector of bytes to a Camera.
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

/// Converts a Texture to a vector of bytes.
fn texture_to_bytes((index, texture): &(u16, Texture)) -> Vec<u8> {
    let name = texture.name();
    let str_len = name.bytes().len();
    assert!(str_len < 256);
    let miplvls = texture.mipmap_levels();
    let mut tex_data = Vec::new();
    for level in 0..miplvls {
        let mut mip_data = Vec::new();
        let (w_mip, h_mip) = texture.dimensions(level);
        let mip = texture.raw(level);
        // ad-hoc compression surely better than general purpose one
        PngEncoder::new_with_quality(&mut mip_data, CompressionType::Fast, FilterType::Up)
            .encode(
                mip,
                w_mip as u32,
                h_mip as u32,
                texture.format().to_color_type(),
            )
            .expect("Failed to encode texture");
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

/// Converts a vector of bytes to a Texture.
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

/// Converts a Material to a vector of bytes.
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

/// Converts a vector of bytes to a Material.
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
    fn write_and_read_everything() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0x98DA1392A52639C2);
        let meshes = gen_meshes(100, 0xEFB101FDF7F185FB);
        let cameras = gen_cameras(100, 0x9B20F550F7EDF740);
        let textures = gen_textures(4, 0x122C4C9E1A51AC4B);
        let materials = gen_materials(100, 0x76D7971188303D82);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_everything.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &vertices,
            &meshes,
            &cameras,
            &textures,
            &materials,
        )?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_vertices = read.vertices()?;
        let read_meshes = read.meshes()?;
        let read_cameras = read.cameras()?;
        let read_textures = read.textures()?;
        let read_materials = read.materials()?;
        assert_eq!(read_vertices.len(), vertices.len());
        assert_eq!(read_meshes.len(), meshes.len());
        assert_eq!(read_cameras.len(), cameras.len());
        assert_eq!(read_textures.len(), textures.len());
        assert_eq!(read_materials.len(), materials.len());
        for i in 0..read_vertices.len() {
            let val = &read_vertices.get(i).unwrap();
            let expected = &vertices.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_meshes.len() {
            let val = &read_meshes.get(i).unwrap();
            let expected = &meshes.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_cameras.len() {
            let val = &read_cameras.get(i).unwrap();
            let expected = &cameras.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_textures.len() {
            let val = &read_textures.get(i).unwrap();
            let expected = &textures.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_materials.len() {
            let val = &read_materials.get(i).unwrap();
            let expected = &materials.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn corrupted_offset() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0x8794A1E593281F2F);
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
        assert!(read_ok.is_ok());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start((HEADER_LEN + HASH_SIZE + 10) as u64))?;
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let read_corrupted = parse(file.as_path());
        remove_file(file.as_path())?;
        assert!(read_corrupted.is_err());
        Ok(())
    }

    #[test]
    fn corrupted_vertices() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0x62A9F273AF56253C);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_vert.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &vertices,
            &[],
            &[],
            &[],
            &[],
        )?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.vertices().is_ok());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start(1000))?;
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let mut read_corrupted = parse(file.as_path()).unwrap();
        remove_file(file.as_path())?;
        assert!(read_corrupted.vertices().is_err());
        Ok(())
    }

    #[test]
    fn corrupted_meshes() -> Result<(), std::io::Error> {
        let meshes = gen_meshes(100, 0x9EB228E1A9586DD2);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_mesh.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &meshes,
            &[],
            &[],
            &[],
        )?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.meshes().is_ok());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start(1000))?;
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let mut read_corrupted = parse(file.as_path()).unwrap();
        remove_file(file.as_path())?;
        assert!(read_corrupted.meshes().is_err());
        Ok(())
    }

    #[test]
    fn corrupted_cameras() -> Result<(), std::io::Error> {
        let cameras = gen_cameras(100, 0x0090DE663C0E2450);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_cam.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &[],
            &cameras,
            &[],
            &[],
        )?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.cameras().is_ok());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start(1000))?;
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let mut read_corrupted = parse(file.as_path()).unwrap();
        remove_file(file.as_path())?;
        assert!(read_corrupted.cameras().is_err());
        Ok(())
    }

    #[test]
    fn corrupted_textures() -> Result<(), std::io::Error> {
        let textures = gen_textures(4, 0x8D1B42A77650F752);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_textures.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &[],
            &[],
            &textures,
            &[],
        )?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.textures().is_ok());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start(1000))?;
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let mut read_corrupted = parse(file.as_path()).unwrap();
        remove_file(file.as_path())?;
        assert!(read_corrupted.textures().is_err());
        Ok(())
    }

    #[test]
    fn corrupted_materials() -> Result<(), std::io::Error> {
        let materials = gen_materials(100, 0xA7446EB290EF428F);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_materials.bin");
        serialize(
            file.as_path(),
            ParserVersion::V1,
            &[],
            &[],
            &[],
            &[],
            &materials,
        )?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.materials().is_ok());
        {
            //corrupt file
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(file.as_path())?;
            file.seek(SeekFrom::Start(1000))?;
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        }
        let mut read_corrupted = parse(file.as_path()).unwrap();
        remove_file(file.as_path())?;
        assert!(read_corrupted.materials().is_err());
        Ok(())
    }

    #[test]
    fn compress_decompress() {
        let data = "The quick brown fox jumps over the lazy dog";
        let compressed = compress(data.as_bytes());
        let decompressed = decompress(&compressed[..]);
        assert_ne!(&compressed, &decompressed);
        let result = String::from_utf8(decompressed).unwrap();
        assert_eq!(result, data);
    }
}
