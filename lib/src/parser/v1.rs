use super::{write_header, Meta, ParsedScene, HEADER_LEN};
use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
use crate::materials::{TextureFormat, TextureInfo};
use crate::{Light, Material, MeshInstance, Metal, ShaderMat, Spectrum, Texture, Transform};
use cgmath::{Point2, Point3, Vector3 as Vec3};
use fnv::FnvHashMap;
use image::png::{CompressionType, FilterType, PngDecoder, PngEncoder};
use image::{GrayImage, ImageDecoder, RgbaImage};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use std::convert::TryInto;
use std::fs::File;
use std::hash::Hasher;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::slice::from_ref;
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
    Transform = 5,
    Instance = 6,
    Light = 7,
    Meta = 250,
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
            5 => Ok(ChunkID::Transform),
            6 => Ok(ChunkID::Instance),
            7 => Ok(ChunkID::Light),
            250 => Ok(ChunkID::Meta),
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
            ChunkID::Transform => 5,
            ChunkID::Instance => 6,
            ChunkID::Light => 7,
            ChunkID::Meta => 250,
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
pub(super) struct ContentV1 {
    /// Handle to the BufferedReader
    reader: BufReader<File>,
    /// Path of the file contained in the reader (used for reopenings),
    filepath: PathBuf,
    /// Offsets of this particular file.
    offsets: OffsetsTable,
}

impl ContentV1 {
    /// Initializes the parser for this particular file format.
    pub(super) fn parse<P: AsRef<Path>>(path: P, file: File) -> Result<Self, Error> {
        let mut reader = BufReader::new(file);
        let offsets = OffsetsTable::seek_and_parse(&mut reader)?;
        Ok(ContentV1 {
            reader,
            filepath: PathBuf::from(path.as_ref()),
            offsets,
        })
    }

    /// Writes the scene structures into the file handled by this parser.
    pub(super) fn serialize<W: Write + Seek>(
        mut fout: W,
        vertices: &[Vertex],
        meshes: &[Mesh],
        transforms: &[Transform],
        instances: &[MeshInstance],
        cameras: &[Camera],
        textures: &[Texture],
        materials: &[Material],
        lights: &[Light],
        meta: Option<&Meta>,
    ) -> Result<(), Error> {
        let mut chunks = vec![
            (
                ChunkID::Vertex,
                Chunk::encode_fixed(vertices, vertex_to_bytes),
            ),
            (ChunkID::Mesh, Chunk::encode_dynamic(meshes, mesh_to_bytes)),
            (
                ChunkID::Camera,
                Chunk::encode_fixed(cameras, camera_to_bytes),
            ),
            (ChunkID::Texture, Chunk::encode_textures(textures)),
            (
                ChunkID::Material,
                Chunk::encode_dynamic(materials, material_to_bytes),
            ),
            (
                ChunkID::Transform,
                Chunk::encode_fixed(transforms, transform_to_bytes),
            ),
            (
                ChunkID::Instance,
                Chunk::encode_fixed(instances, instance_to_bytes),
            ),
            (
                ChunkID::Light,
                Chunk::encode_dynamic(lights, light_to_bytes),
            ),
        ];
        if let Some(meta) = meta {
            chunks.push((
                ChunkID::Meta,
                Chunk::encode_fixed(from_ref(meta), meta_to_bytes),
            ));
        }
        ContentV1::write_chunks(&mut fout, &chunks)
    }

    /// Writes all the chunks to the file. This method **overwrites** existing chunks.
    fn write_chunks<W: Write + Seek>(
        fout: &mut W,
        chunks: &[(ChunkID, Chunk)],
    ) -> Result<(), Error> {
        let mut tab = OffsetsTable::default();
        chunks
            .iter()
            .for_each(|(id, chunk)| tab.set_offset(*id, chunk.data.len() as u64));
        fout.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        fout.write_all(&tab.as_bytes())?;
        chunks
            .iter()
            .map(|(_, chunk)| fout.write_all(&chunk.data))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(())
    }

    /// Reads a chunk with a given ID from the file.
    fn read_chunk(&mut self, id: ChunkID) -> Result<Chunk, Error> {
        let chunk = if let Some((offset, len)) = self.offsets.chunks.get(&id) {
            self.reader.seek(SeekFrom::Start(*offset))?;
            let mut read = Vec::with_capacity(*len as usize);
            (&mut self.reader).take(*len).read_to_end(&mut read)?;
            Chunk { data: read }
        } else {
            Chunk {
                data: Vec::with_capacity(0),
            }
        };
        Ok(chunk)
    }
}

impl ParsedScene for ContentV1 {
    fn vertices(&mut self) -> Result<Vec<Vertex>, Error> {
        self.read_chunk(ChunkID::Vertex)?
            .decode_fixed(bytes_to_vertex, "Vertex")
    }

    fn meshes(&mut self) -> Result<Vec<Mesh>, Error> {
        self.read_chunk(ChunkID::Mesh)?
            .decode_dynamic(bytes_to_mesh, "Mesh")
    }

    fn transforms(&mut self) -> Result<Vec<Transform>, Error> {
        self.read_chunk(ChunkID::Transform)?
            .decode_fixed(bytes_to_transform, "Transform")
    }

    fn instances(&mut self) -> Result<Vec<MeshInstance>, Error> {
        self.read_chunk(ChunkID::Instance)?
            .decode_fixed(bytes_to_instance, "Instance")
    }
    fn cameras(&mut self) -> Result<Vec<Camera>, Error> {
        self.read_chunk(ChunkID::Camera)?
            .decode_fixed(bytes_to_camera, "Camera")
    }

    fn textures(&mut self) -> Result<Vec<Texture>, Error> {
        self.read_chunk(ChunkID::Texture)?.decode_textures()
    }

    fn materials(&mut self) -> Result<Vec<Material>, Error> {
        self.read_chunk(ChunkID::Material)?
            .decode_dynamic(bytes_to_material, "Material")
    }

    fn lights(&mut self) -> Result<Vec<Light>, Error> {
        self.read_chunk(ChunkID::Light)?
            .decode_dynamic(bytes_to_light, "Light")
    }

    fn meta(&mut self) -> Result<Meta, Error> {
        match self
            .read_chunk(ChunkID::Meta)?
            .decode_fixed(bytes_to_meta, "Meta")
        {
            Ok(mut m) => Ok(m.pop().unwrap()),
            Err(e) => Err(e),
        }
    }

    fn update(
        &mut self,
        cameras: Option<&[Camera]>,
        materials: Option<&[Material]>,
        lights: Option<&[Light]>,
        meta: Option<&Meta>,
    ) -> Result<(), Error> {
        let vertices = self.read_chunk(ChunkID::Vertex)?;
        let meshes = self.read_chunk(ChunkID::Mesh)?;
        let textures = self.read_chunk(ChunkID::Texture)?;
        let transforms = self.read_chunk(ChunkID::Transform)?;
        let instances = self.read_chunk(ChunkID::Instance)?;
        let meta = if let Some(meta) = meta {
            Chunk::encode_fixed(from_ref(meta), meta_to_bytes)
        } else {
            self.read_chunk(ChunkID::Meta)?
        };
        let cameras = if let Some(cameras) = cameras {
            Chunk::encode_fixed(cameras, camera_to_bytes)
        } else {
            self.read_chunk(ChunkID::Camera)?
        };
        let materials = if let Some(materials) = materials {
            Chunk::encode_dynamic(materials, material_to_bytes)
        } else {
            self.read_chunk(ChunkID::Material)?
        };
        let lights = if let Some(lights) = lights {
            Chunk::encode_dynamic(lights, light_to_bytes)
        } else {
            self.read_chunk(ChunkID::Light)?
        };
        {
            // Reopens the file in write mode (actually creates a new file, as most content will be
            // shifted).
            let mut writer = BufWriter::new(File::create(&self.filepath)?);
            write_header(&mut writer)?;
            let chunks = [
                (ChunkID::Vertex, vertices),
                (ChunkID::Mesh, meshes),
                (ChunkID::Camera, cameras),
                (ChunkID::Texture, textures),
                (ChunkID::Material, materials),
                (ChunkID::Transform, transforms),
                (ChunkID::Instance, instances),
                (ChunkID::Light, lights),
                (ChunkID::Meta, meta),
            ];
            ContentV1::write_chunks(&mut writer, &chunks)?;
            self.reader = BufReader::new(File::open(&self.filepath)?);
        }
        self.offsets = OffsetsTable::seek_and_parse(&mut self.reader)?;
        Ok(())
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
    /// Converts the given elements into a Chunk. Used for elements with fixed size when serialized.
    /// func is the function to convert a single element into an array of bytes.
    fn encode_fixed<T, const SIZE: usize>(items: &[T], func: fn(&T) -> [u8; SIZE]) -> Self {
        if !items.is_empty() {
            let uncompressed = items.iter().flat_map(func).collect::<Vec<u8>>();
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

    /// Converts back the Chunk into a slice of elements.
    /// Used for elements with fixed size when serialized.
    /// func is the function to convert an array of bytes into a vector of elements.
    fn decode_fixed<T, const SIZE: usize>(
        self,
        func: fn([u8; SIZE]) -> T,
        name: &'static str,
    ) -> Result<Vec<T>, Error> {
        if !self.data.is_empty() {
            if let Some(verified_data) = verify_hash(self.data) {
                let decompressed = decompress(&verified_data[..]);
                Ok(decompressed
                    .chunks_exact(SIZE)
                    .map(|s| {
                        let mut a = [0; SIZE];
                        a.copy_from_slice(s);
                        a
                    })
                    .map(func)
                    .collect())
            } else {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Corrupted {}", name),
                ))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Converts the givens element into a Chunk.
    /// Used for elements with dynamic size when serialized.
    /// func is the function to convert a single element into a slice of bytes.
    fn encode_dynamic<T>(items: &[T], func: fn(&T) -> Vec<u8>) -> Self {
        if !items.is_empty() {
            let len = items.len() as u16;
            let mut uncompressed = len.to_le_bytes().iter().copied().collect::<Vec<_>>();
            for item in items {
                let encoded = func(item);
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

    /// Converts back the Chunk into a vector of elements.
    /// Used for elements with dynamic size when serialized.
    /// func is the function to convert a slice of bytes into an vector of elements.
    fn decode_dynamic<T>(self, func: fn(&[u8]) -> T, name: &'static str) -> Result<Vec<T>, Error> {
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
                    let instance = func(&decompressed[index..index + encoded_len]);
                    index += encoded_len;
                    retval.push(instance);
                }
                Ok(retval)
            } else {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Corrupted chunk: {}", name),
                ))
            }
        } else {
            Ok(Vec::with_capacity(0))
        }
    }

    /// Creates a chunk from a slice of textures.
    /// Textures uses an ad-hoc compression so they cannot use [Chunk::encode_dynamic]
    fn encode_textures(items: &[Texture]) -> Self {
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

    /// Converts back the Chunk into a vector of textures.
    /// Textures uses an ad-hoc compression so they cannot use [Chunk::decode_dynamic]
    fn decode_textures(self) -> Result<Vec<Texture>, Error> {
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
}

/// Converts a Vertex to a vector of bytes.
fn vertex_to_bytes(vert: &Vertex) -> [u8; 32] {
    let vv: [f32; 3] = Point3::into(vert.vv);
    let vn: [f32; 3] = Vec3::into(vert.vn);
    let vt: [f32; 2] = Point2::into(vert.vt);
    let mut retval = [0; 32];
    let mut i = 0;
    for val in vv.iter().chain(vn.iter()).chain(vt.iter()) {
        let bytes = f32::to_le_bytes(*val);
        retval[i] = bytes[0];
        retval[i + 1] = bytes[1];
        retval[i + 2] = bytes[2];
        retval[i + 3] = bytes[3];
        i += 4;
    }
    retval
}

/// Converts a vector of bytes to a Vertex.
fn bytes_to_vertex(data: [u8; 32]) -> Vertex {
    let vv = Point3::new(
        f32::from_le_bytes(data[0..4].try_into().unwrap()),
        f32::from_le_bytes(data[4..8].try_into().unwrap()),
        f32::from_le_bytes(data[8..12].try_into().unwrap()),
    );
    let vn = Vec3::new(
        f32::from_le_bytes(data[12..16].try_into().unwrap()),
        f32::from_le_bytes(data[16..20].try_into().unwrap()),
        f32::from_le_bytes(data[20..24].try_into().unwrap()),
    );
    let vt = Point2::new(
        f32::from_le_bytes(data[24..28].try_into().unwrap()),
        f32::from_le_bytes(data[28..32].try_into().unwrap()),
    );
    Vertex { vv, vn, vt }
}

/// Converts a Mesh to a vector of bytes.
fn mesh_to_bytes(mesh: &Mesh) -> Vec<u8> {
    let id = u16::to_le_bytes(mesh.id);
    let faces_no = u32::to_le_bytes(mesh.indices.len() as u32);
    let material = u16::to_le_bytes(mesh.material);
    id.iter()
        .chain(faces_no.iter())
        .chain(material.iter())
        .copied()
        .chain(mesh.indices.iter().copied().flat_map(u32::to_le_bytes))
        .collect::<Vec<_>>()
}

/// Converts a vector of bytes to a Mesh.
fn bytes_to_mesh(data: &[u8]) -> Mesh {
    let id = u16::from_le_bytes(data[0..2].try_into().unwrap());
    let faces_no = u32::from_le_bytes(data[2..6].try_into().unwrap());
    let material = u16::from_le_bytes(data[6..8].try_into().unwrap());
    let face_end = (8 + faces_no * 4) as usize;
    let indices = data[8..face_end]
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    Mesh {
        id,
        indices,
        material,
    }
}

/// Converts a Camera to a vector of bytes.
fn camera_to_bytes(camera: &Camera) -> [u8; 49] {
    let camera_type;
    let position;
    let target;
    let up;
    let near_plane;
    let far_plane;
    let other_arg;
    match camera {
        Camera::Perspective(cam) => {
            camera_type = 0;
            position = cam.position;
            target = cam.target;
            up = cam.up;
            other_arg = cam.fovx;
            near_plane = cam.near;
            far_plane = cam.far;
        }
        Camera::Orthographic(cam) => {
            camera_type = 1;
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
    let mut whole = [0; 49];
    whole[0] = camera_type;
    whole[1..5].copy_from_slice(&f32::to_le_bytes(pos[0]));
    whole[5..9].copy_from_slice(&f32::to_le_bytes(pos[1]));
    whole[9..13].copy_from_slice(&f32::to_le_bytes(pos[2]));
    whole[13..17].copy_from_slice(&f32::to_le_bytes(tgt[0]));
    whole[17..21].copy_from_slice(&f32::to_le_bytes(tgt[1]));
    whole[21..25].copy_from_slice(&f32::to_le_bytes(tgt[2]));
    whole[25..29].copy_from_slice(&f32::to_le_bytes(upp[0]));
    whole[29..33].copy_from_slice(&f32::to_le_bytes(upp[1]));
    whole[33..37].copy_from_slice(&f32::to_le_bytes(upp[2]));
    whole[37..41].copy_from_slice(&f32::to_le_bytes(other_arg));
    whole[41..45].copy_from_slice(&f32::to_le_bytes(near_plane));
    whole[45..49].copy_from_slice(&f32::to_le_bytes(far_plane));
    whole
}

/// Converts a vector of bytes to a Camera.
fn bytes_to_camera(data: [u8; 49]) -> Camera {
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
fn texture_to_bytes(texture: &Texture) -> Vec<u8> {
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
    let total_len = 3 * std::mem::size_of::<u8>() + str_len + tex_len;
    let mut retval = Vec::with_capacity(total_len);
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
        TextureFormat::RgbaSrgb => 2,
        TextureFormat::RgbaNorm => 3,
    }
}

fn u8_to_format(format: u8) -> Result<TextureFormat, Error> {
    match format {
        1 => Ok(TextureFormat::Gray),
        2 => Ok(TextureFormat::RgbaSrgb),
        3 => Ok(TextureFormat::RgbaNorm),
        _ => panic!("Texture format unexpected"),
    }
}

/// Converts a vector of bytes to a Texture.
fn bytes_to_texture(data: &[u8]) -> Result<Texture, Error> {
    let format = u8_to_format(data[0])?;
    let str_len = data[1] as usize;
    let mut index = 2;
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
        TextureFormat::RgbaSrgb | TextureFormat::RgbaNorm => {
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
    Ok(texture)
}

/// Converts a Material to a vector of bytes.
fn material_to_bytes(material: &Material) -> Vec<u8> {
    let str_len = material.name.bytes().len();
    let total_len = 6 * std::mem::size_of::<u8>() // shader id, metal id, diffuse mul
        + 4 * std::mem::size_of::<f32>() // ior, roughness mul, metal mul, anisotropy
        + 5 * std::mem::size_of::<u16>() // diff, rough, metal, normal, opacity textures
        + str_len; // material name
    let mut retval = Vec::with_capacity(total_len);
    retval.push(material.shader.into());
    retval.push(material.metal.into());
    retval.extend(material.diffuse_mul);
    retval.extend(f32::to_le_bytes(material.ior));
    retval.extend(f32::to_le_bytes(material.roughness_mul));
    retval.extend(f32::to_le_bytes(material.metalness_mul));
    retval.extend(f32::to_le_bytes(material.anisotropy));
    retval.extend(u16::to_le_bytes(material.diffuse));
    retval.extend(u16::to_le_bytes(material.roughness));
    retval.extend(u16::to_le_bytes(material.metalness));
    retval.extend(u16::to_le_bytes(material.normal));
    retval.extend(u16::to_le_bytes(material.opacity));
    retval.extend(material.name.bytes());
    retval
}

/// Converts a vector of bytes to a Material.
fn bytes_to_material(data: &[u8]) -> Material {
    let shader = ShaderMat::from(data[0]);
    let metal = Metal::from(data[1]);
    let diffuse_mul = data[2..6].try_into().unwrap();
    let ior = f32::from_le_bytes(data[6..10].try_into().unwrap());
    let roughness_mul = f32::from_le_bytes(data[10..14].try_into().unwrap());
    let metalness_mul = f32::from_le_bytes(data[14..18].try_into().unwrap());
    let anisotropy = f32::from_le_bytes(data[18..22].try_into().unwrap());
    let diffuse = u16::from_le_bytes(data[22..24].try_into().unwrap());
    let roughness = u16::from_le_bytes(data[24..26].try_into().unwrap());
    let metalness = u16::from_le_bytes(data[26..28].try_into().unwrap());
    let normal = u16::from_le_bytes(data[28..30].try_into().unwrap());
    let opacity = u16::from_le_bytes(data[30..32].try_into().unwrap());
    let name = String::from_utf8(data[32..].to_vec()).unwrap();
    Material {
        name,
        shader,
        metal,
        ior,
        diffuse,
        diffuse_mul,
        roughness,
        roughness_mul,
        metalness,
        metalness_mul,
        anisotropy,
        opacity,
        normal,
    }
}

/// Converts a Transform to a vector of bytes.
fn transform_to_bytes(transform: &Transform) -> [u8; 64] {
    transform.to_bytes()
}

/// Converts a vector of bytes to a Transform.
fn bytes_to_transform(data: [u8; 64]) -> Transform {
    Transform::from_bytes(data)
}

/// Converts a MeshInstance to a vector of bytes.
fn instance_to_bytes(instance: &MeshInstance) -> [u8; 4] {
    let i0 = u16::to_le_bytes(instance.mesh_id);
    let i1 = u16::to_le_bytes(instance.transform_id);
    [i0[0], i0[1], i1[0], i1[1]]
}

/// Converts a vector of bytes to a MeshInstance.
fn bytes_to_instance(data: [u8; 4]) -> MeshInstance {
    let mesh_id = u16::from_le_bytes(data[0..2].try_into().unwrap());
    let transform_id = u16::from_le_bytes(data[2..4].try_into().unwrap());
    MeshInstance {
        mesh_id,
        transform_id,
    }
}

/// Converts a Light to a vector of bytes.
fn light_to_bytes(light: &Light) -> Vec<u8> {
    let ltype = match light {
        Light::Omni(_) => 0,
        Light::Sun(_) => 1,
    };
    let color = light.emission().to_bytes();
    let posdir = match light {
        Light::Omni(l) => {
            let pos: [f32; 3] = l.position.into();
            pos.into_iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        }
        Light::Sun(l) => {
            let pos: [f32; 3] = l.position.into();
            let dir: [f32; 3] = l.direction.into();
            pos.into_iter()
                .chain(dir)
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        }
    };
    let name_len = light.name().bytes().len();
    let mut retval = Vec::with_capacity(1 + color.len() + posdir.len() + name_len);
    retval.push(ltype);
    retval.extend(color);
    retval.extend(posdir);
    retval.extend(light.name().bytes());
    retval
}

/// Converts a vector of bytes to a Light.
fn bytes_to_light(data: &[u8]) -> Light {
    let color = Spectrum::from_bytes(data[1..65].try_into().unwrap());
    let mut index = 65;
    match data[0] {
        0 => {
            let position = Point3::new(
                f32::from_le_bytes(data[index..index + 4].try_into().unwrap()),
                f32::from_le_bytes(data[index + 4..index + 8].try_into().unwrap()),
                f32::from_le_bytes(data[index + 8..index + 12].try_into().unwrap()),
            );
            index += 12;
            let name = String::from_utf8(data[index..].to_vec()).unwrap();
            Light::new_omni(name, color, position)
        }
        1 => {
            let position = Point3::new(
                f32::from_le_bytes(data[index..index + 4].try_into().unwrap()),
                f32::from_le_bytes(data[index + 4..index + 8].try_into().unwrap()),
                f32::from_le_bytes(data[index + 8..index + 12].try_into().unwrap()),
            );
            index += 12;
            let direction = Vec3::new(
                f32::from_le_bytes(data[index..index + 4].try_into().unwrap()),
                f32::from_le_bytes(data[index + 4..index + 8].try_into().unwrap()),
                f32::from_le_bytes(data[index + 8..index + 12].try_into().unwrap()),
            );
            index += 12;
            let name = String::from_utf8(data[index..].to_vec()).unwrap();
            Light::new_sun(name, color, position, direction)
        }
        _ => panic!(),
    }
}

/// Converts a Light to a vector of bytes.
fn meta_to_bytes(meta: &Meta) -> [u8; 8] {
    let mut retval = [0; 8];
    let mut index = 0;
    retval[index..index + 4].copy_from_slice(&f32::to_le_bytes(meta.scene_radius));
    index += 4;
    retval[index..index + 4].copy_from_slice(&f32::to_le_bytes(meta.exposure));
    retval
}

/// Converts a vector of bytes to the Meta struct.
fn bytes_to_meta(data: [u8; 8]) -> Meta {
    let mut index = 0;
    let scene_radius = f32::from_le_bytes(data[index..index + 4].try_into().unwrap());
    index += 4;
    let exposure = f32::from_le_bytes(data[index..index + 4].try_into().unwrap());
    Meta {
        scene_radius,
        exposure,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_to_camera, bytes_to_mesh, bytes_to_meta, bytes_to_texture, bytes_to_transform,
        bytes_to_vertex, camera_to_bytes, compress, decompress, mesh_to_bytes, meta_to_bytes,
        texture_to_bytes, transform_to_bytes, vertex_to_bytes,
    };
    use crate::geometry::{Camera, Mesh, OrthographicCam, PerspectiveCam, Vertex};
    use crate::materials::{TextureFormat, TextureInfo};
    use crate::parser::v1::{
        bytes_to_instance, bytes_to_light, bytes_to_material, instance_to_bytes, light_to_bytes,
        material_to_bytes, HASH_SIZE,
    };
    use crate::parser::{parse, Meta, ParserVersion, HEADER_LEN};
    use crate::{
        Light, Material, MeshInstance, Metal, Serializer, ShaderMat, Spectrum, Texture, Transform,
    };
    use cgmath::{Matrix4, Point2, Point3, Vector3 as Vec3};
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
            let vv = Point3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let vn = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
            let vt = Point2::<f32>::new(rng.gen(), rng.gen());
            let vertex = Vertex { vv, vn, vt };
            buffer.push(vertex);
        }
        buffer
    }

    fn gen_meshes(count: u16, seed: u64) -> Vec<Mesh> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut buffer = Vec::with_capacity(count as usize);
        for id in 0..count {
            let high_range = rng.gen_bool(0.1);
            let indices_no: u32 = if high_range {
                rng.gen_range(0..100000)
            } else {
                rng.gen_range(0..1000)
            };
            let material = rng.gen();
            let indices = (0..indices_no).map(|_| rng.gen()).collect::<Vec<_>>();
            let mesh = Mesh {
                id,
                indices,
                material,
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

    fn gen_textures(count: u16, seed: u64) -> Vec<Texture> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let data = include_bytes!("../../../resources/checker.jpg");
        let image = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg).unwrap();
        let mut data = Vec::with_capacity(count as usize);
        for _ in 0..count {
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
            let format = if rng.gen_bool(0.33) {
                TextureFormat::Gray
            } else if rng.gen_bool(0.5) {
                TextureFormat::RgbaSrgb
            } else {
                TextureFormat::RgbaNorm
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
                TextureFormat::RgbaSrgb => Texture::new_rgba(info, cur_img.into_rgba8()),
                TextureFormat::RgbaNorm => Texture::new_rgba(info, cur_img.into_rgba8()),
            };
            data.push(texture);
        }
        data.into_iter().collect()
    }

    fn gen_materials(count: u16, seed: u64) -> Vec<Material> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut data = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let shaders = ShaderMat::all_values();
            let shader = shaders[rng.gen_range(0..shaders.len())];
            let ior = rng.gen_range(1.0..3.0);
            let diffuse = rng.gen_range(0..u16::MAX - 1);
            let roughness = rng.gen_range(0..u16::MAX - 1);
            let roughness_mul = rng.gen_range(0.0..1.0);
            let metalness = rng.gen_range(0..u16::MAX - 1);
            let metalness_mul = rng.gen_range(0.0..1.0);
            let metal = Metal::from(rng.gen_range(0..Metal::all_types().len()) as u8);
            let anisotropy = rng.gen_range(-1.0..1.0);
            let name = Xoshiro128StarStar::seed_from_u64(rng.gen())
                .sample_iter(&Alphanumeric)
                .take(rng.gen_range(0..255))
                .map(char::from)
                .collect::<String>();
            let diffuse_mul = [
                rng.gen_range(0..255),
                rng.gen_range(0..255),
                rng.gen_range(0..255),
                rng.gen_range(0..255),
            ];
            let opacity = if rng.gen_bool(0.1) {
                rng.gen_range(0..u16::MAX - 1)
            } else {
                0
            };
            let normal = if rng.gen_bool(0.1) {
                rng.gen_range(0..u16::MAX - 1)
            } else {
                0
            };
            let material = Material {
                name,
                shader,
                metal,
                ior,
                diffuse,
                diffuse_mul,
                opacity,
                normal,
                roughness,
                roughness_mul,
                metalness,
                metalness_mul,
                anisotropy,
            };
            data.push(material);
        }
        data.into_iter().collect()
    }

    fn gen_transforms(count: u16, seed: u64) -> Vec<Transform> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut data = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let matrix = Matrix4::new(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            );
            data.push(Transform::from(matrix));
        }
        data
    }

    fn gen_instances(count: u16, seed: u64) -> Vec<MeshInstance> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut data = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mesh_id = rng.gen();
            let transform_id = rng.gen();
            let instance = MeshInstance {
                mesh_id,
                transform_id,
            };
            data.push(instance)
        }
        data
    }

    fn gen_lights(count: u16, seed: u64) -> Vec<Light> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let mut data = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let name = Xoshiro128StarStar::seed_from_u64(rng.gen())
                .sample_iter(&Alphanumeric)
                .take(rng.gen_range(0..255))
                .map(char::from)
                .collect::<String>();
            let color = Spectrum::from_blackbody(rng.gen_range(800.0..10000.0));
            let light = if rng.gen_bool(0.5) {
                let position = Point3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
                Light::new_omni(name, color, position)
            } else {
                let position = Point3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
                let direction = Vec3::<f32>::new(rng.gen(), rng.gen(), rng.gen());
                Light::new_sun(name, color, position, direction)
            };
            data.push(light);
        }
        data
    }

    fn gen_meta(seed: u64) -> Meta {
        let mut rng = Xoshiro128StarStar::seed_from_u64(seed);
        let scene_radius = rng.gen();
        let exposure = rng.gen_range(1E-3..1E3);
        Meta {
            scene_radius,
            exposure,
        }
    }

    #[test]
    fn encode_decode_vertex() {
        let vertices = gen_vertices(32, 0xC2B4D5A5A9E49945);
        for vertex in vertices {
            let data = vertex_to_bytes(&vertex);
            let decoded = bytes_to_vertex(data);
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
            let decoded = bytes_to_camera(data);
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
        let materials = gen_materials(485, 0xC7F8CE22512B15FB);
        for material in materials {
            let data = material_to_bytes(&material);
            let decoded = bytes_to_material(&data);
            assert_eq!(decoded, material);
        }
    }

    #[test]
    fn encode_decode_transforms() {
        let transforms = gen_transforms(654, 0xBD5EC77C790B70F4);
        for transform in transforms {
            let data = transform_to_bytes(&transform);
            let decoded = bytes_to_transform(data);
            assert_eq!(decoded, transform);
        }
    }

    #[test]
    fn encode_decode_instances() {
        let instances = gen_instances(1435, 0x9714321951EF2533);
        for instance in instances {
            let data = instance_to_bytes(&instance);
            let decoded = bytes_to_instance(data);
            assert_eq!(decoded, instance);
        }
    }

    #[test]
    fn encode_decode_lights() {
        let lights = gen_lights(118, 0xF7B3E064F943374E);
        for light in lights {
            let data = light_to_bytes(&light);
            let decoded = bytes_to_light(&data);
            assert_eq!(decoded, light);
        }
    }

    #[test]
    fn encode_decode_meta() {
        let meta = gen_meta(0x546DB57AB5589A5A);
        let data = meta_to_bytes(&meta);
        let decoded = bytes_to_meta(data);
        assert_eq!(decoded, meta);
    }

    #[test]
    fn write_and_read_only_vert() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000, 0xBE8AE7F7E3A5248E);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_vertices.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_meshes(&meshes)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_cameras(&cameras)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_textures(&textures)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_materials(&materials)
            .serialize()?;
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
    fn write_and_read_only_transforms() -> Result<(), std::io::Error> {
        let transforms = gen_transforms(512, 0x091722F61F6D3E1A);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_transforms.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_transforms(&transforms)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_transforms = read.transforms()?;
        assert_eq!(read_transforms.len(), transforms.len());
        for i in 0..read_transforms.len() {
            let val = &read_transforms.get(i).unwrap();
            let expected = &transforms.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_instances() -> Result<(), std::io::Error> {
        let instances = gen_instances(512, 0x3B88C0CCB06D05CB);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_instances.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_instances(&instances)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_instances = read.instances()?;
        assert_eq!(read_instances.len(), instances.len());
        for i in 0..read_instances.len() {
            let val = &read_instances.get(i).unwrap();
            let expected = &instances.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_lights() -> Result<(), std::io::Error> {
        let lights = gen_lights(512, 0x6C1A6FE161CFC7DE);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_lights.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_lights(&lights)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_lights = read.lights()?;
        assert_eq!(read_lights.len(), lights.len());
        for i in 0..read_lights.len() {
            let val = &read_lights.get(i).unwrap();
            let expected = &lights.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn write_and_read_only_meta() -> Result<(), std::io::Error> {
        let meta = gen_meta(0x0FC1E162A949E22A);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_meta.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_metadata(&meta)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_meta = read.meta()?;
        assert_eq!(read_meta, meta);
        Ok(())
    }

    #[test]
    fn write_and_read_everything() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0x98DA1392A52639C2);
        let meshes = gen_meshes(100, 0xEFB101FDF7F185FB);
        let cameras = gen_cameras(100, 0x9B20F550F7EDF740);
        let textures = gen_textures(4, 0x122C4C9E1A51AC4B);
        let materials = gen_materials(100, 0x76D7971188303D82);
        let instances = gen_instances(100, 0xCAB0F2794E10665C);
        let transforms = gen_transforms(100, 0x8AFE0C931FBD4D69);
        let lights = gen_lights(150, 0x10FD94C4A4B032C0);
        let meta = gen_meta(0xC6FB668642859F83);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read_everything.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .with_meshes(&meshes)
            .with_instances(&instances)
            .with_transforms(&transforms)
            .with_textures(&textures)
            .with_materials(&materials)
            .with_lights(&lights)
            .with_cameras(&cameras)
            .with_metadata(&meta)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        let read_vertices = read.vertices()?;
        let read_meshes = read.meshes()?;
        let read_cameras = read.cameras()?;
        let read_textures = read.textures()?;
        let read_materials = read.materials()?;
        let read_transforms = read.transforms()?;
        let read_instances = read.instances()?;
        let read_lights = read.lights()?;
        let read_meta = read.meta()?;
        assert_eq!(read_vertices.len(), vertices.len());
        assert_eq!(read_meshes.len(), meshes.len());
        assert_eq!(read_cameras.len(), cameras.len());
        assert_eq!(read_textures.len(), textures.len());
        assert_eq!(read_materials.len(), materials.len());
        assert_eq!(read_transforms.len(), transforms.len());
        assert_eq!(read_instances.len(), instances.len());
        assert_eq!(read_lights.len(), lights.len());
        assert_eq!(read_meta, meta);
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
        for i in 0..read_transforms.len() {
            let val = &read_transforms.get(i).unwrap();
            let expected = &transforms.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_instances.len() {
            let val = &read_instances.get(i).unwrap();
            let expected = &instances.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_lights.len() {
            let val = &read_lights.get(i).unwrap();
            let expected = &lights.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }

    #[test]
    fn corrupted_offset() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0x8794A1E593281F2F);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_off.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_meshes(&meshes)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_cameras(&cameras)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_textures(&textures)
            .serialize()?;
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
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_materials(&materials)
            .serialize()?;
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
    fn corrupted_transforms() -> Result<(), std::io::Error> {
        let transforms = gen_transforms(100, 0x52220531E08FB566);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_transforms.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_transforms(&transforms)
            .serialize()?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.transforms().is_ok());
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
        assert!(read_corrupted.transforms().is_err());
        Ok(())
    }

    #[test]
    fn corrupted_instances() -> Result<(), std::io::Error> {
        let instances = gen_instances(500, 0x52C7F0FE60D65D06);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_instances.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_instances(&instances)
            .serialize()?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.instances().is_ok());
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
        assert!(read_corrupted.instances().is_err());
        Ok(())
    }

    #[test]
    fn corrupted_lights() -> Result<(), std::io::Error> {
        let lights = gen_lights(250, 0xB9206AA3C2681F81);
        let dir = tempdir()?;
        let file = dir.path().join("corrupted_lights.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_lights(&lights)
            .serialize()?;
        let mut read_ok = parse(file.as_path()).unwrap();
        assert!(read_ok.lights().is_ok());
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
        assert!(read_corrupted.lights().is_err());
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

    #[test]
    fn update_reopen() -> Result<(), std::io::Error> {
        // regression test: I forgot to write the header on update
        let vertices = gen_vertices(100, 0xBD4D59BF04981A1A);
        let dir = tempdir()?;
        let file = dir.path().join("update_some.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        assert_eq!(read.vertices()?.len(), vertices.len());
        read.update(None, None, None, None)?;
        assert_eq!(read.vertices()?.len(), vertices.len());
        // close file and reopen it
        let mut read = parse(file.as_path())?;
        assert_eq!(read.vertices()?.len(), vertices.len());
        Ok(())
    }

    #[test]
    fn update_some() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0xBD4D59BF04981A1A);
        let dir = tempdir()?;
        let file = dir.path().join("update_some.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        assert_eq!(read.vertices()?.len(), vertices.len());
        let new_cameras = gen_cameras(100, 0xECD7D80A8A4C4C95);
        let new_materials = gen_materials(100, 0xAA9475DE05B6CE41);
        let new_lights = gen_lights(100, 0xEF2F6EF8FD11E92E);
        let new_meta = gen_meta(0x3A77182EE1A0747E);
        read.update(
            Some(&new_cameras),
            Some(&new_materials),
            Some(&new_lights),
            Some(&new_meta),
        )?;
        assert_eq!(read.vertices()?.len(), vertices.len());
        assert_eq!(read.cameras()?.len(), new_cameras.len());
        assert_eq!(read.materials()?.len(), new_materials.len());
        assert_eq!(read.lights()?.len(), new_lights.len());
        Ok(())
    }

    #[test]
    fn update_all() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(100, 0x7CE285088B15CD6C);
        let meshes = gen_meshes(100, 0x0360E2B31852DCDA);
        let cameras = gen_cameras(25, 0x91D237698C5717D3);
        let textures = gen_textures(4, 0x4AE5995B104BBAB1);
        let materials = gen_materials(25, 0x6FEC53A488FBDB4F);
        let transforms = gen_transforms(25, 0x1E5CBA94679D9D3B);
        let instances = gen_instances(100, 0xC79389E3BBC74BCF);
        let lights = gen_lights(50, 0xA39F34BA2C56A7DC);
        let meta = gen_meta(0x16B1FF1406A24CA6);
        let dir = tempdir()?;
        let file = dir.path().join("update_all.bin");
        Serializer::new(file.to_str().unwrap(), ParserVersion::V1)
            .with_vertices(&vertices)
            .with_meshes(&meshes)
            .with_instances(&instances)
            .with_transforms(&transforms)
            .with_textures(&textures)
            .with_materials(&materials)
            .with_lights(&lights)
            .with_cameras(&cameras)
            .with_metadata(&meta)
            .serialize()?;
        let mut read = parse(file.as_path())?;
        remove_file(file.as_path())?;
        assert_eq!(read.vertices()?.len(), vertices.len());
        assert_eq!(read.meshes()?.len(), meshes.len());
        assert_eq!(read.cameras()?.len(), cameras.len());
        assert_eq!(read.textures()?.len(), textures.len());
        assert_eq!(read.materials()?.len(), materials.len());
        assert_eq!(read.transforms()?.len(), transforms.len());
        assert_eq!(read.instances()?.len(), instances.len());
        assert_eq!(read.lights()?.len(), lights.len());
        assert_eq!(read.meta()?, meta);
        let new_cameras = gen_cameras(100, 0x056F0B996A248BC4);
        let new_materials = gen_materials(100, 0x3ABE1A9BEB00DA7B);
        let new_lights = gen_lights(100, 0x5871F342932A7B6A);
        let new_meta = gen_meta(0xF4AF4CA42889AAD0);
        read.update(
            Some(&new_cameras),
            Some(&new_materials),
            Some(&new_lights),
            Some(&new_meta),
        )?;
        let read_vertices = read.vertices()?;
        let read_meshes = read.meshes()?;
        let read_cameras = read.cameras()?;
        let read_textures = read.textures()?;
        let read_materials = read.materials()?;
        let read_transforms = read.transforms()?;
        let read_instances = read.instances()?;
        let read_lights = read.lights()?;
        let read_meta = read.meta()?;
        assert_eq!(read_vertices.len(), vertices.len());
        assert_eq!(read_meshes.len(), meshes.len());
        assert_eq!(read_cameras.len(), new_cameras.len());
        assert_eq!(read_textures.len(), textures.len());
        assert_eq!(read_materials.len(), new_materials.len());
        assert_eq!(read_transforms.len(), transforms.len());
        assert_eq!(read_instances.len(), instances.len());
        assert_eq!(read_lights.len(), new_lights.len());
        assert_eq!(read_meta, new_meta);
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
            let expected = &new_cameras.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_textures.len() {
            let val = &read_textures.get(i).unwrap();
            let expected = &textures.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_materials.len() {
            let val = &read_materials.get(i).unwrap();
            let expected = &new_materials.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_transforms.len() {
            let val = &read_transforms.get(i).unwrap();
            let expected = &transforms.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_instances.len() {
            let val = &read_instances.get(i).unwrap();
            let expected = &instances.get(i).unwrap();
            assert_eq!(val, expected);
        }
        for i in 0..read_lights.len() {
            let val = &read_lights.get(i).unwrap();
            let expected = &new_lights.get(i).unwrap();
            assert_eq!(val, expected);
        }
        Ok(())
    }
}
