use super::ParsedContent;
use super::HEADER_LEN;
use crate::geometry::Vertex;
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
    vert_offset: u64, // offset in bytes
    vert_len: u64,    // amount of data in bytes
}

impl Offsets {
    fn parse(file: &mut File) -> Result<Offsets, Error> {
        file.seek(SeekFrom::Start(HEADER_LEN as u64))?;
        let mut cl_data = [0; CONTENT_LIST_SIZE];
        file.read_exact(&mut cl_data)?;
        let vert_offset = u64::from_le_bytes(cl_data[0..8].try_into().unwrap());
        let vert_len = u64::from_le_bytes(cl_data[8..16].try_into().unwrap());
        Ok(Offsets {
            vert_offset,
            vert_len,
        })
    }
}

pub(super) struct ContentV1 {
    vertices: Vec<Vertex>,
}

impl ContentV1 {
    pub(super) fn parse(file: &mut File) -> Result<Self, Error> {
        let offsets = Offsets::parse(file)?;
        let vert_seek = offsets.vert_offset;
        file.seek(SeekFrom::Start(vert_seek))?;
        let mut vert_data = Vec::with_capacity(offsets.vert_len as usize);
        file.take(offsets.vert_len).read_to_end(&mut vert_data)?; // avoid zeroing vec
        let vertices = vert_data
            .chunks_exact(32)
            .map(bytes_to_vertex)
            .collect::<Vec<_>>();
        Ok(ContentV1 { vertices })
    }

    pub(super) fn serialize(vert: &[Vertex]) -> Vec<u8> {
        let vertices = vert
            .into_iter()
            .flat_map(vertex_to_bytes)
            .collect::<Vec<_>>();
        let vert_offset = (HEADER_LEN + CONTENT_LIST_SIZE) as u64;
        let vert_size = vertices.len();
        let mut retval = Vec::with_capacity(CONTENT_LIST_SIZE + vert_size);
        retval.extend_from_slice(&u64::to_le_bytes(vert_offset));
        retval.extend_from_slice(&u64::to_le_bytes(vert_size as u64));
        retval.extend(vertices.into_iter());
        retval
    }
}

impl ParsedContent for ContentV1 {
    fn vertices(&self) -> &Vec<Vertex> {
        &self.vertices
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

#[cfg(test)]
mod tests {
    use crate::geometry::Vertex;
    use crate::parser::parse;
    use crate::parser::serialize_v1;
    use cgmath::Vector2 as Vec2;
    use cgmath::Vector3 as Vec3;
    use rand::prelude::*;
    use rand::Rng;
    use rand_xoshiro::Xoshiro128StarStar;
    use std::fs::remove_file;
    use tempfile::tempdir;

    fn gen_vertices(len: usize) -> Vec<Vertex> {
        let mut rng = Xoshiro128StarStar::seed_from_u64(0x1234567890abcdef);
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

    #[test]
    fn write_and_read() -> Result<(), std::io::Error> {
        let vertices = gen_vertices(1000);
        let dir = tempdir()?;
        let file = dir.path().join("write_and_read.bin");
        serialize_v1(file.as_path(), &vertices)?;
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
}
