use self::v1::ContentV1;
use crate::geometry::Camera;
use crate::geometry::Mesh;
use crate::geometry::Scene;
use crate::geometry::Vertex;
use std::convert::TryInto;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Read;
use std::io::Write;
use std::path::Path;

// DO NOT CHANGE THESE TWO! Any changes should be made with a new ParserVersion, changing the inner
// content of the file, but the header must not change to retain backward compatibility.
const MAGIC_NUMBER: u16 = 0x2F64;
const HEADER_LEN: usize = 16;

pub enum ParserVersion {
    V1,
}

impl ParserVersion {
    fn to_byte(&self) -> u8 {
        match self {
            ParserVersion::V1 => 1,
        }
    }

    fn from_byte(byte: u8) -> Result<Self, Error> {
        match byte {
            1 => Ok(ParserVersion::V1),
            _ => Err(Error::new(
                ErrorKind::InvalidInput,
                "Unsupported file version",
            )),
        }
    }
}

pub fn parse(file: &Path) -> Result<Box<dyn ParsedContent>, Error> {
    let mut fin = File::open(file)?;
    let mut header = [0; HEADER_LEN];
    if fin.read_exact(&mut header).is_ok() {
        let magic = u16::from_be_bytes(header[0..2].try_into().unwrap());
        if magic != MAGIC_NUMBER {
            Err(Error::new(
                ErrorKind::InvalidInput,
                "Wrong or empty input file",
            ))
        } else {
            let version = ParserVersion::from_byte(header[2])?;
            let parsed = match version {
                ParserVersion::V1 => Box::new(ContentV1::parse(&mut fin)?),
            };
            Ok(parsed)
        }
    } else {
        Err(Error::new(
            ErrorKind::InvalidInput,
            "Wrong or empty input file",
        ))
    }
}

pub fn serialize(file: &Path, version: ParserVersion, scene: &Scene) -> Result<(), Error> {
    let mut fout = File::create(file)?;
    let magic = u16::to_be_bytes(MAGIC_NUMBER);
    fout.write_all(&magic)?;
    fout.write_all(&[1_u8])?;
    fout.write_all(&[0; 13])?;
    let content = match version {
        ParserVersion::V1 => ContentV1::serialize(scene),
    };
    fout.write_all(&content)?;
    Ok(())
}

pub trait ParsedContent {
    fn scene(self) -> Scene;
    fn vertices(&self) -> &Vec<Vertex>;
    fn meshes(&self) -> &Vec<Mesh>;
    fn cameras(&self) -> &Vec<Camera>;
}

mod v1;
