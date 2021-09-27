use self::v1::ContentV1;
use crate::geometry::{Camera, Mesh, Scene, Vertex};
use crate::materials::{Library, Texture};
use std::convert::TryInto;
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Write};
use std::path::Path;

// DO NOT CHANGE THESE TWO! Any changes should be made with a new ParserVersion, changing the inner
// content of the file, but the header must not change to retain backward compatibility.
const MAGIC_NUMBER: u16 = 0x2F64;
const HEADER_LEN: usize = 16;

/// The version of the parser.
///
/// This is used to determine the file format when writing to a file.
pub enum ParserVersion {
    /// Version 1.
    ///
    /// Very basic version. Loads everything in memory and may be tremendously inefficient in terms
    /// of memory usage, up to 2X the size of the file.
    ///
    /// Features:
    /// - None LOL
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

/// Parses a file using the ad-hoc format expected by this crate.
///
/// Given a path to a file written by this crate, this function parses it and return its content.
/// The content is returned in different formats, depending on the [ParserVersion] of the written
/// file. One should use the methods provided by the trait [ParsedContent] to access the actual
/// content of the file.
///
/// # Errors
/// This function may return an error if the file is corrupted or impossible to parse.
///
/// # Examples
/// ```no_run
/// let parsed = glaze::parse("test.bin").expect("Failed to parse file");
/// let scene = parsed.scene();
/// ```
pub fn parse<P: AsRef<Path>>(file: P) -> Result<Box<dyn ParsedContent>, Error> {
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

/// Save a file using the format expected by this crate.
///
/// This function saves a [Scene] to the given path using the provided [ParserVersion].
///
/// # Errors
/// This function may return an error if the file is impossible to write, either for permissions
/// reasons or because theres no space left on the disk.
///
/// # Examples
/// ```no_run
/// let scene = glaze::Scene::default();
/// glaze::serialize("test.bin", glaze::ParserVersion::V1, &scene).expect("Failed to save file");
/// ```
pub fn serialize<P: AsRef<Path>>(
    file: P,
    version: ParserVersion,
    scene: &Scene,
) -> Result<(), Error> {
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

/// Trait used for accessing the content of the parsed file.
///
/// This trait is used to access the content of the parsed file. Various parser versions may
/// implement this trait and return a `Box<dyn ParsedContent>`.
pub trait ParsedContent {
    /// Retrieve the entire [Scene] contained in the file.
    fn scene(&self) -> Scene;
    /// Retrieve only the [Vertex]s contained in the file.
    fn vertices(&self) -> Vec<Vertex>;
    /// Retrieve only the [Mesh]es contained in the file.
    fn meshes(&self) -> Vec<Mesh>;
    /// Retrieve only the [Camera]s contained in the file.
    fn cameras(&self) -> Vec<Camera>;
    /// Retrieve only the [Texture]s contained in the file.
    fn textures(&self) -> Library<Texture>;
}

mod filehasher;
mod v1;
