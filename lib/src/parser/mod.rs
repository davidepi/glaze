use self::v1::ContentV1;
use crate::geometry::{Camera, Mesh, Vertex};
use crate::{Material, Texture};
use std::fmt::Display;
use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind, Read, Seek, Write};
use std::path::Path;
use std::str::FromStr;

// DO NOT CHANGE THESE TWO! Any changes should be made with a new ParserVersion, changing the inner
// content of the file, but the header must not change to retain backward compatibility.
pub const MAGIC_NUMBER: [u8; 5] = [0x67, 0x6C, 0x61, 0x7A, 0x65];
pub const HEADER_LEN: usize = 16;

/// The version of the parser.
///
/// This is used to determine the file format when writing to a file.
pub enum ParserVersion {
    /// Version 1.
    ///
    /// Features:
    /// - Vertices/meshes/materials/cameras compressions with `xz` preset 9.
    /// - Textures compression with `libpng` preset `FAST` and filter `UP`.
    /// - Each chunk of vertices/meshes/materials/camera/textures can be accessed independently.
    /// - The scene is not stored in memory but read at runtime.
    V1,
}

impl ParserVersion {
    /// Converts the ParserVersion to a number.
    fn from_byte(byte: u8) -> Result<Self, Error> {
        match byte {
            1 => Ok(ParserVersion::V1),
            _ => Err(Error::new(
                ErrorKind::InvalidInput,
                "Unsupported file version",
            )),
        }
    }

    /// Converts the ParserVersion to a string.
    pub fn to_str(&self) -> &'static str {
        match self {
            ParserVersion::V1 => "V1",
        }
    }
}

impl FromStr for ParserVersion {
    type Err = ParseVersionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "V1" => Ok(ParserVersion::V1),
            _ => Err(ParseVersionError {}),
        }
    }
}

impl Display for ParserVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_str())
    }
}

#[derive(Debug)]
/// Error struct used to report versioning error.
///
/// This is used in [ParserVersion::from_str] to report a wrong or unsupported version.
pub struct ParseVersionError;

impl Display for ParseVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unrecognized parser version")
    }
}

/// Parses a file using the ad-hoc format expected by this crate.
///
/// Given a path to a file written by this crate, this function parses it and return its content.
/// The content is returned in different formats, depending on the [ParserVersion] of the written
/// file. One should use the methods provided by the trait [ParsedScene] to access the actual
/// content of the file.
///
/// # Errors
/// This function may return an error if the file is corrupted or impossible to parse.
///
/// # Examples
/// ```no_run
/// let mut parsed = glaze::parse("test.bin").expect("Failed to parse file");
/// let vertices = parsed.vertices().unwrap();
/// ```
pub fn parse<P: AsRef<Path>>(file: P) -> Result<Box<dyn ParsedScene + Send>, Error> {
    let mut fin = File::open(file.as_ref())?;
    let mut header = [0; HEADER_LEN];
    if fin.read_exact(&mut header).is_ok() {
        let magic = &header[0..5];
        if magic != MAGIC_NUMBER {
            Err(Error::new(
                ErrorKind::InvalidInput,
                "Wrong or empty input file",
            ))
        } else {
            let version = ParserVersion::from_byte(header[5])?;
            let parsed = match version {
                ParserVersion::V1 => Box::new(ContentV1::parse(file, fin)?),
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

/// Saves a file using the format expected by this crate.
///
/// This function saves a list of vertices/meshes/camera/textures/materials to the given path
/// using the provided [ParserVersion].
///
/// # Errors
/// This function may return an error if the file is impossible to write, either for permissions
/// reasons or because theres no space left on the disk.
///
/// # Examples
/// ```no_run
/// let vertices = Vec::new();
/// glaze::serialize(
///     "test.bin",
///     glaze::ParserVersion::V1,
///     &vertices,
///     &[],
///     &[],
///     &[],
///     &[],
/// )
/// .expect("Failed to save file");
/// ```
pub fn serialize<P: AsRef<Path>>(
    file: P,
    version: ParserVersion,
    vertices: &[Vertex],
    meshes: &[Mesh],
    cameras: &[Camera],
    textures: &[(u16, Texture)],
    materials: &[(u16, Material)],
) -> Result<(), Error> {
    let mut fout = BufWriter::new(File::create(file)?);
    write_header(&mut fout)?;
    match version {
        ParserVersion::V1 => {
            ContentV1::serialize(fout, vertices, meshes, cameras, textures, materials)?
        }
    };
    Ok(())
}

/// Writes the header of the file.
pub(super) fn write_header<W: Write + Seek>(file: &mut W) -> Result<(), Error> {
    let magic = MAGIC_NUMBER;
    file.write_all(&magic)?;
    file.write_all(&[1_u8])?;
    Ok(())
}

/// Returns true if the file has already been converted to a format supported by this crate.
///
/// This crate requires file to be in a specific version. This function checks whether the file
/// is supported or not.
///
/// In case the file does not exist or can not be read, false is returned.
pub fn converted_file<P: AsRef<Path>>(file: P) -> bool {
    let mut header = [0; HEADER_LEN];
    if let Ok(mut file) = File::open(file) {
        if let Ok(()) = file.read_exact(&mut header) {
            header[0..5] == MAGIC_NUMBER
        } else {
            false
        }
    } else {
        false
    }
}

/// Trait used for accessing the content of the parsed file and updating them.
///
/// This trait is used to access the content of the parsed file. Various parser versions may
/// implement this trait and return a `Box<dyn ParsedContent>`.
pub trait ParsedScene {
    /// Retrieve only the [Vertex]s contained in the file.
    fn vertices(&mut self) -> Result<Vec<Vertex>, Error>;
    /// Retrieve only the [Mesh]es contained in the file.
    fn meshes(&mut self) -> Result<Vec<Mesh>, Error>;
    /// Retrieve only the [Camera]s contained in the file.
    fn cameras(&mut self) -> Result<Vec<Camera>, Error>;
    /// Retrieve only the [Texture]s contained in the file.
    fn textures(&mut self) -> Result<Vec<(u16, Texture)>, Error>;
    /// Retrieve only the [Material]s contained in the file.
    fn materials(&mut self) -> Result<Vec<(u16, Material)>, Error>;
    /// Updates an existing file.
    fn update(
        &mut self,
        cameras: Option<&[Camera]>,
        materials: Option<&[(u16, Material)]>,
    ) -> Result<(), Error>;
}

mod v1;
