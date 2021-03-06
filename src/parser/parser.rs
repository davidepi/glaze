use std::error::Error;
use std::fmt::Display;

/// Error related to bad parsing of a 3D geometry file.
///
/// This error may be due to:
/// - wrong I/O operations in case of [GeometryError::IoError]
/// - bad file content in case of [GeometryError::ParseError]
#[derive(Debug)]
pub enum GeometryError {
    IoError(std::io::Error),
    ParseError(ParseGeometryError),
}

impl Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::IoError(err) => write!(f, "{}", err),
            GeometryError::ParseError(err) => write!(f, "{}", err),
        }
    }
}

impl Error for GeometryError {}

impl From<std::io::Error> for GeometryError {
    fn from(err: std::io::Error) -> Self {
        GeometryError::IoError(err)
    }
}

impl From<ParseGeometryError> for GeometryError {
    fn from(err: ParseGeometryError) -> Self {
        GeometryError::ParseError(err)
    }
}

/// Error originating from content inside a 3D geometry file.
#[derive(Debug)]
pub struct ParseGeometryError {
    /// File where the error originated.
    pub file: String,
    /// Line where the error originated.
    pub line: usize,
    /// Cause of error.
    pub cause: String,
}

impl Error for ParseGeometryError {}

impl std::fmt::Display for ParseGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Error while parsing obj file {}: at line {}, {}",
            self.file, self.line, self.cause
        )
    }
}

/// Trait representing the ability to parse a file containing a 3D geometry description.
///
/// This file should contain all the vertices, normals and faces to be imported.
pub trait GeometryParser {
    /// Parse the file and return the collected data.
    ///
    /// The file name should be stored in a field inside the containing structure and not passed
    /// as parameter.
    ///
    /// This function returns a vector of parsed geometries found inside the file, or an error
    /// specifying the file and line where it originated along with the cause.
    /// # Examples
    /// Basic usage:
    /// ```no_run
    /// use glaze::parser::{GeometryParser, Obj};
    ///
    /// let obj = Obj::new("suzanne.obj");
    /// let res = obj.parse();
    /// ```
    fn parse(&self) -> Result<Vec<ParsedGeometry>, GeometryError>;
}

/// Struct containing parsed geometric data
///
/// This struct is mostly file and library agnostic so just a big vector of vertices, normals and
/// faces.
#[derive(Clone)]
pub struct ParsedGeometry {
    /// Name of the geometry
    pub name: String,
    /// Vector of vertices. Each vertex is an array of three floats representing `x`, `y` and `z`.
    pub vv: Vec<[f32; 3]>,
    /// Vector of texture coordinates.
    /// Each coordinate is an array of two floats representing `x` and `y`.
    pub vt: Vec<[f32; 2]>,
    /// Vector of normals. Each normal is an array of three floats representing `x`, `y` and `z`.
    pub vn: Vec<[f32; 3]>,
    /// Vector of faces.
    ///
    /// Each face is a triangle composed by a sequence of 9 indices 0-based.
    /// Each index refers to a particular structure contained in the instance of this struct.
    /// Each triplet of 3 indices correspond to vertex, vertex texture and normal of the triangle
    /// vertex respectively.
    ///
    /// Negative numbers allows indexing from the back, i.e -1 refers to the last position in `vv`
    /// or `vt` or `vn`.
    /// If a vertex does not have `vt` or `vn`, the maximum value for `i32` will be used.
    pub ff: Vec<[i32; 9]>,
}
