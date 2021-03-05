use std::error::Error;
use std::fmt::Display;

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

#[derive(Debug)]
pub struct ParseGeometryError {
    pub file: String,
    pub line: usize,
    pub cause: String,
}

impl std::fmt::Display for ParseGeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Error while parsing obj file {}: at line {}, {}",
            self.file, self.line, self.cause
        )
    }
}

pub trait GeometryParser {
    fn parse(&self) -> Result<Vec<ParsedGeometry>, GeometryError>;
}

#[derive(Clone)]
pub struct ParsedGeometry {
    pub name: String,
    pub vv: Vec<[f32; 3]>,
    pub vn: Vec<[f32; 3]>,
    pub vt: Vec<[f32; 2]>,
    pub ff: Vec<[i32; 9]>,
}
