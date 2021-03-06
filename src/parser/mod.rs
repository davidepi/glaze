// all the modules are re-exported!
#[allow(clippy::module_inception)]
mod parser;
pub use self::parser::GeometryError;
pub use self::parser::GeometryParser;
pub use self::parser::ParseGeometryError;
pub use self::parser::ParsedGeometry;
mod obj;
pub use self::obj::Obj;
