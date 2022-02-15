mod vertex;
pub use self::vertex::Vertex;
mod mesh;
pub use self::mesh::{Mesh, MeshInstance, Transform};
mod camera;
pub use self::camera::{Camera, OrthographicCam, PerspectiveCam};
mod color;
pub use self::color::{ColorRGB, ColorXYZ};
mod spectrum;
pub use self::spectrum::Spectrum;
mod light;
pub use self::light::{Light, LightType};

pub(crate) const SBT_LIGHT_STRIDE: usize = 1;
pub(crate) const SBT_LIGHT_TYPES: usize = 2;
