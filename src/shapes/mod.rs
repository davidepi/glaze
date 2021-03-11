mod aabb;
pub use crate::shapes::aabb::AABB;
mod shape;
pub use crate::shapes::shape::HitPoint;
pub use crate::shapes::shape::Intersection;
pub use crate::shapes::shape::Shape;
pub use crate::shapes::shape::VertexBuffer;
mod sphere;
pub use crate::shapes::sphere::Sphere;
mod accelerator;
pub use crate::shapes::accelerator::Accelerator;
// This mod contains some corner case that REQUIRES strict comparisons to avoid NaN values
#[allow(clippy::float_cmp)]
mod kdtree;
pub use crate::shapes::kdtree::KdTree;
mod mesh;
pub use crate::shapes::mesh::Mesh;
