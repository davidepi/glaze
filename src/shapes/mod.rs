mod aabb;
pub use crate::shapes::aabb::AABB;
mod shape;
pub use crate::shapes::shape::HitPoint;
pub use crate::shapes::shape::Intersection;
pub use crate::shapes::shape::Shape;
mod sphere;
pub use crate::shapes::sphere::Sphere;

/// Unique ID for the primitive AABB
const ID_AABB: usize = 0;
/// Unique ID for the primitive Sphere (Normal pointing outside)
const ID_SPHERE: usize = 1;
/// Unique ID for the primitive AABB (Normal pointing inside)
const ID_SPHERE_INVERTED: usize = 2;

#[cfg(test)]
mod tests;
