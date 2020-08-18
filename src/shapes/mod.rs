mod aabb;
pub use crate::shapes::aabb::AABB;
mod shape;
pub use crate::shapes::shape::HitPoint;
pub use crate::shapes::shape::Intersection;
pub use crate::shapes::shape::Shape;
mod sphere;
pub use crate::shapes::sphere::Sphere;

#[cfg(test)]
mod tests;
