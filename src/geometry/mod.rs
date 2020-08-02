//the 0-prefix is extensively used for aligning rows in the matrix
#![allow(clippy::zero_prefixed_literal)]
mod vec;
pub use self::vec::Normal;
pub use self::vec::Vec2;
pub use self::vec::Vec3;
mod point;
pub use self::point::Point2;
pub use self::point::Point3;
mod ray;
pub use self::ray::Ray;
mod matrix;
pub use self::matrix::Matrix4;

#[cfg(test)]
mod tests;
