// this lint is a nightmare when working with linear algebra
#![allow(clippy::suspicious_operation_groupings)]
/// Module containing basic linear algebra like Vectors, Points and Matrices.
pub mod linear;
/// Module containing geometric shapes like AABBs, Spheres and Triangles and intersection helpers.
pub mod shapes;
/// Module containing utility functions that does not properly fit anywhere else.
pub mod utility;
