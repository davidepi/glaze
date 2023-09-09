/// A vector made of two components `x` and `y`.
///
/// Used to represent a physical, two-dimensional length.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Extent2D<T> {
    pub x: T,
    pub y: T,
}

/// A vector made of three components `x` and `y`.
///
/// Used to represent a physical, three-dimensional length.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Extent3D<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A rectangle made of top-left corner coordinates and axis width/height.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}
