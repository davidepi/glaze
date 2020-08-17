use crate::geometry::matrix::Matrix4;
use crate::geometry::vec::Vec2;
use crate::geometry::vec::Vec3;
use crate::utility::{float_eq, gamma};
use overload::overload;
use std::fmt::Formatter;
use std::ops;

///  Two points representing a location in space
///
///  Point2 class represents a zero dimensional location in a two dimensional
///  cartesian space. It is designed as a separate class because by representing
///  a location, and not a direction, it shows a different behaviour in some
///  situations.
///
/// A Point2 consists of two coordinates, usually called `x` and `y`
#[derive(Copy, Clone)]
pub struct Point2 {
    /// A single precision floating point representing the `x` coordinate of the point
    pub x: f32,
    /// A single precision floating point representing the `y` coordinate of the point
    pub y: f32,
}

impl Point2 {
    /// Construct a point in the origin of the cartesian space, with coordinates `(0.0, 0.0)`
    /// # Examples
    /// ```
    /// use glaze::geometry::Point2;
    /// let p = Point2::zero();
    /// assert_eq!(p.x, 0.0);
    /// assert_eq!(p.y, 0.0);
    /// ```
    pub fn zero() -> Point2 {
        Point2 { x: 0., y: 0. }
    }

    /// Constructs a point in the space with the given `(x, y)` coordinates
    /// # Examples
    /// ```
    /// use glaze::geometry::Point2;
    /// let p = Point2::new(3.5, -2.2);
    /// assert_eq!(p.x, 3.5);
    /// assert_eq!(p.y, -2.2);
    /// ```
    pub fn new(x: f32, y: f32) -> Point2 {
        Point2 { x, y }
    }

    /// Computes the euclidean distance between two points, by computing the length of the segment
    /// between them
    /// # Examples
    /// ```
    /// use glaze::geometry::Point2;
    /// let p1 = Point2::new(-1., -1.);
    /// let p2 = Point2::new(2., 3.);
    /// assert_eq!(Point2::distance(&p1, &p2), 5.);
    /// ```
    pub fn distance(point_a: &Point2, point_b: &Point2) -> f32 {
        let x = point_b.x - point_a.x;
        let y = point_b.y - point_a.y;
        (x * x + y * y).sqrt()
    }

    /// Calculates the minimum between two points.
    /// The returned point will have the minimum value for each component.
    ///
    /// This function is implemented because the canonical std::cmp::min requires the trait Ord
    /// which in turn requires Eq. The trait Eq, however, has subtle implication for floating point
    /// values.
    /// # Examples
    /// ```
    /// use glaze::geometry::Point2;
    ///
    /// let a = Point2::new(-1.0, 1.0);
    /// let b = Point2::new(-2.0, 2.0);
    /// let min = Point2::min(&a, &b);
    ///
    /// assert_eq!(min.x, b.x);
    /// assert_eq!(min.y, a.y);
    /// ```
    pub fn min(a: &Point2, b: &Point2) -> Point2 {
        Point2 {
            x: a.x.min(b.x),
            y: a.y.min(b.y),
        }
    }

    /// Calculates the maximum between two points.
    /// The returned point will have the maximum value for each component.
    ///
    /// This function is implemented because the canonical std::cmp::min requires the trait Ord
    /// which in turn requires Eq. The trait Eq, however, has subtle implication for floating point
    /// values.
    /// # Examples
    /// ```
    /// use glaze::geometry::Point2;
    ///
    /// let a = Point2::new(-1.0, 1.0);
    /// let b = Point2::new(-2.0, 2.0);
    /// let max = Point2::max(&a, &b);
    ///
    /// assert_eq!(max.x, a.x);
    /// assert_eq!(max.y, b.y);
    /// ```
    pub fn max(a: &Point2, b: &Point2) -> Point2 {
        Point2 {
            x: a.x.max(b.x),
            y: a.y.max(b.y),
        }
    }
}

impl std::fmt::Display for Point2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point2[{}, {}]", self.x, self.y)
    }
}

overload!((a: ?Point2) + (b: ?Vec2) -> Point2 {Point2{x:a.x+b.x,y:a.y+b.y}});
overload!((a: ?Point2) + (b: ?Point2) -> Vec2 {Vec2{x:a.x+b.x,y:a.y+b.y}});
overload!((a: ?Point2) - (b: ?Vec2) -> Point2 {Point2{x:a.x-b.x,y:a.y-b.y}});
overload!((a: ?Point2) - (b: ?Point2) -> Vec2 {Vec2{x:a.x-b.x,y:a.y-b.y}});
overload!(- (a: ?Point2) -> Point2 {Point2{x:-a.x,y:-a.y}});
overload!((a: &mut Point2) += (b: ?Vec2) {a.x+=b.x;a.y+=b.y;});
overload!((a: &mut Point2) += (b: ?Point2) {a.x+=b.x;a.y+=b.y;});
overload!((a: &mut Point2) -= (b: ?Vec2) {a.x-=b.x;a.y-=b.y;});
overload!((a: &mut Point2) -= (b: ?Point2) {a.x-=b.x;a.y-=b.y;});

/// Three points representing a location in space
///
/// Point3 class represents a zero dimensional location in a three dimensional
/// cartesian space. It is designed as a separate class because by representing
/// a location, and not a direction, it shows a different behaviour in some
/// situations.
///
/// A Point3 consists of three coordinates, usually called `x`, `y` and `z`.
#[derive(Copy, Clone)]
pub struct Point3 {
    /// A single precision floating point representing the `x` coordinate of the point
    pub x: f32,
    /// A single precision floating point representing the `y` coordinate of the point
    pub y: f32,
    /// A single precision floating point representing the `z` coordinate of the point
    pub z: f32,
}

// group together a floating point and the possible rounding error for each component
pub(super) struct ErrorTrackedPoint3 {
    pub(super) value: Point3,
    pub(super) error: Vec3,
}

impl Point3 {
    /// Constructs a point in the Origin of the cartesian space, with coordinates (0.0, 0.0, 0.0)
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
    /// let p = Point3::zero();
    /// assert_eq!(p.x, 0.0);
    /// assert_eq!(p.y, 0.0);
    /// assert_eq!(p.z, 0.0);
    /// ```
    pub fn zero() -> Point3 {
        Point3 {
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    ///Constructs a point in the space with the given `(x, y, z)` coordinates
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
    /// let p = Point3::new(3.5, -2.2, 1.0);
    /// assert_eq!(p.x, 3.5);
    /// assert_eq!(p.y, -2.2);
    /// assert_eq!(p.z, 1.0);
    /// ```
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Point3 { x, y, z }
    }

    /// Computes the euclidean distance between two points, by computing the length of the segment
    /// between them
    /// # Examples
    /// ```
    /// use glaze::geometry::Point2;
    /// let p1 = Point2::new(-1., -1.);
    /// let p2 = Point2::new(2., 3.);
    /// assert_eq!(Point2::distance(&p1, &p2), 5.);
    /// ```
    pub fn distance(point_a: &Point3, point_b: &Point3) -> f32 {
        let x = point_b.x - point_a.x;
        let y = point_b.y - point_a.y;
        let z = point_b.z - point_a.z;
        (x * x + y * y + z * z).sqrt()
    }

    /// Calculates the mimimun between two points.
    /// The returned point will have the mimimum value for each component.
    ///
    /// This function is implemented because the canonical std::cmp::min requires the trait Ord
    /// which in turn requires Eq. The trait Eq, however, has subtle implication for floating point
    /// values.
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
    ///
    /// let a = Point3::new(-1.0, 1.0, 0.0);
    /// let b = Point3::new(-2.0, 2.0, 0.0);
    /// let max = Point3::max(&a, &b);
    ///
    /// assert_eq!(max.x, a.x);
    /// assert_eq!(max.y, b.y);
    /// assert_eq!(max.z, 0.0);
    /// ```
    pub fn min(a: &Point3, b: &Point3) -> Point3 {
        Point3 {
            x: a.x.min(b.x),
            y: a.y.min(b.y),
            z: a.z.min(b.z),
        }
    }

    /// Calculates the maximum between two points.
    /// The returned point will have the maximum value for each component.
    ///
    /// This function is implemented because the canonical std::cmp::min requires the trait Ord
    /// which in turn requires Eq. The trait Eq, however, has subtle implication for floating point
    /// values.
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
    ///
    /// let a = Point3::new(-1.0, 1.0, 0.0);
    /// let b = Point3::new(-2.0, 2.0, 0.0);
    /// let max = Point3::max(&a, &b);
    ///
    /// assert_eq!(max.x, a.x);
    /// assert_eq!(max.y, b.y);
    /// assert_eq!(max.z, 0.0);
    /// ```
    pub fn max(a: &Point3, b: &Point3) -> Point3 {
        Point3 {
            x: a.x.max(b.x),
            y: a.y.max(b.y),
            z: a.z.max(b.z),
        }
    }

    /// Transforms the current point with the matrix passed as `mat`
    /// # Examples
    /// ```
    /// use glaze::geometry::{Matrix4, Point3, Vec3};
    ///
    /// let mov = Vec3::new(0.0, -1.0, 0.0);
    /// let magnitude = Vec3::new(5.0, 1.0, 1.0);
    /// let transform = Matrix4::scale(&magnitude) * Matrix4::translation(&mov);
    /// let original = Point3::new(1.0, 1.0, 1.0);
    /// let transformed = original.transform(&transform);
    ///
    /// assert_eq!(transformed.x, 5.0);
    /// assert_eq!(transformed.y, 0.0);
    /// assert_eq!(transformed.z, 1.0);
    /// ```
    pub fn transform(&self, mat: &Matrix4) -> Self {
        let mut x = mat.m[00] * self.x + mat.m[01] * self.y + mat.m[02] * self.z + mat.m[03];
        let mut y = mat.m[04] * self.x + mat.m[05] * self.y + mat.m[06] * self.z + mat.m[07];
        let mut z = mat.m[08] * self.x + mat.m[09] * self.y + mat.m[10] * self.z + mat.m[11];
        let mut w = mat.m[12] * self.x + mat.m[13] * self.y + mat.m[14] * self.z + mat.m[15];
        if !float_eq(w, 1.0, 1E-5) {
            w = 1.0 / w;
            x *= w;
            y *= w;
            z *= w;
        }
        Point3 { x, y, z }
    }

    /// Transforms the point with the matrix passed as `mat`, tracking the floating point error.
    ///
    /// Not public as this is used only internally by the ray class
    pub(super) fn transform_with_error(&self, mat: &Matrix4) -> ErrorTrackedPoint3 {
        let mut x = mat.m[00] * self.x + mat.m[01] * self.y + mat.m[02] * self.z + mat.m[03];
        let mut y = mat.m[04] * self.x + mat.m[05] * self.y + mat.m[06] * self.z + mat.m[07];
        let mut z = mat.m[08] * self.x + mat.m[09] * self.y + mat.m[10] * self.z + mat.m[11];
        let mut w = mat.m[12] * self.x + mat.m[13] * self.y + mat.m[14] * self.z + mat.m[15];
        let abs_x = (mat.m[00] * self.x).abs()
            + (mat.m[01] * self.y).abs()
            + (mat.m[02] * self.z).abs()
            + mat.m[03];
        let abs_y = (mat.m[04] * self.x).abs()
            + (mat.m[05] * self.y).abs()
            + (mat.m[06] * self.z).abs()
            + mat.m[07];
        let abs_z = (mat.m[08] * self.x).abs()
            + (mat.m[09] * self.y).abs()
            + (mat.m[10] * self.z).abs()
            + mat.m[11];
        if !float_eq(w, 1.0, 1E-5) {
            w = 1.0 / w;
            x *= w;
            y *= w;
            z *= w;
        }
        let gamma3 = gamma(3);
        ErrorTrackedPoint3 {
            value: Point3::new(x, y, z),
            error: Vec3::new(gamma3 * abs_x, gamma3 * abs_y, gamma3 * abs_z),
        }
    }
}

impl std::fmt::Display for Point3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point3[{}, {}, {}]", self.x, self.y, self.z)
    }
}

overload!((a: ?Point3) + (b: ?Vec3) -> Point3 {Point3{x:a.x+b.x,y:a.y+b.y,z:a.z+b.z}});
overload!((a: ?Point3) + (b: ?Point3) -> Vec3 {Vec3{x:a.x+b.x,y:a.y+b.y,z:a.z+b.z}});
overload!((a: ?Point3) - (b: ?Vec3) -> Point3 {Point3{x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}});
overload!((a: ?Point3) - (b: ?Point3) -> Vec3 {Vec3{x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}});
overload!(- (a: ?Point3) -> Point3 {Point3{x:-a.x,y:-a.y, z:-a.z}});
overload!((a: &mut Point3) += (b: ?Vec3) {a.x+=b.x;a.y+=b.y;a.z+=b.z;});
overload!((a: &mut Point3) += (b: ?Point3) {a.x+=b.x;a.y+=b.y;a.z+=b.z;});
overload!((a: &mut Point3) -= (b: ?Vec3) {a.x-=b.x;a.y-=b.y;a.z-=b.z;});
overload!((a: &mut Point3) -= (b: ?Point3) {a.x-=b.x;a.y-=b.y;a.z-=b.z;});
