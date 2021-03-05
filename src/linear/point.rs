use crate::linear::matrix::{Matrix4, Transform3};
use crate::linear::vec::Vec2;
use crate::linear::vec::Vec3;
use crate::utility::float_eq;
use overload::overload;
use std::fmt::Formatter;
use std::ops;
use std::ops::{Index, IndexMut};

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
    /// Single precision floating point representing the `x` coordinate of the point. Indexed by 0.
    pub x: f32,
    /// Single precision floating point representing the `y` coordinate of the point. Indexed by 1.
    pub y: f32,
}

impl Point2 {
    /// Construct a point in the origin of the cartesian space, with coordinates `(0.0, 0.0)`
    /// # Examples
    /// ```
    /// use glaze::linear::Point2;
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
    /// use glaze::linear::Point2;
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
    /// use glaze::linear::Point2;
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
    /// use glaze::linear::Point2;
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
    /// use glaze::linear::Point2;
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

impl Index<u8> for Point2 {
    type Output = f32;

    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Out of bounds index {} for Point2", index),
        }
    }
}

impl IndexMut<u8> for Point2 {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Out of bounds index {} for Point2", index),
        }
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
    /// Single precision floating point representing the `x` coordinate of the point. Indexed by 0.
    pub x: f32,
    /// Single precision floating point representing the `y` coordinate of the point. Indexed by 1.
    pub y: f32,
    /// Single precision floating point representing the `z` coordinate of the point. Indexed by 2.
    pub z: f32,
}

impl Point3 {
    /// Constructs a point in the Origin of the cartesian space, with coordinates (0.0, 0.0, 0.0)
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
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
    /// use glaze::linear::Point3;
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
    /// use glaze::linear::Point2;
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
    /// use glaze::linear::Point3;
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
    /// use glaze::linear::Point3;
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
}

impl Transform3 for Point3 {
    fn transform(&self, mat: &Matrix4) -> Self {
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
}

impl std::fmt::Display for Point3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point3[{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl Index<u8> for Point3 {
    type Output = f32;

    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Out of bounds index {} for Point3", index),
        }
    }
}

impl IndexMut<u8> for Point3 {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Out of bounds index {} for Point3", index),
        }
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

#[cfg(test)]
mod tests {
    use crate::linear::matrix::Transform3;
    use crate::linear::{Matrix4, Point2, Point3, Vec2, Vec3};
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn point2_zero_constructor() {
        let p = Point2::zero();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn point2_coordinates_constructor() {
        let p = Point2::new(1.0, -1.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, -1.0);
    }

    #[test]
    fn point2_distance() {
        let p0 = Point2::new(1.0, 2.0);
        let p1 = Point2::new(4.0, 5.0);
        let distance = Point2::distance(&p0, &p1);
        assert_approx_eq!(distance, 4.242641);
    }

    #[test]
    fn point2_min() {
        let sample = Point2::new(0.5, 1.5);
        let mut compare;
        let mut min;
        //x is max
        compare = Point2::new(0.2, 0.0);
        min = Point2::min(&sample, &compare);
        assert_eq!(min.x, compare.x);
        //x is not max
        compare = Point2::new(1.0, 0.0);
        min = Point2::min(&sample, &compare);
        assert_eq!(min.x, sample.x);
        //y is max
        compare = Point2::new(0.0, 1.0);
        min = Point2::min(&sample, &compare);
        assert_eq!(min.y, compare.y);
        //y is not max
        compare = Point2::new(0.0, 10.0);
        min = Point2::min(&sample, &compare);
        assert_eq!(min.y, sample.y);
    }

    #[test]
    fn point2_max() {
        let sample = Point2::new(0.5, 1.5);
        let mut compare;
        let mut max;
        //x is max
        compare = Point2::new(0.20, 0.0);
        max = Point2::max(&sample, &compare);
        assert_eq!(max.x, sample.x);
        //x is not max
        compare = Point2::new(1.0, 0.0);
        max = Point2::max(&sample, &compare);
        assert_eq!(max.x, compare.x);
        //y is max
        compare = Point2::new(0.0, 1.0);
        max = Point2::max(&sample, &compare);
        assert_eq!(max.y, sample.y);
        //y is not max
        compare = Point2::new(0.0, 10.0);
        max = Point2::max(&sample, &compare);
        assert_eq!(max.y, compare.y);
    }

    #[test]
    fn point2_index() {
        let v = Point2::new(0.3, 0.6);
        assert_eq!(v[0], 0.3);
        assert_eq!(v[1], 0.6);
    }

    #[test]
    fn point2_index_mut() {
        let mut v = Point2::new(0.9, 0.0);
        v[0] = 0.2;
        v[1] = 0.5;
        assert_eq!(v.x, 0.2);
        assert_eq!(v.y, 0.5);
    }

    #[test]
    fn point2_sum_vector() {
        let v1 = Point2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, 5.5);
        let res = v1 + v2;
        assert_approx_eq!(res.x, 5.0);
        assert_approx_eq!(res.y, 7.5);
    }

    #[test]
    fn point2_sum_vector_this() {
        let mut v1 = Point2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, 5.5);

        v1 += v2;
        assert_approx_eq!(v1.x, 5.0);
        assert_approx_eq!(v1.y, 7.5);
    }

    #[test]
    fn point2_sub_point() {
        let v1 = Point2::new(1.0, 2.0);
        let v2 = Point2::new(4.0, 5.5);
        let res = v2 - v1;
        assert_approx_eq!(res.x, 3.0);
        assert_approx_eq!(res.y, 3.5);
    }

    #[test]
    fn point2_sub_vector() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Point2::new(4.0, 5.5);
        let res = v2 - v1;
        assert_approx_eq!(res.x, 3.0);
        assert_approx_eq!(res.y, 3.5);
    }

    #[test]
    fn point2_sub_vector_this() {
        let mut v1 = Point2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, 5.5);
        v1 -= v2;
        assert_approx_eq!(v1.x, -3.0);
        assert_approx_eq!(v1.y, -3.5);
    }

    #[test]
    fn point2_display() {
        let p = Point2::new(0.1, 1.2);
        let str = format!("{}", &p);
        assert_eq!(str, "Point2[0.1, 1.2]");
    }

    #[test]
    fn point3_zero_constructor() {
        let p = Point3::zero();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn point3_coordinates_constructor() {
        let p = Point3::new(1.0, -1.0, 0.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, -1.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn point3_distance() {
        let p0 = Point3::new(1.0, 2.0, 3.0);
        let p1 = Point3::new(4.0, 5.0, 6.0);
        let distance = Point3::distance(&p0, &p1);
        assert_approx_eq!(distance, 5.196152);
    }

    #[test]
    fn point3_min() {
        let sample = Point3::new(0.5, 1.5, -3.5);
        let mut value;
        let mut compare;
        //x is max
        compare = Point3::new(0.2, 0.0, 0.0);
        value = Point3::min(&sample, &compare);
        assert_eq!(value.x, compare.x);
        //x is not max
        compare = Point3::new(1.0, 0.0, 0.0);
        value = Point3::min(&sample, &compare);
        assert_eq!(value.x, sample.x);
        //y is max
        compare = Point3::new(0.0, 1.0, 0.0);
        value = Point3::min(&sample, &compare);
        assert_eq!(value.y, compare.y);
        //y is not max
        compare = Point3::new(0.0, 10.0, 0.0);
        value = Point3::min(&sample, &compare);
        assert_eq!(value.y, sample.y);
        //z is max
        compare = Point3::new(0.0, 0.0, -5.0);
        value = Point3::min(&sample, &compare);
        assert_eq!(value.z, compare.z);
        //z is not max
        compare = Point3::new(0.0, 0.0, 0.0);
        value = Point3::min(&sample, &compare);
        assert_eq!(value.z, sample.z);
    }

    #[test]
    fn point3_max() {
        let sample = Point3::new(0.5, 1.5, -3.5);
        let mut value;
        let mut compare;

        //x is max
        compare = Point3::new(0.2, 0.0, 0.0);
        value = Point3::max(&sample, &compare);
        assert_eq!(value.x, sample.x);
        //x is not max
        compare = Point3::new(1.0, 0.0, 0.0);
        value = Point3::max(&sample, &compare);
        assert_eq!(value.x, compare.x);
        //y is max
        compare = Point3::new(0.0, 1.0, 0.0);
        value = Point3::max(&sample, &compare);
        assert_eq!(value.y, sample.y);
        //y is not max
        compare = Point3::new(0.0, 10.0, 0.0);
        value = Point3::max(&sample, &compare);
        assert_eq!(value.y, compare.y);
        //z is max
        compare = Point3::new(0.0, 0.0, -5.0);
        value = Point3::max(&sample, &compare);
        assert_eq!(value.z, sample.z);
        //z is not max
        compare = Point3::new(0.0, 0.0, 0.0);
        value = Point3::max(&sample, &compare);
        assert_eq!(value.z, compare.z);
    }

    #[test]
    fn point3_index() {
        let v = Vec3::new(0.4, 0.8, 0.6);
        assert_eq!(v[0], 0.4);
        assert_eq!(v[1], 0.8);
        assert_eq!(v[2], 0.6);
    }

    #[test]
    fn point3_index_mut() {
        let mut v = Vec3::new(0.9, 0.1, 0.5);
        v[0] = 0.6;
        v[1] = 0.7;
        v[2] = 0.1;
        assert_eq!(v.x, 0.6);
        assert_eq!(v.y, 0.7);
        assert_eq!(v.z, 0.1);
    }

    #[test]
    fn point3_sum_vector_this() {
        let mut v1 = Point3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.5, -3.0);

        v1 += v2;
        assert_approx_eq!(v1.x, 5.0);
        assert_approx_eq!(v1.y, 7.5);
        assert_approx_eq!(v1.z, 0.0);
    }

    #[test]
    fn point3_sub_point() {
        let v1 = Point3::new(1.0, 2.0, 3.0);
        let v2 = Point3::new(4.0, 5.5, -3.0);
        let res = v2 - v1;
        assert_approx_eq!(res.x, 3.0);
        assert_approx_eq!(res.y, 3.5);
        assert_approx_eq!(res.z, -6.0);
    }

    #[test]
    fn point3_sub_vector() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Point3::new(4.0, 5.5, -3.0);
        let res = v2 - v1;
        assert_approx_eq!(res.x, 3.0);
        assert_approx_eq!(res.y, 3.5);
        assert_approx_eq!(res.z, -6.0);
    }

    #[test]
    fn point3_sub_vector_this() {
        let mut v1 = Point3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.5, -3.0);
        v1 -= v2;
        assert_approx_eq!(v1.x, -3.0);
        assert_approx_eq!(v1.y, -3.5);
        assert_approx_eq!(v1.z, 6.0);
    }

    #[test]
    fn point3_display() {
        let p = Point3::new(0.1, 1.2, 2.3);
        let str = format!("{}", &p);
        assert_eq!(str, "Point3[0.1, 1.2, 2.3]");
    }

    #[test]
    //w component is what really differentiate a Point3(x,y,z) from a Vec3(x,y,z).
    fn point3_transform_no_w_component() {
        //no w component
        let p = Point3::new(1.0, 1.0, 1.0);
        let m = Matrix4::scale(&Vec3::new(3.0, 3.0, 3.0));
        let transformed = p.transform(&m);
        assert_eq!(transformed.x, 3.0);
        assert_eq!(transformed.y, 3.0);
        assert_eq!(transformed.z, 3.0);
    }

    #[test]
    //w component is what really differentiate a Point3(x,y,z) from a Vec3(x,y,z).
    //it appears in certain perspective matrix, here I faked it.
    fn point3_transform_with_w_component() {
        let p = Point3::new(0.0, 1.0, 1.0);
        let mut m = Matrix4::translation(&Vec3::new(0.0, -1.0, 2.5));
        m.m[12] = 1.0;
        m.m[14] = 1.0;
        let transformed = p.transform(&m);
        assert_eq!(transformed.x, 0.0);
        assert_eq!(transformed.y, 0.0);
        assert_approx_eq!(transformed.z, 3.5 / 2.0);
    }
}
