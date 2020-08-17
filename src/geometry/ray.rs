use crate::geometry::matrix::Transform3;
use crate::geometry::{Matrix4, Point3, Vec3};
use crate::utility::{float_eq, gamma};
use std::fmt::{Display, Formatter};

#[derive(Copy, Clone)]
/// A ray defined by origin and destination.
///
/// Ray class represents a semi-infinite line. A Ray is denoted by a Point3 the `origin`,
/// and a Vec3, the `direction`.
pub struct Ray {
    /// The origin of the Ray in space
    pub origin: Point3,
    /// The direction of the Ray in space
    pub direction: Vec3,
    ///An unsigned char representing the maximum number of bounces for this ray.
    ///
    /// In a raytrace-based renderer, rays bounce from a surface to another until they are absorbed.
    /// However, if the scene is full of specular reflecting surfaces, a ray could bounce a lot of
    /// times, if not forever. This value is a sort of "time-to-live" for the ray: when it
    /// reaches a determined value, the ray must be destroyed. However, this is responsibility of
    /// the integrator algorithm and not of the ray itself.
    pub bounces: u8,
    /// The accumulated floating point error (during transformations) on the origin's components.
    /// This is used for more accurate intersections.
    pub(crate) error_origin: Vec3,
    /// The accumulated floating point error (during transformations) on the direction's components.
    /// This is used for more accurate intersections.
    pub(crate) error_direction: Vec3,
}

impl Ray {
    /// Constructs a Ray with origin in (0.0, 0.0, 0.0) and direction towards the positive z axis
    /// # Examples
    /// ```
    /// use glaze::geometry::Ray;
    ///
    /// let ray = Ray::zero();
    ///
    /// assert_eq!(ray.origin.x, 0.0);
    /// assert_eq!(ray.origin.y, 0.0);
    /// assert_eq!(ray.origin.z, 0.0);
    /// assert_eq!(ray.direction.x, 0.0);
    /// assert_eq!(ray.direction.y, 0.0);
    /// assert_eq!(ray.direction.z, 1.0);
    /// ```
    pub fn zero() -> Ray {
        Ray {
            origin: Point3::zero(),
            direction: Vec3::new(0.0, 0.0, 1.0),
            bounces: 0,
            error_origin: Vec3::zero(),
            error_direction: Vec3::zero(),
        }
    }

    /// Constructs a Ray with the given origin and direction
    /// # Examples
    /// ```
    /// use glaze::geometry::{Point3, Ray, Vec3};
    ///
    /// let origin = Point3::new(1.0, 2.0, 3.0);
    /// let direction = Vec3::new(0.0, 1.0, 0.0);
    /// let ray = Ray::new(&origin, &direction);
    ///
    /// assert_eq!(ray.origin.x, origin.x);
    /// assert_eq!(ray.origin.y, origin.y);
    /// assert_eq!(ray.origin.z, origin.z);
    /// assert_eq!(ray.direction.x, direction.x);
    /// assert_eq!(ray.direction.y, direction.y);
    /// assert_eq!(ray.direction.z, direction.z);
    /// ```
    pub fn new(origin: &Point3, direction: &Vec3) -> Ray {
        Ray {
            origin: *origin,
            direction: *direction,
            bounces: 0,
            error_origin: Vec3::zero(),
            error_direction: Vec3::zero(),
        }
    }

    /// Find a point at a particular position along a ray, at a given `distance`from the origin of
    /// the ray.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Point3, Ray, Vec3};
    ///
    /// let ray = Ray::new(&Point3::zero(), &Vec3::new(0.0, 1.0, 0.0));
    /// let point_along_ray = ray.point_along(2.5);
    ///
    /// assert_eq!(point_along_ray.x, 0.0);
    /// assert_eq!(point_along_ray.y, 2.5);
    /// assert_eq!(point_along_ray.z, 0.0);
    /// ```
    pub fn point_along(&self, distance: f32) -> Point3 {
        Point3 {
            x: self.origin.x + self.direction.x * distance,
            y: self.origin.y + self.direction.y * distance,
            z: self.origin.z + self.direction.z * distance,
        }
    }
}

struct ErrorTrackedPoint3 {
    value: Point3,
    error: Vec3,
}

fn transform_point_with_error(pt: &Point3, mat: &Matrix4) -> ErrorTrackedPoint3 {
    let mut x = mat.m[00] * pt.x + mat.m[01] * pt.y + mat.m[02] * pt.z + mat.m[03];
    let mut y = mat.m[04] * pt.x + mat.m[05] * pt.y + mat.m[06] * pt.z + mat.m[07];
    let mut z = mat.m[08] * pt.x + mat.m[09] * pt.y + mat.m[10] * pt.z + mat.m[11];
    let mut w = mat.m[12] * pt.x + mat.m[13] * pt.y + mat.m[14] * pt.z + mat.m[15];
    let abs_x =
        (mat.m[00] * pt.x).abs() + (mat.m[01] * pt.y).abs() + (mat.m[02] * pt.z).abs() + mat.m[03];
    let abs_y =
        (mat.m[04] * pt.x).abs() + (mat.m[05] * pt.y).abs() + (mat.m[06] * pt.z).abs() + mat.m[07];
    let abs_z =
        (mat.m[08] * pt.x).abs() + (mat.m[09] * pt.y).abs() + (mat.m[10] * pt.z).abs() + mat.m[11];
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

struct ErrorTrackedVec3 {
    value: Vec3,
    error: Vec3,
}

fn transform_vec_with_error(vec: &Vec3, mat: &Matrix4) -> ErrorTrackedVec3 {
    let x = mat.m[00] * vec.x + mat.m[01] * vec.y + mat.m[02] * vec.z;
    let y = mat.m[04] * vec.x + mat.m[05] * vec.y + mat.m[06] * vec.z;
    let z = mat.m[08] * vec.x + mat.m[09] * vec.y + mat.m[10] * vec.z;
    let abs_x = (mat.m[00] * vec.x).abs() + (mat.m[01] * vec.y).abs() + (mat.m[02] * vec.z).abs();
    let abs_y = (mat.m[04] * vec.x).abs() + (mat.m[05] * vec.y).abs() + (mat.m[06] * vec.z).abs();
    let abs_z = (mat.m[08] * vec.x).abs() + (mat.m[09] * vec.y).abs() + (mat.m[10] * vec.z).abs();
    let gamma3 = gamma(3);
    ErrorTrackedVec3 {
        value: Vec3::new(x, y, z),
        error: Vec3::new(gamma3 * abs_x, gamma3 * abs_y, gamma3 * abs_z),
    }
}

impl Transform3 for Ray {
    fn transform(&self, mat: &Matrix4) -> Self {
        //this keeps track of errors to avoid self intersections
        let mut origin = transform_point_with_error(&self.origin, &mat);
        let direction = transform_vec_with_error(&self.direction, &mat);
        let length2 = direction.value.length2();
        if length2 > 0.0 {
            // adjust origin based on fp error
            let dt = Vec3::dot(&self.direction.abs(), &origin.error) / length2;
            origin.value += direction.value * dt;
        }
        Ray {
            origin: origin.value,
            direction: direction.value,
            bounces: self.bounces,
            error_origin: origin.error,
            error_direction: direction.error,
        }
    }
}

impl Display for Ray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ray[({}, {}, {}) -> ({}, {}, {})]",
            self.origin.x,
            self.origin.y,
            self.origin.z,
            self.direction.x,
            self.direction.y,
            self.direction.z
        )
    }
}
