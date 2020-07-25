use crate::geometry::{Matrix4, Point3, Vec3};
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
    ///```
    pub fn zero() -> Ray {
        Ray {
            origin: Point3::zero(),
            direction: Vec3::new(0.0, 0.0, 1.0),
            bounces: 0,
        }
    }

    /// Constructs a Ray with the given origin and direction
    /// # Examples
    /// ```
    /// use glaze::geometry::{Ray, Vec3, Point3};
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
        }
    }

    /// Transforms the Ray using the provided transformation matrix. This translates to transforming
    /// its origin and direction independently.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Ray, Matrix4};
    /// use assert_approx_eq::assert_approx_eq;
    ///
    /// let ray = Ray::zero();
    /// let transform = Matrix4::rotate_x(std::f32::consts::PI);
    /// let transformed = ray.transform(&transform);
    ///
    /// assert_approx_eq!(transformed.origin.x, 0.0);
    /// assert_approx_eq!(transformed.origin.y, 0.0);
    /// assert_approx_eq!(transformed.origin.z, 0.0);
    /// assert_approx_eq!(transformed.direction.x, 0.0);
    /// assert_approx_eq!(transformed.direction.y, 0.0);
    /// assert_approx_eq!(transformed.direction.z, -1.0);
    /// ```
    pub fn transform(&self, matrix: &Matrix4) -> Ray {
        Ray {
            origin: self.origin.transform(&matrix),
            direction: self.direction.transform(&matrix),
            bounces: self.bounces,
        }
    }

    /// Find a point at a particular position along a ray, at a given `distance`from the origin of
    /// the ray.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Ray, Point3, Vec3};
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
