use crate::linear::{Normal, Point2, Point3, Ray, Vec3};
use crate::shapes::AABB;

/// Represents the point in space hit by a Ray, along with some properties.
pub struct HitPoint {
    /// The hit point. Could be either in world space or object space.
    pub point: Point3,

    /// The geometric normal of the hit point. Could be either world space or object space.
    pub normal: Normal,

    /// Differential of hit point, varying the x coordinate on the surface.
    pub dpdu: Vec3,

    /// Differential of hit point, varying the y coordinate on the surface.
    pub dpdv: Vec3,

    /// Mapping coordinate u of the point, used for texture mapping.
    pub uv: Point2,
}

/// Type representing the result of an intersection between a Ray and a Shape.
pub struct Intersection {
    /// The distance of the closest hit point from the ray.
    pub distance: f32,
    /// Data about the hit point.
    pub hit: HitPoint,
}

/// A trait used to represent a geometric primitive in object-space.
pub trait Shape {
    /// Intersects a Ray with the current shape.
    ///
    /// In its implementations, this method should try to intersect a ray passed as a parameter with
    /// the shape, returning if the intersection happened in the range defined by the minimum
    /// epsilon and the current distance.
    ///
    /// This method should guarantee that the field `distance` on the returned `Intersection` is
    /// always greater than 0 and finite, if the returned value is not `None`. No such guarantee
    /// is necessary for `far_distance`.
    ///
    /// The ray should be in the object space of the primitive being intersected
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::linear::{Point3, Ray, Vec3};
    /// use glaze::shapes::{Shape, Sphere};
    ///
    /// let ray = Ray::new(&Point3::new(0.0, -10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
    /// let sphere = Sphere::new();
    /// let intersection = sphere.intersect(&ray);
    ///
    /// assert!(intersection.is_some());
    /// assert_eq!(intersection.unwrap().distance, 9.0);
    /// ```
    // TODO: add MaskBoolean as parameter for alpha masking after porting the textures
    fn intersect(&self, ray: &Ray) -> Option<Intersection>;

    /// Intersects a Ray with the current shape.
    ///
    /// Unlike the intersect method, this one should return just true or false if any kind of
    /// intersection happened. Setting the Intersection value is expensive and sometimes not needed.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::linear::{Point3, Ray, Vec3};
    /// use glaze::shapes::{Shape, Sphere};
    ///
    /// let ray = Ray::new(&Point3::new(0.0, -10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
    /// let sphere = Sphere::new();
    /// let intersection = sphere.intersect_fast(&ray);
    ///
    /// assert!(intersection);
    /// ```
    // TODO: add MaskBoolean as parameter for alpha masking after porting the textures
    fn intersect_fast(&self, ray: &Ray) -> bool;

    /// Returns the AABB for this shape.
    ///
    /// In its implementations, this method should return an AABB that can fit well on this shape
    /// in object space.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::shapes::{Shape, Sphere};
    ///
    /// let sphere = Sphere::new();
    /// let aabb = sphere.bounding_box();
    ///
    /// assert_eq!(aabb.volume(), 8.0);
    /// ```
    fn bounding_box(&self) -> AABB;
}

/// Exactly like the Shape trait, but accepts an optional buffer parameter in every method
///
/// Useful for shapes like Triangle that needs a buffer containing the actual data.
///
/// Since Triangle is private, so is this trait.
pub(crate) trait BufferedShape {
    /// Same as [Shape::intersect] but with an extra VertexBuffer parameter
    fn intersect(&self, ray: &Ray, vb: &VertexBuffer) -> Option<Intersection>;
    /// Same as [Shape::intersect_fast] but with an extra VertexBuffer parameter
    fn intersect_fast(&self, ray: &Ray, vb: &VertexBuffer) -> bool;
    /// Same as [Shape::bounding_box] but with an extra VertexBuffer parameter
    fn bounding_box(&self, vb: &VertexBuffer) -> AABB;
}

/// Struct used to store vertices of a triangle
pub struct VertexBuffer {
    /// Vertex coordinates buffer.
    pub point_buffer: Vec<Point3>,
    /// Vertex textures buffer.
    pub texture_buffer: Vec<Point2>,
    /// Normal buffer.
    pub normal_buffer: Vec<Normal>,
}

// TODO: add `EmittiveShape: Shape` when porting AreaLight class
