use crate::geometry::{Normal, Point2, Point3, Ray, Vec3};
use crate::shapes::AABB;

/// Represents the point in space hit by a ray, along with some of its properties.
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
    /// The distance of the hit point from the ray.
    pub distance: f32,
    /// Data about the hit point.
    pub hit: HitPoint,
}

/// A trait used to represent a geometric primitive (Sphere, Triangle, etc.) in object-space.
pub trait Shape {
    /// Returns the ID of this shape.
    fn get_id(&self) -> usize;

    /// Intersects a Ray with the current shape.
    ///
    /// In its implementations, this method should try to intersect a ray passed as a parameter with
    /// the shape, returning if the intersection happened in the range defined by the minimum
    /// epsilon and the current distance.
    ///
    /// **NOTE**: If there is an intersection, but it is outside the range defined
    /// by SELF_INTERSECTION_ERROR and the current distance value, this method
    /// should return None.
    ///
    /// The ray should be in the object space of the primitive being intersected
    // TODO: add MaskBoolean as parameter for alpha masking after porting the textures
    fn intersect(&self, ray: &Ray) -> Option<Intersection>;

    /// Intersects a Ray with the current shape.
    ///
    /// Unlike the intersect method, this one should return just true or false if any kind of
    /// intersection happened. Setting the Intersection value is expensive and sometimes not needed.
    // TODO: add MaskBoolean as parameter for alpha masking after porting the textures
    fn intersect_fast(&self, ray: &Ray) -> bool;

    /// Returns the AABB for this shape.
    ///
    /// In its implementations, this method should return an AABB that can fit well on this shape
    /// in object space.
    fn bounding_box(&self) -> AABB;
}

// TODO: add `EmittiveShape: Shape` when porting AreaLight class
