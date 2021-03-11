use crate::linear::{Point2, Point3, Ray, Vec3};
use crate::shapes::{HitPoint, Intersection, Shape, AABB};
use crate::utility::{quadratic, Ef32};

/// A primitive representing a sphere centered in `(0.0, 0.0, 0.0)` with a radius of `1.0`.
///
/// In order to have bigger or shifted spheres one should wrap a transformation matrix along with
/// this shape in an `Asset` type.
///
/// The object-space for this primitive is pivoted in the exact center of the sphere, with the
/// z-axis pointing up and the y-axis pointing front.
///
/// This representation distinguishes between inside and outside of the sphere (mostly for
/// light-emission issues). Be sure to call the correct constructor, either `new` or
/// `inverted`.
pub struct Sphere {
    /// true if the sphere is inverted
    inverted: bool,
}

impl Sphere {
    /// Creates a new Sphere with radius 1.0 .
    #[allow(clippy::new_without_default)] // it would suck only for this class out of all Shapes...
    pub fn new() -> Sphere {
        Sphere { inverted: false }
    }

    /// Creates a new Sphere with radius 1.0, but the sphere will be flipped.
    ///
    /// Every normal will point towards the inside.
    pub fn inverted() -> Sphere {
        Sphere { inverted: true }
    }
}

/// calculate uvs and differential on hit point for a specific hit point
fn compute_interaction(hit: &Point3) -> (Point2, Vec3, Vec3) {
    // this method assumes ISO 80000-2 for spherical coordinates:
    // axis z points up
    // the value `phi` (ϕ in the comments) in this method represents the "longitude"
    // the value `theta` (θ in the comments) in this method represents the "latitude"
    const THETA_MIN: f32 = std::f32::consts::PI;
    const THETA_MAX: f32 = 0.0;
    const THETAD: f32 = THETA_MAX - THETA_MIN;
    const INV_THETAD: f32 = 1.0 / THETAD;
    const TWO_PI: f32 = std::f32::consts::PI * 2.0;
    const INV_TWO_PI: f32 = 1.0 / TWO_PI;
    // calculate ϕ using the (x,y) coordinate of the hit point as tangent.
    // use always a positive value, so add 2π if that's not the case.
    let mut phi = hit.y.atan2(hit.x);
    if phi < 0.0 {
        phi += TWO_PI;
    }
    // calculates θ by using the z value of the hit point as tangent.
    // although the radius is in [-1.0, 1.0] clamping is necessary due to fp errors.
    let theta = hit.z.clamp(-1.0, 1.0).acos();
    // for the uv just use the ϕ and θ mapped in the range [0.0, 1.0].
    // ϕ is the longitude so the interval is [0, 2π]. So divide by 1/2π.
    // θ is the latitude so the interval is [0, π]. So divide by 1/π.
    // multiplying by the multiplicative inverse is faster with fp arithmetic.
    let uv = Point2::new(phi * INV_TWO_PI, (theta - THETA_MIN) * INV_THETAD);
    // no space to write the entire math here... figure it out by yourself :^)
    let invzrad = 1.0 / (hit.x * hit.x + hit.y * hit.y).sqrt();
    let cosphi = hit.x * invzrad;
    let sinphi = hit.y * invzrad;
    let dpdu = Vec3::new(-TWO_PI * hit.y, TWO_PI * hit.x, 0.0);
    let dpdv = Vec3::new(hit.z * cosphi, hit.z * sinphi, (-theta).sin());
    (uv, dpdu, dpdv * THETAD)
}

impl Shape for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        // sphere equation is x² + y² + z² - 1 = 0
        // ray equation is origin + distance * direction where distance >= 0
        // we replace each term x, y and z with the particular component of the ray equation and
        // solve for the distance parameter
        let origin_x = Ef32::new(ray.origin.x, ray.error_origin.x);
        let origin_y = Ef32::new(ray.origin.y, ray.error_origin.y);
        let origin_z = Ef32::new(ray.origin.z, ray.error_origin.z);
        let dir_x = Ef32::new(ray.direction.x, ray.error_direction.x);
        let dir_y = Ef32::new(ray.direction.y, ray.error_direction.y);
        let dir_z = Ef32::new(ray.direction.z, ray.error_direction.z);
        let one = Ef32::new(1.0, 0.0);
        let two = Ef32::new(2.0, 0.0);

        let a = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);
        let b = two * ((dir_x * origin_x) + (dir_y * origin_y) + (dir_z * origin_z));
        let c = (origin_x * origin_x) + (origin_y * origin_y) + (origin_z * origin_z) - one;

        let intersection = quadratic(a, b, c);
        let closest;
        if let Some(intersection) = intersection {
            if intersection.1.lower() < 0.0 {
                // intersection happened behind ray origin
                closest = f32::INFINITY;
            } else if intersection.0.lower() < 0.0 {
                closest = intersection.1.value();
            } else {
                closest = intersection.0.value();
            }
        } else {
            closest = f32::INFINITY;
        };

        // now we have the intersection distance. Finding the hit point is as simple as following
        // the ray direction starting from the origin for the found distance. This was already
        // implemented in the method `point_along`
        if closest != f32::INFINITY {
            let mut hit_point = ray.point_along(closest);
            // the next lines refines the hit point accounting for the error
            let refine_offset = 1.0 / Point3::distance(&Point3::zero(), &hit_point);
            hit_point.x *= refine_offset;
            hit_point.y *= refine_offset;
            hit_point.z *= refine_offset;
            // the normal is just the hit point... given that we have a radius of 1.
            // flip the direction if the sphere is inverted (very unlikely)
            let normal = if !self.inverted {
                Vec3::new(hit_point.x, hit_point.y, hit_point.z)
            } else {
                -Vec3::new(hit_point.x, hit_point.y, hit_point.z)
            };
            // add an epsilon value to avoid a non-differentiable point in the sphere
            if hit_point.x == 0.0 && hit_point.y == 0.0 {
                hit_point.x = 1E-5;
            }
            // compute uvs and differential for the intersection point
            let interaction = compute_interaction(&hit_point);
            Some(Intersection {
                distance: closest,
                hit: HitPoint {
                    point: hit_point,
                    normal,
                    uv: interaction.0,
                    dpdu: interaction.1,
                    dpdv: interaction.2,
                },
            })
        } else {
            None
        }
    }

    fn intersect_fast(&self, ray: &Ray) -> bool {
        // just check the intersect method for explanation
        let origin_x = Ef32::new(ray.origin.x, ray.error_origin.x);
        let origin_y = Ef32::new(ray.origin.y, ray.error_origin.y);
        let origin_z = Ef32::new(ray.origin.z, ray.error_origin.z);
        let dir_x = Ef32::new(ray.direction.x, ray.error_direction.x);
        let dir_y = Ef32::new(ray.direction.y, ray.error_direction.y);
        let dir_z = Ef32::new(ray.direction.z, ray.error_direction.z);
        let one = Ef32::new(1.0, 0.0);
        let two = Ef32::new(2.0, 0.0);

        let a = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);
        let b = two * ((dir_x * origin_x) + (dir_y * origin_y) + (dir_z * origin_z));
        let c = (origin_x * origin_x) + (origin_y * origin_y) + (origin_z * origin_z) - one;

        let solutions = quadratic(a, b, c);
        if let Some(intersection) = solutions {
            intersection.1.lower() >= 0.0
        } else {
            false
        }
    }

    fn bounding_box(&self) -> AABB {
        AABB {
            bot: Point3::new(-1.0, -1.0, -1.0),
            top: Point3::new(1.0, 1.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linear::{Point3, Ray, Vec3};
    use crate::shapes::{Shape, Sphere};

    #[test]
    fn sphere_bounding_box() {
        let sphere = Sphere::new();
        let aabb = sphere.bounding_box();
        assert_eq!(aabb.bot.x, -1.0);
        assert_eq!(aabb.bot.y, -1.0);
        assert_eq!(aabb.bot.z, -1.0);
        assert_eq!(aabb.top.x, 1.0);
        assert_eq!(aabb.top.y, 1.0);
        assert_eq!(aabb.top.z, 1.0);
    }

    #[test]
    fn sphere_intersect_hit_origin_before() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::new(0.0, -10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let res = sphere.intersect(&ray);
        assert!(res.is_some());
        let h = res.unwrap();
        assert_eq!(h.distance, 9.0);
        assert_eq!(h.hit.point.x, 0.0);
        assert_eq!(h.hit.point.y, -1.0);
        assert_eq!(h.hit.point.z, 0.0);
        assert_eq!(h.hit.normal.x, 0.0);
        assert_eq!(h.hit.normal.y, -1.0);
        assert_eq!(h.hit.normal.z, 0.0);
        let dpdu = h.hit.dpdu.normalize();
        let dpdv = h.hit.dpdv.normalize();
        assert_eq!(dpdu.x, 1.0);
        assert_eq!(dpdu.y, 0.0);
        assert_eq!(dpdu.z, 0.0);
        assert_eq!(dpdv.x, 0.0);
        assert_eq!(dpdv.y, 0.0);
        assert_eq!(dpdv.z, 1.0);
        assert_eq!(h.hit.uv.x, 0.75);
        assert_eq!(h.hit.uv.y, 0.5);
    }

    #[test]
    fn sphere_intersect_hit_origin_inside() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::zero(), &Vec3::new(1.0, 0.0, 0.0));
        let res = sphere.intersect(&ray);
        assert!(res.is_some());
        let h = res.unwrap();
        assert_eq!(h.distance, 1.0);
        assert_eq!(h.hit.point.x, 1.0);
        assert_eq!(h.hit.point.y, 0.0);
        assert_eq!(h.hit.point.z, 0.0);
        assert_eq!(h.hit.normal.x, 1.0);
        assert_eq!(h.hit.normal.y, 0.0);
        assert_eq!(h.hit.normal.z, 0.0);
        let dpdu = h.hit.dpdu.normalize();
        let dpdv = h.hit.dpdv.normalize();
        assert_eq!(dpdu.x, 0.0);
        assert_eq!(dpdu.y, 1.0);
        assert_eq!(dpdu.z, 0.0);
        assert_eq!(dpdv.x, 0.0);
        assert_eq!(dpdv.y, 0.0);
        assert_eq!(dpdv.z, 1.0);
        assert_eq!(h.hit.uv.x, 0.0);
        assert_eq!(h.hit.uv.y, 0.5);
    }

    #[test]
    fn sphere_intersect_hit_non_differentiable() {
        //hit in the exact vertical axis of the sphere
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::zero(), &Vec3::new(0.0, 0.0, 1.0));
        let res = sphere.intersect(&ray);
        assert!(res.is_some());
        let h = res.unwrap();
        assert_eq!(h.distance, 1.0);
        assert_eq!(h.hit.point.x, 1E-5);
        assert_eq!(h.hit.point.y, 0.0);
        assert_eq!(h.hit.point.z, 1.0);
        assert_eq!(h.hit.normal.x, 0.0);
        assert_eq!(h.hit.normal.y, 0.0);
        assert_eq!(h.hit.normal.z, 1.0);
        let dpdu = h.hit.dpdu.normalize();
        let dpdv = h.hit.dpdv.normalize();
        assert_eq!(dpdu.x, 0.0);
        assert_eq!(dpdu.y, 1.0);
        assert_eq!(dpdu.z, 0.0);
        assert_eq!(dpdv.x, -1.0);
        assert_eq!(dpdv.y, 0.0);
        assert_eq!(dpdv.z, 0.0);
        assert_eq!(h.hit.uv.x, 0.0);
        assert_eq!(h.hit.uv.y, 1.0);
    }

    #[test]
    fn sphere_intersect_miss() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::new(-2.0, -2.0, -10.0), &Vec3::new(0.0, 0.0, 1.0));
        let res = sphere.intersect(&ray);
        assert!(res.is_none());
    }

    #[test]
    //would be hit, but origin is beyond sphere => so it's miss
    fn sphere_intersect_origin_after() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::new(0.0, 10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let res = sphere.intersect(&ray);
        assert!(res.is_none());
    }

    #[test]
    fn sphere_intersect_fast_hit_origin_before() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::new(0.0, -10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let res = sphere.intersect_fast(&ray);
        assert!(res);
    }

    #[test]
    fn sphere_intersect_fast_hit_origin_inside() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::zero(), &Vec3::new(1.0, 0.0, 0.0));
        let res = sphere.intersect_fast(&ray);
        assert!(res);
    }

    #[test]
    //would be hit, but origin is beyond sphere => so it's miss
    fn sphere_intersect_fast_origin_after() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::new(0.0, 10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let res = sphere.intersect_fast(&ray);
        assert!(!res);
    }

    #[test]
    fn sphere_intersect_fast_miss() {
        let sphere = Sphere::new();
        let ray = Ray::new(&Point3::new(-2.0, -2.0, -10.0), &Vec3::new(0.0, 0.0, 1.0));
        let res = sphere.intersect_fast(&ray);
        assert!(!res);
    }

    #[test]
    fn sphere_inverted_normal() {
        let sphere = Sphere::inverted();
        let ray = Ray::new(&Point3::new(0.0, -10.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let res = sphere.intersect(&ray);
        assert!(res.is_some());
        let h = res.unwrap();
        assert_eq!(h.hit.normal.x, 0.0);
        assert_eq!(h.hit.normal.y, 1.0);
        assert_eq!(h.hit.normal.z, 0.0);
    }
}
