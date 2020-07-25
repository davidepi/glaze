use crate::geometry::{Matrix4, Point3, Ray, Vec3};
use assert_approx_eq::assert_approx_eq;

#[test]
fn ray_default_constructor() {
    let r = Ray::zero();
    assert_eq!(r.origin.x, 0.0);
    assert_eq!(r.origin.y, 0.0);
    assert_eq!(r.origin.z, 0.0);
    assert_eq!(r.direction.x, 0.0);
    assert_eq!(r.direction.y, 0.0);
    assert_eq!(r.direction.z, 1.0);
    assert_eq!(r.bounces, 0);
}

#[test]
fn ray_vector_constructor() {
    let origin = Point3::new(-1.0, 2.4, 5.0);
    let direction = Vec3::new(-8.0, -3.50, 0.1);
    let r = Ray::new(&origin, &direction);
    assert_eq!(r.origin.x, origin.x);
    assert_eq!(r.origin.y, origin.y);
    assert_eq!(r.origin.z, origin.z);
    assert_eq!(r.direction.x, direction.x);
    assert_eq!(r.direction.y, direction.y);
    assert_eq!(r.direction.z, direction.z);
    assert_eq!(r.bounces, 0);
}

#[test]
fn ray_point_along() {
    let origin = Point3::zero();
    let direction = Vec3::new(0.408248, 0.408248, 0.816497);
    let distance = 2.5;
    let r = Ray::new(&origin, &direction);
    let point = r.point_along(distance);
    assert_approx_eq!(point.x, 1.02062);
    assert_approx_eq!(point.y, 1.02062);
    assert_approx_eq!(point.z, 2.041242);
}

#[test]
fn matrix4_transform_ray() {
    let p = Point3::zero();
    let v = Vec3::new(0.0, 1.0, 0.0);
    let ray = Ray::new(&p, &v);
    let m = Matrix4::translation(&Vec3::new(0.0, -1.0, 2.5));
    let transformed_point = p.transform(&m);
    let transformed_vec = v.transform(&m);
    let transformed_ray = ray.transform(&m);
    //assert that transforming a ray is exactly like transforming origin and
    //direction separately
    assert_approx_eq!(transformed_point.x, transformed_ray.origin.x);
    assert_approx_eq!(transformed_point.y, transformed_ray.origin.y);
    assert_approx_eq!(transformed_point.z, transformed_ray.origin.z);
    assert_approx_eq!(transformed_vec.x, transformed_ray.direction.x);
    assert_approx_eq!(transformed_vec.y, transformed_ray.direction.y);
    assert_approx_eq!(transformed_vec.z, transformed_ray.direction.z);
}

#[test]
fn ray_display() {
    let r = Ray::new(
        &Point3::new(0.5, 1.5, 2.5),
        &Vec3::new(1.0, 1.0, 1.0).normalize(),
    );
    let str = format!("{}", &r);
    assert_eq!(
        str,
        "Ray[(0.5, 1.5, 2.5) -> (0.57735026, 0.57735026, 0.57735026)]"
    );
}
