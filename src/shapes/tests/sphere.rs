use crate::geometry::{Point3, Ray, Vec3};
use crate::shapes::{Shape, Sphere, ID_SPHERE, ID_SPHERE_INVERTED};

#[test]
fn sphere_get_id() {
    let s0 = Sphere::new();
    assert_eq!(s0.get_id(), ID_SPHERE);
}

#[test]
fn sphere_inverted_get_id() {
    let s0 = Sphere::inverted();
    assert_eq!(s0.get_id(), ID_SPHERE_INVERTED);
}

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
