use crate::geometry::{Vec2, Vec3};
use assert_approx_eq::assert_approx_eq;

#[test]
fn vec2_zero_constructor() {
    let v = Vec2::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
}

#[test]
fn vec2_components_constructor() {
    let v = Vec2::new(1.0, 0.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 0.0);
}

#[test]
fn vec2_clone() {
    let v = Vec2::new(-83.27705, 79.29129);
    let v2 = v.clone();
    assert_eq!(v.x, v2.x);
    assert_eq!(v.y, v2.y);
}

#[test]
fn vec2_dot() {
    let v = Vec2::new(1.0, 2.0);
    let v2 = Vec2::new(4.0, -5.0);
    assert_eq!(Vec2::dot(&v, &v2), -6.0);

    let v3 = Vec2::new(6.0, -1.0);
    let v4 = Vec2::new(4.0, 18.0);
    assert_eq!(Vec2::dot(&v3, &v4), 6.0);

    let v5 = Vec2::new(6.0, -1.0);
    let v6 = Vec2::new(4.0, 24.0);
    assert_eq!(Vec2::dot(&v5, &v6), 0.0);
}

#[test]
fn vec2_length() {
    let v = Vec2::new(-15.0, -2.0);
    let length = v.length();
    assert_approx_eq!(length, 15.132745);

    let v1 = Vec2::new(-3.0, 2.0);
    assert_approx_eq!(v1.length2(), 13.0);

    assert_approx_eq!(v1.length() * v1.length(), v1.length2());

    let v2 = Vec2::zero();
    assert_eq!(v2.length(), 0.0);
}

#[test]
fn vec2_normalize() {
    let v1 = Vec2::new(3.0, 1.0);
    assert_approx_eq!(v1.length(), 3.162277);
    assert!(!v1.is_normalized());

    let v2 = v1.normalize();
    assert_approx_eq!(v2.x, 0.948683);
    assert_approx_eq!(v2.y, 0.316228);
    assert!(v2.is_normalized());
}

#[test]
#[should_panic]
fn vec2_normalize_zero() {
    let _ = Vec2::zero().normalize();
}

#[test]
fn vec2_clamp() {
    let sample = Vec2::new(4.9, -5.8);
    let mut clamped;
    let min = Vec2::new(-10.0, -10.0);
    let max = Vec2::new(10.0, 10.0);

    //in range
    clamped = sample.clamp(&min, &max);
    assert_eq!(clamped.x, sample.x);
    assert_eq!(clamped.y, sample.y);
    //min x
    let new_min = Vec2::new(5.0, 5.0);
    clamped = sample.clamp(&new_min, &max);
    assert_eq!(clamped.x, new_min.x);
    assert_eq!(clamped.y, new_min.y);
    //max x
    let new_max = Vec2::new(3.0, -6.0);
    clamped = sample.clamp(&min, &new_max);
    assert_eq!(clamped.x, new_max.x);
    assert_eq!(clamped.y, new_max.y);
}

#[test]
fn vec2_add_vec2() {
    let v1 = Vec2::new(1.0, 2.0);
    let v2 = Vec2::new(4.0, 5.5);
    let res = v1 + v2;
    assert_approx_eq!(res.x, 5.0);
    assert_approx_eq!(res.y, 7.5);
}

#[test]
fn vec2_add_float() {
    let v1 = Vec2::new(1.0, 2.65);
    let f = -0.75;
    let res = v1 + f;
    assert_approx_eq!(res.x, 0.25);
    assert_approx_eq!(res.y, 1.9);
}

#[test]
fn vec2_add_assign_vec2() {
    let mut v1 = Vec2::new(1.0, 2.0);
    let v2 = Vec2::new(4.0, 5.5);
    v1 += v2;
    assert_approx_eq!(v1.x, 5.0);
    assert_approx_eq!(v1.y, 7.5);
}

#[test]
fn vec2_add_assign_float() {
    let mut v1 = Vec2::new(3.3, 1.2);
    let f = 0.5;
    v1 += f;
    assert_approx_eq!(v1.x, 3.8);
    assert_approx_eq!(v1.y, 1.7);
}

#[test]
fn vec2_sub_vec2() {
    let v1 = Vec2::new(7.0, 6.5);
    let v2 = Vec2::new(7.0, 5.3);
    let res = v1 - v2;
    assert_approx_eq!(res.x, 0.0);
    assert_approx_eq!(res.y, 1.2);
}

#[test]
fn vec2_sub_float() {
    let v1 = Vec2::new(3.3, 1.2);
    let f = 0.5;
    let res = v1 - f;
    assert_approx_eq!(res.x, 2.8);
    assert_approx_eq!(res.y, 0.7);
}

#[test]
fn vec2_sub_assign_vec2() {
    let mut v1 = Vec2::new(5.0, 3.0);
    let v2 = Vec2::new(7.0, 3.3);
    v1 -= v2;
    assert_approx_eq!(v1.x, -2.0);
    assert_approx_eq!(v1.y, -0.3);
}

#[test]
fn vec2_sub_assign_float() {
    let mut v1 = Vec2::new(3.3, 1.2);
    let f = 0.5;
    v1 -= f;
    assert_approx_eq!(v1.x, 2.8);
    assert_approx_eq!(v1.y, 0.7);
}

#[test]
fn vec2_mul_float() {
    let v1 = Vec2::new(3.3, 1.2);
    let f = 7.33;
    let res = v1 * f;
    assert_approx_eq!(res.x, 24.189);
    assert_approx_eq!(res.y, 8.796);
}

#[test]
fn vec2_mul_assign_float() {
    let mut v1 = Vec2::new(3.3, 1.2);
    let f = -53.477;
    v1 *= f;
    assert_approx_eq!(v1.x, -176.4741);
    assert_approx_eq!(v1.y, -64.1724);
}

#[test]
fn vec2_neg() {
    let v1 = Vec2::new(6.0, 8.5);
    let res = -v1;
    assert_eq!(res.x, -v1.x);
    assert_eq!(res.y, -v1.y);
}

#[test]
fn vec2_display() {
    let v = Vec2::new(0.1, 1.2);
    let str = format!("{}", &v);
    assert_eq!(str, "Vec2[0.1, 1.2]");
}

#[test]
fn vec3_zero_constructor() {
    let v = Vec3::zero();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);
}

#[test]
fn vec3_components_constructor() {
    let v = Vec3::new(1.0, 0.0, -1.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, -1.0);
}

#[test]
fn vec3_clone() {
    let v = Vec3::new(-83.27705, 79.29129, -51.32018);
    let v2 = v;

    assert_eq!(v.x, v2.x);
    assert_eq!(v.y, v2.y);
    assert_eq!(v.z, v2.z);
}

#[test]
fn vec3_dot() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, -5.0, 6.0);
    assert_eq!(Vec3::dot(&v, &v2), 12.0);

    let v3 = Vec3::new(6.0, -1.0, 3.0);
    let v4 = Vec3::new(4.0, 18.0, -2.0);
    assert_eq!(Vec3::dot(&v3, &v4), 0.0);
}

#[test]
fn vec3_cross() {
    let v0 = Vec3::new(3.0, -3.0, 1.0);
    let v1 = Vec3::new(4.0, 9.0, 2.0);
    let outv = Vec3::cross(&v0, &v1);

    assert_eq!(outv.x, -15.0);
    assert_eq!(outv.y, -2.0);
    assert_eq!(outv.z, 39.0);
}

#[test]
fn vec3_length() {
    let v = Vec3::new(-15.0, -2.0, 39.0);
    let length = v.length();
    assert_approx_eq!(length, 41.833);

    let v1 = Vec3::new(-32.0, -53.0, 23.0);
    assert_eq!(v1.length2(), 4362.0);
    assert_approx_eq!(v1.length() * v1.length(), v1.length2());

    let v2 = Vec3::zero();
    assert_eq!(v2.length(), 0.0);
}

#[test]
fn vec3_normalize() {
    let v1 = Vec3::new(3.0, 1.0, 2.0);
    assert_approx_eq!(v1.length(), 3.741657);
    assert!(!v1.is_normalized());

    let normalized = v1.normalize();
    assert_approx_eq!(normalized.x, 0.801783);
    assert_approx_eq!(normalized.y, 0.267261);
    assert_approx_eq!(normalized.z, 0.534522);
    assert!(normalized.is_normalized());
}

#[test]
#[should_panic]
fn vec3_normalize_zero() {
    let _ = Vec3::zero().normalize();
}

#[test]
fn vec3_clamp() {
    let sample = Vec3::new(4.9, -5.8, 3.6);
    let mut clamped;
    let min = Vec3::new(-10.0, -10.0, -10.0);
    let max = Vec3::new(10.0, 10.0, 10.0);
    //in range
    clamped = sample.clamp(&min, &max);
    assert_eq!(clamped.x, sample.x);
    assert_eq!(clamped.y, sample.y);
    assert_eq!(clamped.z, sample.z);
    //min
    let new_min = Vec3::new(5.0, 5.0, 5.0);
    clamped = sample.clamp(&new_min, &max);
    assert_eq!(clamped.x, new_min.x);
    assert_eq!(clamped.y, new_min.y);
    assert_eq!(clamped.z, new_min.z);
    //max
    let new_max = Vec3::new(4.0, -6.0, 3.0);
    clamped = sample.clamp(&min, &new_max);
    assert_eq!(clamped.x, new_max.x);
    assert_eq!(clamped.y, new_max.y);
    assert_eq!(clamped.z, new_max.z);
}

#[test]
fn vec3_reflect() {
    let sample = Vec3::new(0.5, 0.3, -0.5);
    let centre = Vec3::new(0.0, 0.0, 1.0);
    let reflected = sample.reflect(&centre);
    assert_eq!(reflected.x, sample.x);
    assert_eq!(reflected.y, sample.y);
    assert_eq!(reflected.z, -sample.z);
}

#[test]
fn vec3_refract_tir() {
    let v = Vec3::new(0.5, 0.3, -0.5);
    let interface = Vec3::new(0.0, 0.0, 1.0);
    let eta = 1.45;
    let refracted = v.refract(&interface, eta);
    assert!(refracted.is_none());
}

#[test]
fn vec3_refract_no_tir() {
    let v = Vec3::new(0.3, 0.1, 0.8);
    let interface = Vec3::new(0.0, 0.0, 1.0);
    let eta = 1.45;
    let refracted = v.refract(&interface, eta).unwrap();
    assert_approx_eq!(refracted.x, -0.435);
    assert_approx_eq!(refracted.y, -0.145);
    assert_approx_eq!(refracted.z, -0.493051767);
}

#[test]
fn vec3_add_vec3() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.5, -3.0);
    let res = v1 + v2;
    assert_approx_eq!(res.x, 5.0);
    assert_approx_eq!(res.y, 7.50);
    assert_approx_eq!(res.z, 0.0);
}

#[test]
fn vec3_add_float() {
    let v1 = Vec3::new(1.0, 2.0, -3.0);
    let f = -0.75;
    let res = v1 + f;
    assert_approx_eq!(res.x, 0.25);
    assert_approx_eq!(res.y, 1.25);
    assert_approx_eq!(res.z, -3.75);
}

#[test]
fn vec3_add_assign_vec3() {
    let mut v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.5, -3.0);
    v1 += v2;
    assert_approx_eq!(v1.x, 5.0);
    assert_approx_eq!(v1.y, 7.5);
    assert_approx_eq!(v1.z, 0.0);
}

#[test]
fn vec3_add_assign_float() {
    let mut v1 = Vec3::new(3.3, 1.2, -1.5);
    let f = 0.5;
    v1 += f;
    assert_approx_eq!(v1.x, 3.8);
    assert_approx_eq!(v1.y, 1.7);
    assert_approx_eq!(v1.z, -1.0);
}

#[test]
fn vec3_sub_vec3() {
    let v1 = Vec3::new(7.0, 6.5, -5.0);
    let v2 = Vec3::new(7.0, 5.3, -4.0);

    let res = v1 - v2;
    assert_approx_eq!(res.x, 0.0);
    assert_approx_eq!(res.y, 1.2);
    assert_approx_eq!(res.z, -1.0);
}

#[test]
fn vec3_sub_float() {
    let v1 = Vec3::new(3.3, 1.2, -1.5);
    let f = 0.5;
    let res = v1 - f;
    assert_approx_eq!(res.x, 2.8);
    assert_approx_eq!(res.y, 0.7);
    assert_approx_eq!(res.z, -2.0);
}

#[test]
fn vec3_sub_assign_vec3() {
    let mut v1 = Vec3::new(5.0, 3.0, 1.0);
    let v2 = Vec3::new(7.0, 3.3, 4.0);
    v1 -= v2;
    assert_approx_eq!(v1.x, -2.0);
    assert_approx_eq!(v1.y, -0.3);
    assert_approx_eq!(v1.z, -3.0);
}

#[test]
fn vec3_sub_assign_float() {
    let mut v1 = Vec3::new(3.3, 1.2, -1.5);
    let f = 0.5;
    v1 -= f;
    assert_approx_eq!(v1.x, 2.8);
    assert_approx_eq!(v1.y, 0.7);
    assert_approx_eq!(v1.z, -2.0);
}

#[test]
fn vec3_mul_float() {
    let v1 = Vec3::new(3.3, 1.2, -1.5);
    let f = 7.33;
    let res = v1 * f;
    assert_approx_eq!(res.x, 24.189);
    assert_approx_eq!(res.y, 8.796);
    assert_approx_eq!(res.z, -10.995);
}

#[test]
fn vec3_mul_assign_float() {
    let mut v1 = Vec3::new(3.3, 1.2, -1.5);
    let f = -53.477;
    v1 *= f;
    assert_approx_eq!(v1.x, -176.4741);
    assert_approx_eq!(v1.y, -64.1724);
    assert_approx_eq!(v1.z, 80.2155);
}

#[test]
fn vec3_neg() {
    let v1 = Vec3::new(6.0, 8.5, -3.76);
    let res = -v1;

    assert_approx_eq!(res.x, -6.0);
    assert_approx_eq!(res.y, -8.5);
    assert_approx_eq!(res.z, 3.76);
}

#[test]
fn vec3_display() {
    let v = Vec3::new(0.1, 1.2, 2.3);
    let str = format!("{}", &v);
    assert_eq!(str, "Vec3[0.1, 1.2, 2.3]");
}
