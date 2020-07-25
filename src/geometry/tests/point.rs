use crate::geometry::{Point2, Point3, Vec2, Vec3};
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
