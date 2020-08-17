use crate::geometry::{Matrix4, Point3, Transform3};
use crate::shapes::AABB;
use assert_approx_eq::assert_approx_eq;
use std::f32::INFINITY;

#[test]
fn aabb_zero() {
    let aabb = AABB::zero();
    assert!((aabb.bot.x).is_infinite());
    assert!((aabb.bot.y).is_infinite());
    assert!((aabb.bot.z).is_infinite());
    assert!((aabb.top.x).is_infinite());
    assert!((aabb.top.y).is_infinite());
    assert!((aabb.top.z).is_infinite());
}

#[test]
fn aabb_new() {
    let pmin = Point3::new(60.06347, 33.59238, -37.23738);
    let pmax = Point3::new(-21.18293, 50.33405, 9.33384);
    let aabb = AABB::new(&pmin, &pmax);
    assert_eq!(aabb.bot.x, pmax.x);
    assert_eq!(aabb.bot.y, pmin.y);
    assert_eq!(aabb.bot.z, pmin.z);
    assert_eq!(aabb.top.x, pmin.x);
    assert_eq!(aabb.top.y, pmax.y);
    assert_eq!(aabb.top.z, pmax.z);
}

#[test]
fn aabb_from_point() {
    let p = Point3::new(76.9907, -31.02559, 64.63251);
    let aabb = AABB::point(&p);
    assert_eq!(aabb.bot.x, p.x);
    assert_eq!(aabb.bot.y, p.y);
    assert_eq!(aabb.bot.z, p.z);
    assert_eq!(aabb.top.x, p.x);
    assert_eq!(aabb.top.y, p.y);
    assert_eq!(aabb.top.z, p.z);
}

#[test]
fn aabb_expand_positive() {
    let zero = Point3::zero();
    let one = Point3::new(1.0, 1.0, 1.0);
    let aabb = AABB::new(&zero, &one).expand(50.50);
    assert_approx_eq!(aabb.bot.x, -50.5, 1e-5);
    assert_approx_eq!(aabb.bot.y, -50.5, 1e-5);
    assert_approx_eq!(aabb.bot.z, -50.5, 1e-5);
    assert_approx_eq!(aabb.top.x, 51.5, 1e-5);
    assert_approx_eq!(aabb.top.y, 51.5, 1e-5);
    assert_approx_eq!(aabb.top.z, 51.5, 1e-5);
}

#[test]
fn aabb_expand_negative() {
    let min = Point3::new(-50.50, -50.50, -50.50);
    let max = Point3::new(50.50, 50.50, 50.50);
    let aabb = AABB::new(&min, &max).expand(-0.50);
    assert_approx_eq!(aabb.bot.x, -50.0, 1e-5);
    assert_approx_eq!(aabb.bot.y, -50.0, 1e-5);
    assert_approx_eq!(aabb.bot.z, -50.0, 1e-5);
    assert_approx_eq!(aabb.top.x, 50.0, 1e-5);
    assert_approx_eq!(aabb.top.y, 50.0, 1e-5);
    assert_approx_eq!(aabb.top.z, 50.0, 1e-5);
}

#[test]
fn aabb_engulf_point_outside() {
    let aabb = AABB::point(&Point3::zero());
    let addme = Point3::new(10.0, -15.0, 20.0);
    let merged = aabb.engulf(&addme);
    assert_eq!(merged.bot.x, 0.0);
    assert_eq!(merged.bot.y, -15.0);
    assert_eq!(merged.bot.z, 0.0);
    assert_eq!(merged.top.x, 10.0);
    assert_eq!(merged.top.y, 0.0);
    assert_eq!(merged.top.z, 20.0);
}

#[test]
fn aabb_engulf_point_inside() {
    let aabb = AABB::new(
        &Point3::new(-5.0, -5.0, -5.0),
        &Point3::new(30.0, 30.0, 30.0),
    );
    let addme = Point3::new(2.0, -3.0, 15.0);
    let merged = aabb.engulf(&addme);
    assert_eq!(merged.bot.x, aabb.bot.x);
    assert_eq!(merged.bot.y, aabb.bot.y);
    assert_eq!(merged.bot.z, aabb.bot.z);
    assert_eq!(merged.top.x, aabb.top.x);
    assert_eq!(merged.top.y, aabb.top.y);
    assert_eq!(merged.top.z, aabb.top.z);
}

#[test]
fn aabb_engulf_point_infinite() {
    let aabb = AABB::point(&Point3::zero());
    let addme = Point3::new(INFINITY, -INFINITY, 0.0);
    let merged = aabb.engulf(&addme);
    assert_approx_eq!(merged.bot.x, 0.0, 1e-5);
    assert!(merged.bot.y.is_infinite());
    assert_approx_eq!(merged.bot.z, 0.0, 1e-5);
    assert!(merged.top.x.is_infinite());
    assert_approx_eq!(merged.top.y, 0.0, 1e-5);
    assert_approx_eq!(merged.top.z, 0.0, 1e-5);
}

#[test]
fn aabb_merge_aabb() {
    let aabb = AABB::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 1.0, 1.0));
    let outside = AABB::new(
        &Point3::new(59.28244, -3.01509, 47.61078),
        &Point3::new(67.30925, 53.29163, 82.07844),
    );
    let merged = aabb.merge(&outside);
    assert_eq!(merged.bot.x, aabb.bot.x);
    assert_eq!(merged.bot.y, outside.bot.y);
    assert_eq!(merged.bot.z, aabb.bot.z);
    assert_eq!(merged.top.x, outside.top.x);
    assert_eq!(merged.top.y, outside.top.y);
    assert_eq!(merged.top.z, outside.top.z);
}
#[test]
fn aabb_merge_aabb_inside() {
    let aabb = AABB::new(
        &Point3::new(-100.0, -100.0, -100.0),
        &Point3::new(100.0, 100.0, 100.0),
    );
    let point_inside = AABB::new(
        &Point3::new(-9.30374, 8.49896, -35.41399),
        &Point3::new(58.56126, 18.59649, 37.76507),
    );
    let merged = aabb.merge(&point_inside);
    assert_eq!(merged.bot.x, aabb.bot.x);
    assert_eq!(merged.bot.y, aabb.bot.y);
    assert_eq!(merged.bot.z, aabb.bot.z);
    assert_eq!(merged.top.x, aabb.top.x);
    assert_eq!(merged.top.y, aabb.top.y);
    assert_eq!(merged.top.z, aabb.top.z);
}

#[test]
fn aabb_merge_aabb_infinite() {
    let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    let target = AABB::new(
        &Point3::new(-INFINITY, -73.22298, 53.70019),
        &Point3::new(-138.73003, INFINITY, INFINITY),
    );
    let res = aabb.merge(&target);
    assert!(res.bot.x.is_infinite());
    assert_approx_eq!(res.bot.y, target.bot.y, 1e-5);
    assert_approx_eq!(res.bot.z, aabb.bot.z, 1e-5);
    assert_approx_eq!(res.top.x, aabb.top.x, 1e-5);
    assert!(res.top.y.is_infinite());
    assert!(res.top.z.is_infinite());
}

#[test]
fn aabb_contains_point_inside() {
    let aabb = AABB::new(
        &Point3::new(-10.0, -10.0, -10.0),
        &Point3::new(10.0, 10.0, 10.0),
    );
    let ins = Point3::new(-5.0, -5.0, -5.0);
    assert!(aabb.contains(&ins));
}

#[test]
fn aabb_contains_point_on_border() {
    let aabb = AABB::new(
        &Point3::new(-10.0, -10.0, -10.0),
        &Point3::new(10.0, 10.0, 10.0),
    );
    let border = Point3::new(10.0, 10.0, 10.0);
    assert!(aabb.contains(&border));
}

#[test]
fn aabb_contains_point_outside() {
    let aabb = AABB::new(
        &Point3::new(-10.0, -10.0, -10.0),
        &Point3::new(10.0, 10.0, 10.0),
    );
    let only_x = Point3::new(-10.000001, -5.0, -5.0);
    let only_y = Point3::new(-5.0, -10.000001, -5.0);
    let only_z = Point3::new(-5.0, -5.0, -10.000001);
    let all = Point3::new(11.0, 11.0, 11.0);
    assert!(!aabb.contains(&only_x));
    assert!(!aabb.contains(&only_y));
    assert!(!aabb.contains(&only_z));
    assert!(!aabb.contains(&all));
}
#[test]
fn aabb_contains_point_infinite() {
    let aabb = AABB::new(
        &Point3::new(-10.0, -10.0, -10.0),
        &Point3::new(10.0, 10.0, 10.0),
    );
    let inf = Point3::new(INFINITY, 0.0, 0.0);
    let minf = Point3::new(0.0, -INFINITY, 0.0);
    let inf2 = Point3::new(0.0, 0.0, INFINITY);
    assert!(!aabb.contains(&inf));
    assert!(!aabb.contains(&minf));
    assert!(!aabb.contains(&inf2));
}

#[test]
fn aabb_surface_zero() {
    //corner case, zero length
    let p = Point3::new(-0.53123, -24.29362, 84.26433);
    let aabb = AABB::point(&p);
    assert_eq!(aabb.surface(), 0.0);
}

#[test]
fn aabb_surface() {
    //normal length
    let aabb = AABB::new(&Point3::new(-1.0, -1.0, -1.0), &Point3::new(3.0, 4.0, 5.0));
    assert_eq!(aabb.surface(), 148.0);
}

#[test]
fn aabb_surface_infinite() {
    //infinite length
    let aabb = AABB::new(
        &Point3::new(-1.0, -1.0, -1.0),
        &Point3::new(1.0, 1.0, INFINITY),
    );
    assert_eq!(aabb.surface(), INFINITY);
    let aabb2 = AABB::new(
        &Point3::new(-INFINITY, -INFINITY, -INFINITY),
        &Point3::new(INFINITY, INFINITY, INFINITY),
    );
    assert_eq!(aabb2.surface(), INFINITY);
}

#[test]
fn aabb_volume_zero() {
    let p = Point3::new(-0.53123, -24.29362, 84.26433);
    let aabb = AABB::point(&p);
    assert_eq!(aabb.volume(), 0.0);
}

#[test]
fn aabb_volume() {
    let aabb = AABB::new(&Point3::new(-1.0, -1.0, -1.0), &Point3::new(3.0, 4.0, 5.0));
    assert_eq!(aabb.volume(), 120.0);
}

#[test]
fn aabb_volume_infinite() {
    let aabb = AABB::new(
        &Point3::new(-1.0, -1.0, -1.0),
        &Point3::new(1.0, 1.0, INFINITY),
    );
    assert_eq!(aabb.volume(), INFINITY);
    let aabb2 = AABB::new(
        &Point3::new(-INFINITY, -INFINITY, -INFINITY),
        &Point3::new(INFINITY, INFINITY, INFINITY),
    );
    assert_eq!(aabb2.volume(), INFINITY);
}

#[test]
fn aabb_longest_axis_zero() {
    //zero length -> return x
    let p = Point3::new(-0.53123, -24.29362, 84.26433);
    let aabb = AABB::point(&p);
    assert_eq!(aabb.longest_axis(), 0);
}

#[test]
fn aabb_longest_axis() {
    //longest x - non inf
    let aabbx = AABB::new(
        &Point3::new(-85.77731, 5.98468, -10.75332),
        &Point3::new(74.13619, 99.79995, 37.72758),
    );
    assert_eq!(aabbx.longest_axis(), 0);
    //longest y - non inf
    let aabby = AABB::new(
        &Point3::new(-27.68684, -73.58186, -69.54105),
        &Point3::new(65.46841, 95.43746, -51.04507),
    );
    assert_eq!(aabby.longest_axis(), 1);
    //longest z - non inf
    let aabbz = AABB::new(
        &Point3::new(17.90233, -6.71415, -88.93419),
        &Point3::new(76.75507, 50.73106, 95.81359),
    );
    assert_eq!(aabbz.longest_axis(), 2);
}

#[test]
fn aabb_longest_axis_infinite() {
    //longest x - inf
    let aabb = AABB::new(
        &Point3::new(-INFINITY, 5.98468, -10.75332),
        &Point3::new(74.13619, 99.79995, 37.72758),
    );
    assert_eq!(aabb.longest_axis(), 0);
    //longest y - inf
    let aabb = AABB::new(
        &Point3::new(-27.68684, -73.58186, -69.54105),
        &Point3::new(65.46841, INFINITY, -51.04507),
    );
    assert_eq!(aabb.longest_axis(), 1);
    //longest z - inf
    let aabb = AABB::new(
        &Point3::new(17.90233, -46.71415, -INFINITY),
        &Point3::new(76.75507, 90.73106, 95.81359),
    );
    assert_eq!(aabb.longest_axis(), 2);
    //everything infinite
    let aabb = AABB::new(
        &Point3::new(-INFINITY, -INFINITY, -INFINITY),
        &Point3::new(INFINITY, INFINITY, INFINITY),
    );
    assert_eq!(aabb.longest_axis(), 0);
}

#[test]
fn aabb_center_zero() {
    let p = Point3::new(-0.53123, -24.29362, 84.26433);
    let aabb = AABB::point(&p);
    let center = aabb.center();
    assert_approx_eq!(center.x, p.x);
    assert_approx_eq!(center.y, p.y);
    assert_approx_eq!(center.z, p.z);
}

#[test]
fn aabb_center() {
    //normal aabb
    let aabb = AABB::new(&Point3::new(-1.0, -1.0, -1.0), &Point3::new(1.0, 1.0, 1.0));
    let center = aabb.center();
    assert_approx_eq!(center.x, 0.0, 1e-5);
    assert_approx_eq!(center.y, 0.0, 1e-5);
    assert_approx_eq!(center.z, 0.0, 1e-5);
}

#[test]
fn aaabb_center_infinite() {
    //bot is infinite
    let aabb = AABB::new(
        &Point3::new(-1.0, -INFINITY, -1.0),
        &Point3::new(1.0, -1.0, 1.0),
    );
    let center = aabb.center();
    assert_approx_eq!(center.x, 0.0, 1e-5);
    assert!(center.y.is_infinite());
    assert_approx_eq!(center.z, 0.0, 1e-5);
    //top is infinite
    let aabb2 = AABB::new(
        &Point3::new(-1.0, -1.0, -1.0),
        &Point3::new(1.0, 1.0, INFINITY),
    );
    let center2 = aabb2.center();
    assert_approx_eq!(center2.x, 0.0, 1e-5);
    assert_approx_eq!(center2.y, 0.0, 1e-5);
    assert!(center2.z.is_infinite());
    //all infinite aabb
    let aabb3 = AABB::new(
        &Point3::new(-INFINITY, -INFINITY, -INFINITY),
        &Point3::new(INFINITY, INFINITY, INFINITY),
    );
    let center3 = aabb3.center();
    assert!((center3.x).is_nan());
    assert!((center3.y).is_nan());
    assert!((center3.z).is_nan());
}

#[test]
fn aabb_sum_point() {
    let aabb = AABB::point(&Point3::zero());
    let addme = Point3::new(10.0, -15.0, 20.0);
    let merged = aabb + addme;
    assert_eq!(merged.bot.x, 0.0);
    assert_eq!(merged.bot.y, -15.0);
    assert_eq!(merged.bot.z, 0.0);
    assert_eq!(merged.top.x, 10.0);
    assert_eq!(merged.top.y, 0.0);
    assert_eq!(merged.top.z, 20.0);
}

#[test]
fn aabb_sum_point_inside() {
    let aabb = AABB::new(
        &Point3::new(-5.0, -5.0, -5.0),
        &Point3::new(30.0, 30.0, 30.0),
    );
    let addme = Point3::new(2.0, -3.0, 15.0);
    let merged = aabb + addme;
    assert_eq!(merged.bot.x, aabb.bot.x);
    assert_eq!(merged.bot.y, aabb.bot.y);
    assert_eq!(merged.bot.z, aabb.bot.z);
    assert_eq!(merged.top.x, aabb.top.x);
    assert_eq!(merged.top.y, aabb.top.y);
    assert_eq!(merged.top.z, aabb.top.z);
}

#[test]
fn aabb_sum_point_infinite() {
    let aabb = AABB::point(&Point3::zero());
    let addme = Point3::new(INFINITY, -INFINITY, 0.0);
    let merged = aabb + addme;
    assert_approx_eq!(merged.bot.x, 0.0, 1e-5);
    assert!(merged.bot.y.is_infinite());
    assert_approx_eq!(merged.bot.z, 0.0, 1e-5);
    assert!(merged.top.x.is_infinite());
    assert_approx_eq!(merged.top.y, 0.0, 1e-5);
    assert_approx_eq!(merged.top.z, 0.0, 1e-5);
}

#[test]
fn aabb_sum_aabb() {
    let aabb = AABB::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 1.0, 1.0));
    let outside = AABB::new(
        &Point3::new(59.28244, -3.01509, 47.61078),
        &Point3::new(67.30925, 53.29163, 82.07844),
    );
    let merged = aabb + outside;
    assert_eq!(merged.bot.x, aabb.bot.x);
    assert_eq!(merged.bot.y, outside.bot.y);
    assert_eq!(merged.bot.z, aabb.bot.z);
    assert_eq!(merged.top.x, outside.top.x);
    assert_eq!(merged.top.y, outside.top.y);
    assert_eq!(merged.top.z, outside.top.z);
}
#[test]
fn aabb_sum_aabb_inside() {
    let aabb = AABB::new(
        &Point3::new(-100.0, -100.0, -100.0),
        &Point3::new(100.0, 100.0, 100.0),
    );
    let inside = AABB::new(
        &Point3::new(-9.30374, 8.49896, -35.41399),
        &Point3::new(58.56126, 18.59649, 37.76507),
    );
    let merged = aabb + inside;
    assert_eq!(merged.bot.x, aabb.bot.x);
    assert_eq!(merged.bot.y, aabb.bot.y);
    assert_eq!(merged.bot.z, aabb.bot.z);
    assert_eq!(merged.top.x, aabb.top.x);
    assert_eq!(merged.top.y, aabb.top.y);
    assert_eq!(merged.top.z, aabb.top.z);
}

#[test]
fn aabb_sum_aabb_infinite() {
    let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    let target = AABB::new(
        &Point3::new(-INFINITY, -73.22298, 53.70019),
        &Point3::new(-138.73003, INFINITY, INFINITY),
    );
    let res = aabb + target;
    assert!(res.bot.x.is_infinite());
    assert_approx_eq!(res.bot.y, target.bot.y, 1e-5);
    assert_approx_eq!(res.bot.z, aabb.bot.z, 1e-5);
    assert_approx_eq!(res.top.x, aabb.top.x, 1e-5);
    assert!(res.top.y.is_infinite());
    assert!(res.top.z.is_infinite());
}

#[test]
fn aabb_sum_assign_point() {
    let mut aabb = AABB::point(&Point3::zero());
    let addme = Point3::new(10.0, -15.0, 20.0);
    aabb += addme;
    assert_eq!(aabb.bot.x, 0.0);
    assert_eq!(aabb.bot.y, -15.0);
    assert_eq!(aabb.bot.z, 0.0);
    assert_eq!(aabb.top.x, 10.0);
    assert_eq!(aabb.top.y, 0.0);
    assert_eq!(aabb.top.z, 20.0);
}

#[test]
fn aabb_sum_assign_point_inside() {
    let mut aabb = AABB::new(
        &Point3::new(-5.0, -5.0, -5.0),
        &Point3::new(30.0, 30.0, 30.0),
    );
    let old = aabb;
    let addme = Point3::new(2.0, -3.0, 15.0);
    aabb += addme;
    assert_eq!(aabb.bot.x, old.bot.x);
    assert_eq!(aabb.bot.y, old.bot.y);
    assert_eq!(aabb.bot.z, old.bot.z);
    assert_eq!(aabb.top.x, old.top.x);
    assert_eq!(aabb.top.y, old.top.y);
    assert_eq!(aabb.top.z, old.top.z);
}

#[test]
fn aabb_sum_assign_point_infinite() {
    let mut aabb = AABB::point(&Point3::zero());
    let addme = Point3::new(INFINITY, -INFINITY, 0.0);
    aabb += addme;
    assert_approx_eq!(aabb.bot.x, 0.0, 1e-5);
    assert!(aabb.bot.y.is_infinite());
    assert_approx_eq!(aabb.bot.z, 0.0, 1e-5);
    assert!(aabb.top.x.is_infinite());
    assert_approx_eq!(aabb.top.y, 0.0, 1e-5);
    assert_approx_eq!(aabb.top.z, 0.0, 1e-5);
}

#[test]
fn aabb_sum_assign_aabb() {
    let mut aabb = AABB::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 1.0, 1.0));
    let old = aabb;
    let outside = AABB::new(
        &Point3::new(59.28244, -3.01509, 47.61078),
        &Point3::new(67.30925, 53.29163, 82.07844),
    );
    aabb += outside;
    assert_eq!(aabb.bot.x, old.bot.x);
    assert_eq!(aabb.bot.y, outside.bot.y);
    assert_eq!(aabb.bot.z, old.bot.z);
    assert_eq!(aabb.top.x, outside.top.x);
    assert_eq!(aabb.top.y, outside.top.y);
    assert_eq!(aabb.top.z, outside.top.z);
}
#[test]
fn aabb_sum_assign_aabb_inside() {
    let mut aabb = AABB::new(
        &Point3::new(-100.0, -100.0, -100.0),
        &Point3::new(100.0, 100.0, 100.0),
    );
    let old = aabb;
    let inside = AABB::new(
        &Point3::new(-9.30374, 8.49896, -35.41399),
        &Point3::new(58.56126, 18.59649, 37.76507),
    );
    aabb += inside;
    assert_eq!(aabb.bot.x, old.bot.x);
    assert_eq!(aabb.bot.y, old.bot.y);
    assert_eq!(aabb.bot.z, old.bot.z);
    assert_eq!(aabb.top.x, old.top.x);
    assert_eq!(aabb.top.y, old.top.y);
    assert_eq!(aabb.top.z, old.top.z);
}

#[test]
fn aabb_sum_assign_aabb_infinite() {
    let mut aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    let old = aabb;
    let target = AABB::new(
        &Point3::new(-INFINITY, -73.22298, 53.70019),
        &Point3::new(-138.73003, INFINITY, INFINITY),
    );
    aabb += target;
    assert!(aabb.bot.x.is_infinite());
    assert_approx_eq!(aabb.bot.y, target.bot.y, 1e-5);
    assert_approx_eq!(aabb.bot.z, old.bot.z, 1e-5);
    assert_approx_eq!(aabb.top.x, old.top.x, 1e-5);
    assert!(aabb.top.y.is_infinite());
    assert!(aabb.top.z.is_infinite());
}

#[test]
fn aabb_transform() {
    let one = Point3::new(1.0, 1.0, 1.0);
    let aabb = AABB::new(&-one, &one);
    let matrix = Matrix4::rotate_z((-45.0 as f32).to_radians());
    let transformed = aabb.transform(&matrix);

    assert_approx_eq!(transformed.bot.x, -1.414213);
    assert_approx_eq!(transformed.bot.y, -1.414213);
    assert_eq!(transformed.bot.z, -1.0);
    assert_approx_eq!(transformed.top.x, 1.414213);
    assert_approx_eq!(transformed.top.y, 1.414213);
    assert_eq!(transformed.top.z, 1.0);
}
