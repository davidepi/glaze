use crate::geometry::{Matrix4, Point3, Vec3};
use assert_approx_eq::assert_approx_eq;

#[test]
fn matrix4_new() {
    let vals = [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    let m = Matrix4::new(&vals);
    assert_eq!(m.m[00], vals[0]);
    assert_eq!(m.m[01], vals[1]);
    assert_eq!(m.m[02], vals[2]);
    assert_eq!(m.m[03], vals[3]);
    assert_eq!(m.m[04], vals[4]);
    assert_eq!(m.m[05], vals[5]);
    assert_eq!(m.m[06], vals[6]);
    assert_eq!(m.m[07], vals[7]);
    assert_eq!(m.m[08], vals[8]);
    assert_eq!(m.m[09], vals[9]);
    assert_eq!(m.m[10], vals[10]);
    assert_eq!(m.m[11], vals[11]);
    assert_eq!(m.m[12], vals[12]);
    assert_eq!(m.m[13], vals[13]);
    assert_eq!(m.m[14], vals[14]);
    assert_eq!(m.m[15], vals[15]);
}

#[test]
fn matrix4_set_zero() {
    let m = Matrix4::zero();
    assert_eq!(m.m[00], 0.0);
    assert_eq!(m.m[01], 0.0);
    assert_eq!(m.m[02], 0.0);
    assert_eq!(m.m[03], 0.0);
    assert_eq!(m.m[04], 0.0);
    assert_eq!(m.m[05], 0.0);
    assert_eq!(m.m[06], 0.0);
    assert_eq!(m.m[07], 0.0);
    assert_eq!(m.m[08], 0.0);
    assert_eq!(m.m[09], 0.0);
    assert_eq!(m.m[10], 0.0);
    assert_eq!(m.m[11], 0.0);
    assert_eq!(m.m[12], 0.0);
    assert_eq!(m.m[13], 0.0);
    assert_eq!(m.m[14], 0.0);
    assert_eq!(m.m[15], 0.0);
}

#[test]
fn matrix4_set_identity() {
    let m = Matrix4::identity();
    assert_eq!(m.m[00], 1.0);
    assert_eq!(m.m[01], 0.0);
    assert_eq!(m.m[02], 0.0);
    assert_eq!(m.m[03], 0.0);
    assert_eq!(m.m[04], 0.0);
    assert_eq!(m.m[05], 1.0);
    assert_eq!(m.m[06], 0.0);
    assert_eq!(m.m[07], 0.0);
    assert_eq!(m.m[08], 0.0);
    assert_eq!(m.m[09], 0.0);
    assert_eq!(m.m[10], 1.0);
    assert_eq!(m.m[11], 0.0);
    assert_eq!(m.m[12], 0.0);
    assert_eq!(m.m[13], 0.0);
    assert_eq!(m.m[14], 0.0);
    assert_eq!(m.m[15], 1.0);
}

#[test]
fn matrix4_set_translation() {
    let dir = Vec3::new(-87.39175, 8.35182, -93.43325);
    let m = Matrix4::translation(&dir);
    assert_eq!(m.m[00], 1.0);
    assert_eq!(m.m[01], 0.0);
    assert_eq!(m.m[02], 0.0);
    assert_eq!(m.m[03], dir.x);
    assert_eq!(m.m[04], 0.0);
    assert_eq!(m.m[05], 1.0);
    assert_eq!(m.m[06], 0.0);
    assert_eq!(m.m[07], dir.y);
    assert_eq!(m.m[08], 0.0);
    assert_eq!(m.m[09], 0.0);
    assert_eq!(m.m[10], 1.0);
    assert_eq!(m.m[11], dir.z);
    assert_eq!(m.m[12], 0.0);
    assert_eq!(m.m[13], 0.0);
    assert_eq!(m.m[14], 0.0);
    assert_eq!(m.m[15], 1.0);
}

#[test]
fn matrix4_set_scale_non_uniform() {
    let magnitude = Vec3::new(41.24096, -93.12313, 31.83295);
    let m = Matrix4::scale(&magnitude);
    assert_eq!(m.m[00], magnitude.x);
    assert_eq!(m.m[01], 0.0);
    assert_eq!(m.m[02], 0.0);
    assert_eq!(m.m[03], 0.0);
    assert_eq!(m.m[04], 0.0);
    assert_eq!(m.m[05], magnitude.y);
    assert_eq!(m.m[06], 0.0);
    assert_eq!(m.m[07], 0.0);
    assert_eq!(m.m[08], 0.0);
    assert_eq!(m.m[09], 0.0);
    assert_eq!(m.m[10], magnitude.z);
    assert_eq!(m.m[11], 0.0);
    assert_eq!(m.m[12], 0.0);
    assert_eq!(m.m[13], 0.0);
    assert_eq!(m.m[14], 0.0);
    assert_eq!(m.m[15], 1.0);
}

#[test]
fn matrix4_set_rotate_x() {
    let roll = 3.0 / 4.0 * std::f32::consts::PI;
    let m = Matrix4::rotate_x(roll);
    assert_approx_eq!(m.m[00], 1.0);
    assert_approx_eq!(m.m[01], 0.0);
    assert_approx_eq!(m.m[02], 0.0);
    assert_approx_eq!(m.m[03], 0.0);
    assert_approx_eq!(m.m[04], 0.0);
    assert_approx_eq!(m.m[05], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[06], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[07], 0.0);
    assert_approx_eq!(m.m[08], 0.0);
    assert_approx_eq!(m.m[09], 1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[10], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[11], 0.0);
    assert_approx_eq!(m.m[12], 0.0);
    assert_approx_eq!(m.m[13], 0.0);
    assert_approx_eq!(m.m[14], 0.0);
    assert_approx_eq!(m.m[15], 1.0);
}

#[test]
fn matrix4_set_rotate_y() {
    let pitch = 3.0 / 4.0 * std::f32::consts::PI;
    let m = Matrix4::rotate_y(pitch);
    assert_approx_eq!(m.m[00], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[01], 0.0);
    assert_approx_eq!(m.m[02], 1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[03], 0.0);
    assert_approx_eq!(m.m[04], 0.0);
    assert_approx_eq!(m.m[05], 1.0);
    assert_approx_eq!(m.m[06], 0.0);
    assert_approx_eq!(m.m[07], 0.0);
    assert_approx_eq!(m.m[08], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[09], 0.0);
    assert_approx_eq!(m.m[10], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[11], 0.0);
    assert_approx_eq!(m.m[12], 0.0);
    assert_approx_eq!(m.m[13], 0.0);
    assert_approx_eq!(m.m[14], 0.0);
    assert_approx_eq!(m.m[15], 1.0);
}

#[test]
fn matrix4_set_rotate_z() {
    let yaw = 3.0 / 4.0 * std::f32::consts::PI;
    let m = Matrix4::rotate_z(yaw);
    assert_approx_eq!(m.m[00], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[01], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[02], 0.0);
    assert_approx_eq!(m.m[03], 0.0);
    assert_approx_eq!(m.m[04], 1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[05], -1.0 / (2.0_f32).sqrt(), 1e-5);
    assert_approx_eq!(m.m[06], 0.0);
    assert_approx_eq!(m.m[07], 0.0);
    assert_approx_eq!(m.m[08], 0.0);
    assert_approx_eq!(m.m[09], 0.0);
    assert_approx_eq!(m.m[10], 1.0);
    assert_approx_eq!(m.m[11], 0.0);
    assert_approx_eq!(m.m[12], 0.0);
    assert_approx_eq!(m.m[13], 0.0);
    assert_approx_eq!(m.m[14], 0.0);
    assert_approx_eq!(m.m[15], 1.0);
}

#[test]
#[should_panic]
fn matrix4_camera_to_world_not_normalized() {
    let up = Vec3::new(1.0, 1.0, 0.0);
    let pos = Point3::new(1.0, 0.0, -2.0);
    let target = Point3::new(0.0, 0.0, 1.0);
    let _m = Matrix4::camera_to_world(&pos, &target, &up);
}

#[test]
fn matrix4_camera_to_world() {
    let up = Vec3::new(1.0, 1.0, 0.0).normalize();
    let pos = Point3::new(1.0, 0.0, -2.0);
    let target = Point3::new(0.0, 0.0, 1.0);
    let m = Matrix4::camera_to_world(&pos, &target, &up);

    assert_approx_eq!(m.m[00], 0.688247144, 1e-5);
    assert_approx_eq!(m.m[01], 0.65292853, 1e-5);
    assert_approx_eq!(m.m[02], -0.31622776, 1e-5);
    assert_approx_eq!(m.m[03], 1.0, 1e-5);
    assert_approx_eq!(m.m[04], -0.688247, 1e-5);
    assert_approx_eq!(m.m[05], 0.725476, 1e-5);
    assert_approx_eq!(m.m[06], 0.0, 1e-5);
    assert_approx_eq!(m.m[07], 0.0, 1e-5);
    assert_approx_eq!(m.m[08], 0.22941573, 1e-5);
    assert_approx_eq!(m.m[09], 0.21764286, 1e-5);
    assert_approx_eq!(m.m[10], 0.948683261, 1e-5);
    assert_approx_eq!(m.m[11], -2.0, 1e-5);
    assert_approx_eq!(m.m[12], 0.0, 1e-5);
    assert_approx_eq!(m.m[13], 0.0, 1e-5);
    assert_approx_eq!(m.m[14], 0.0, 1e-5);
    assert_approx_eq!(m.m[15], 1.0, 1e-5);
}

#[test]
fn matrix4_transpose() {
    let vals = [
        27.9484, -88.37513, -25.05486, 0.93192, 19.53558, 55.46225, -92.99693, 13.30983, -39.91206,
        -63.35516, -80.28301, 96.89149, -97.99183, 69.73036, 34.27019, 58.81281,
    ];
    let m = Matrix4::new(&vals);
    let out = m.transpose();
    assert_eq!(vals[00], out.m[00]);
    assert_eq!(vals[01], out.m[04]);
    assert_eq!(vals[02], out.m[08]);
    assert_eq!(vals[03], out.m[12]);
    assert_eq!(vals[04], out.m[01]);
    assert_eq!(vals[05], out.m[05]);
    assert_eq!(vals[06], out.m[09]);
    assert_eq!(vals[07], out.m[13]);
    assert_eq!(vals[08], out.m[02]);
    assert_eq!(vals[09], out.m[06]);
    assert_eq!(vals[10], out.m[10]);
    assert_eq!(vals[11], out.m[14]);
    assert_eq!(vals[12], out.m[03]);
    assert_eq!(vals[13], out.m[07]);
    assert_eq!(vals[14], out.m[11]);
    assert_eq!(vals[15], out.m[15]);
}

#[test]
fn matrix4_transform() {
    //    Point3 p(1.0,0.0,0.0);
    //    let translate = Vec3::new(0.0, 0.0, 0.0);
    //    let rotate = Vec3::new(0.0, radians(90.0), 0.0);
    //    let scale = Vec3::new(1.0);
    //
    //    Matrix4 transform;
    //    transform.set_transform(translate,rotate,scale);
    //
    //    Point3 res = transform*p;
    //    assert_approx_eq!(res.x, 0.0, 1e-5);
    //    assert_approx_eq!(res.y, 3.5, 1e-5);
    //    assert_approx_eq!(res.z, 0.0, 1e-5);
}

#[test]
fn matrix4_inverse_invertible() {
    //invertible
    let vals = [
        44.48, -69.73, 62.26, -89.47, -20.59, 45.01, -77.12, 21.26, 3.27, 42.29, -62.23, -49.23,
        6.83, -80.83, 18.96, -84.16,
    ];
    let m = Matrix4::new(&vals).inverse().unwrap();
    assert_approx_eq!(m.m[00], 0.056968, 1e-5);
    assert_approx_eq!(m.m[01], 0.060421, 1e-5);
    assert_approx_eq!(m.m[02], -0.026891, 1e-5);
    assert_approx_eq!(m.m[03], -0.029569, 1e-5);
    assert_approx_eq!(m.m[04], -0.016213, 1e-5);
    assert_approx_eq!(m.m[05], -0.032876, 1e-5);
    assert_approx_eq!(m.m[06], 0.023122, 1e-5);
    assert_approx_eq!(m.m[07], -0.004594, 1e-5);
    assert_approx_eq!(m.m[08], -0.020370, 1e-5);
    assert_approx_eq!(m.m[09], -0.040761, 1e-5);
    assert_approx_eq!(m.m[10], 0.014874, 1e-5);
    assert_approx_eq!(m.m[11], 0.002657, 1e-5);
    assert_approx_eq!(m.m[12], 0.015605, 1e-5);
    assert_approx_eq!(m.m[13], 0.027296, 1e-5);
    assert_approx_eq!(m.m[14], -0.021038, 1e-5);
    assert_approx_eq!(m.m[15], -0.009270, 1e-5);
}

#[test]
fn matrix4_inverse_non_invertible() {
    let m = Matrix4::zero().inverse();
    assert!(m.is_none());
}

#[test]
fn matrix4_get_translation() {
    let scale = Vec3::new(53.94708, -56.04181, 38.21224);
    let translation = Vec3::new(-5.28423, 22.63478, 22.10424);
    let rotation = Vec3::new(-0.07627, -31.31443, -88.95238);
    let ms = Matrix4::scale(&scale);
    let mt = Matrix4::translation(&translation);
    let mrx = Matrix4::rotate_x(rotation.x);
    let mry = Matrix4::rotate_y(rotation.y);
    let mrz = Matrix4::rotate_z(rotation.z);
    let mut combined = Matrix4::identity();

    combined *= mt;
    combined *= mrz;
    combined *= mry;
    combined *= mrx;
    combined *= ms;

    let extracted = combined.get_translation();
    assert_approx_eq!(extracted.x, translation.x, 1e-5);
    assert_approx_eq!(extracted.y, translation.y, 1e-5);
    assert_approx_eq!(extracted.z, translation.z, 1e-5);
}

#[test]
fn matrix4_get_scale() {
    let scale = Vec3::new(37.99025, 69.85438, 5.79172);
    let translation = Vec3::new(-17.90241, 37.90712, 74.85354);
    let rotation = Vec3::new(-20.80442, 27.33369, -31.58807);
    let ms = Matrix4::scale(&scale);
    let mt = Matrix4::translation(&translation);
    let mrx = Matrix4::rotate_x(rotation.x);
    let mry = Matrix4::rotate_y(rotation.y);
    let mrz = Matrix4::rotate_z(rotation.z);
    let mut combined = Matrix4::identity();

    combined *= mt;
    combined *= mrz;
    combined *= mry;
    combined *= mrx;
    combined *= ms;

    let extracted = combined.get_scale();
    assert_approx_eq!(extracted.x, scale.x, 1e-5);
    assert_approx_eq!(extracted.y, scale.y, 1e-5);
    assert_approx_eq!(extracted.z, scale.z, 1e-5);
}

#[test]
fn matrix4_add() {
    let val1 = [
        -98.96, 98.99, 72.96, 98.37, -61.17, 6.0, -13.05, 18.62, 43.24, -19.56, 39.17, -19.17,
        -49.98, -36.64, 48.0, 45.27,
    ];
    let val2 = [
        59.09, -8.73, -19.45, 88.6, 85.6, -67.18, 31.89, -71.7, 40.15, 38.28, 48.01, -73.72, 37.04,
        34.6, -46.98, -44.3,
    ];
    let m1 = Matrix4::new(&val1);
    let m2 = Matrix4::new(&val2);
    let out = m1 + m2;
    assert_approx_eq!(out.m[00], -39.87, 1e-5);
    assert_approx_eq!(out.m[01], 90.26, 1e-5);
    assert_approx_eq!(out.m[02], 53.51, 1e-5);
    assert_approx_eq!(out.m[03], 186.97, 1e-5);
    assert_approx_eq!(out.m[04], 24.43, 1e-5);
    assert_approx_eq!(out.m[05], -61.18, 1e-5);
    assert_approx_eq!(out.m[06], 18.84, 1e-5);
    assert_approx_eq!(out.m[07], -53.08, 1e-5);
    assert_approx_eq!(out.m[08], 83.39, 1e-5);
    assert_approx_eq!(out.m[09], 18.72, 1e-5);
    assert_approx_eq!(out.m[10], 87.18, 1e-5);
    assert_approx_eq!(out.m[11], -92.89, 1e-5);
    assert_approx_eq!(out.m[12], -12.94, 1e-5);
    assert_approx_eq!(out.m[13], -2.04, 1e-5);
    assert_approx_eq!(out.m[14], 1.02, 1e-5);
    assert_approx_eq!(out.m[15], 0.970001, 1e-5);
}

#[test]
fn matrix4_add_assign() {
    let val1 = [
        -98.96, 98.99, 72.96, 98.37, -61.17, 6.0, -13.05, 18.62, 43.24, -19.56, 39.17, -19.17,
        -49.98, -36.64, 48.0, 45.27,
    ];
    let val2 = [
        59.09, -8.73, -19.45, 88.6, 85.6, -67.18, 31.89, -71.7, 40.15, 38.28, 48.01, -73.72, 37.04,
        34.6, -46.98, -44.3,
    ];
    let mut m1 = Matrix4::new(&val1);
    let m2 = Matrix4::new(&val2);
    m1 += m2;
    assert_approx_eq!(m1.m[00], -39.87, 1e-5);
    assert_approx_eq!(m1.m[01], 90.26, 1e-5);
    assert_approx_eq!(m1.m[02], 53.51, 1e-5);
    assert_approx_eq!(m1.m[03], 186.97, 1e-5);
    assert_approx_eq!(m1.m[04], 24.43, 1e-5);
    assert_approx_eq!(m1.m[05], -61.18, 1e-5);
    assert_approx_eq!(m1.m[06], 18.84, 1e-5);
    assert_approx_eq!(m1.m[07], -53.08, 1e-5);
    assert_approx_eq!(m1.m[08], 83.39, 1e-5);
    assert_approx_eq!(m1.m[09], 18.72, 1e-5);
    assert_approx_eq!(m1.m[10], 87.18, 1e-5);
    assert_approx_eq!(m1.m[11], -92.89, 1e-5);
    assert_approx_eq!(m1.m[12], -12.94, 1e-5);
    assert_approx_eq!(m1.m[13], -2.04, 1e-5);
    assert_approx_eq!(m1.m[14], 1.02, 1e-5);
    assert_approx_eq!(m1.m[15], 0.970001, 1e-5);
}

#[test]
fn matrix4_sub() {
    let val1 = [
        -20.64, -25.69, -17.56, -97.15, 88.28, 28.07, 65.32, 46.34, 1.19, -66.15, 92.73, -3.68,
        -20.59, 62.21, 73.89, -29.04,
    ];
    let val2 = [
        61.53, 75.87, 44.25, -57.29, -81.46, 97.1, -62.07, -27.39, 34.94, 61.01, 5.92, -71.72,
        90.79, 93.32, -63.03, 45.79,
    ];
    let m1 = Matrix4::new(&val1);
    let m2 = Matrix4::new(&val2);
    let out = m1 - m2;
    assert_approx_eq!(out.m[00], -82.17, 1e-5);
    assert_approx_eq!(out.m[01], -101.56, 1e-5);
    assert_approx_eq!(out.m[02], -61.81, 1e-5);
    assert_approx_eq!(out.m[03], -39.86, 1e-5);
    assert_approx_eq!(out.m[04], 169.73999, 1e-5); //fuck floats
    assert_approx_eq!(out.m[05], -69.03, 1e-5);
    assert_approx_eq!(out.m[06], 127.39, 1e-5);
    assert_approx_eq!(out.m[07], 73.73, 1e-5);
    assert_approx_eq!(out.m[08], -33.75, 1e-5);
    assert_approx_eq!(out.m[09], -127.16, 1e-5);
    assert_approx_eq!(out.m[10], 86.81, 1e-5);
    assert_approx_eq!(out.m[11], 68.04, 1e-5);
    assert_approx_eq!(out.m[12], -111.38, 1e-5);
    assert_approx_eq!(out.m[13], -31.11, 1e-5);
    assert_approx_eq!(out.m[14], 136.92, 1e-5);
    assert_approx_eq!(out.m[15], -74.83, 1e-5);
}

#[test]
fn matrix4_sub_assign() {
    let val1 = [
        -20.64, -25.69, -17.56, -97.15, 88.28, 28.07, 65.32, 46.34, 1.19, -66.15, 92.73, -3.68,
        -20.59, 62.21, 73.89, -29.04,
    ];
    let val2 = [
        61.53, 75.87, 44.25, -57.29, -81.46, 97.1, -62.07, -27.39, 34.94, 61.01, 5.92, -71.72,
        90.79, 93.32, -63.03, 45.79,
    ];
    let mut m1 = Matrix4::new(&val1);
    let m2 = Matrix4::new(&val2);
    m1 -= m2;
    assert_approx_eq!(m1.m[00], -82.17, 1e-5);
    assert_approx_eq!(m1.m[01], -101.56, 1e-5);
    assert_approx_eq!(m1.m[02], -61.81, 1e-5);
    assert_approx_eq!(m1.m[03], -39.86, 1e-5);
    assert_approx_eq!(m1.m[04], 169.73999, 1e-5);
    assert_approx_eq!(m1.m[05], -69.03, 1e-5);
    assert_approx_eq!(m1.m[06], 127.39, 1e-5);
    assert_approx_eq!(m1.m[07], 73.73, 1e-5);
    assert_approx_eq!(m1.m[08], -33.75, 1e-5);
    assert_approx_eq!(m1.m[09], -127.16, 1e-5);
    assert_approx_eq!(m1.m[10], 86.81, 1e-5);
    assert_approx_eq!(m1.m[11], 68.04, 1e-5);
    assert_approx_eq!(m1.m[12], -111.38, 1e-5);
    assert_approx_eq!(m1.m[13], -31.11, 1e-5);
    assert_approx_eq!(m1.m[14], 136.92, 1e-5);
    assert_approx_eq!(m1.m[15], -74.83, 1e-5);
}

#[test]
fn matrix4_mul() {
    let val1 = [
        46.3, 6.5, -84.39, 6.06, 91.72, 78.04, -64.94, 32.07, -59.33, -78.26, 54.08, -73.42,
        -27.12, 4.49, 69.9, 91.19,
    ];
    let val2 = [
        -52.91, 12.02, -58.9, 29.93, -16.45, 78.46, 19.49, 27.82, 63.5, 74.75, 51.43, 8.44, 15.54,
        25.59, 73.89, -9.28,
    ];
    let m1 = Matrix4::new(&val1);
    let m2 = Matrix4::new(&val2);
    let out = m1 * m2;
    assert_approx_eq!(out.m[00], -7821.25048, 1e-5);
    assert_approx_eq!(out.m[01], -5086.56104, 1e-5);
    assert_approx_eq!(out.m[02], -6492.78906, 1e-5);
    assert_approx_eq!(out.m[03], 798.100586, 1e-5);
    assert_approx_eq!(out.m[04], -9761.98535, 1e-5);
    assert_approx_eq!(out.m[05], 3191.89941, 1e-5);
    assert_approx_eq!(out.m[06], -4851.52051, 1e-5);
    assert_approx_eq!(out.m[07], 4070.54907, 1e-5);
    assert_approx_eq!(out.m[08], 6719.66064, 1e-5);
    assert_approx_eq!(out.m[09], -4689.76367, 1e-5);
    assert_approx_eq!(out.m[10], -674.419434, 1e-5);
    assert_approx_eq!(out.m[11], -2815.16724, 1e-5);
    assert_approx_eq!(out.m[12], 7216.80126, 1e-5);
    assert_approx_eq!(out.m[13], 7584.87988, 1e-5);
    assert_approx_eq!(out.m[14], 12017.8643, 1e-5);
    assert_approx_eq!(out.m[15], -943.077087, 1e-5);
}

#[test]
fn matrix4_mul_assign() {
    let val1 = [
        46.3, 6.5, -84.39, 6.06, 91.72, 78.04, -64.94, 32.07, -59.33, -78.26, 54.08, -73.42,
        -27.12, 4.49, 69.9, 91.19,
    ];
    let val2 = [
        -52.91, 12.02, -58.9, 29.93, -16.45, 78.46, 19.49, 27.82, 63.5, 74.75, 51.43, 8.44, 15.54,
        25.59, 73.89, -9.28,
    ];
    let mut m1 = Matrix4::new(&val1);
    let m2 = Matrix4::new(&val2);
    m1 *= m2;
    assert_approx_eq!(m1.m[00], -7821.25048, 1e-5);
    assert_approx_eq!(m1.m[01], -5086.56104, 1e-5);
    assert_approx_eq!(m1.m[02], -6492.78906, 1e-5);
    assert_approx_eq!(m1.m[03], 798.100586, 1e-5);
    assert_approx_eq!(m1.m[04], -9761.98535, 1e-5);
    assert_approx_eq!(m1.m[05], 3191.89941, 1e-5);
    assert_approx_eq!(m1.m[06], -4851.52051, 1e-5);
    assert_approx_eq!(m1.m[07], 4070.54907, 1e-5);
    assert_approx_eq!(m1.m[08], 6719.66064, 1e-5);
    assert_approx_eq!(m1.m[09], -4689.76367, 1e-5);
    assert_approx_eq!(m1.m[10], -674.419434, 1e-5);
    assert_approx_eq!(m1.m[11], -2815.16724, 1e-5);
    assert_approx_eq!(m1.m[12], 7216.80126, 1e-5);
    assert_approx_eq!(m1.m[13], 7584.87988, 1e-5);
    assert_approx_eq!(m1.m[14], 12017.8643, 1e-5);
    assert_approx_eq!(m1.m[15], -943.077087, 1e-5);
}
