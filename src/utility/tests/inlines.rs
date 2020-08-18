use crate::utility::inlines::clamp;
use crate::utility::{float_eq, lerp};

#[test]
fn inlines_float_eq_exact() {
    let a = 0.0;
    let b = 0.0;
    assert!(float_eq(a, b, 1E-5));
}

#[test]
fn inlines_float_eq_inexact() {
    let a = 0.3;
    let errored = a + f32::EPSILON;
    assert_ne!(a, errored);
    assert!(float_eq(a, errored, 1E-5));
    assert!(!float_eq(a, errored, 1E-16));
}

#[test]
fn inlines_clamp_in_range() {
    let a = 0.75;
    let res = clamp(a, 0.5, 2.);
    assert_eq!(res, a);
}

#[test]
fn inlines_clamp_lower() {
    let a = -0.5;
    let res = clamp(a, 0.5, 2.0);
    assert_eq!(res, 0.5);
}

#[test]
fn inlines_clamp_higher() {
    let a = 3.50;
    let res = clamp(a, 0.50, 2.0);
    assert_eq!(res, 2.0);
}

#[test]
fn inlines_lerp() {
    let a = 0.250;
    let res = lerp(a, 0.0, 1.0);
    assert_eq!(res, a);
    let res = lerp(a, -1.0, 2.0);
    assert_eq!(res, -0.25);
}
