use crate::utility::efloat::NextRepresentable;
use crate::utility::{lerp, Ef32};
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::f32::INFINITY;

const RANDOM_ITERATIONS: usize = 10000;
const FIXED_SEED: u64 = 0xD18BEC491D021ED4;

#[test]
fn ef32_next_after_neg_zero() {
    assert!((-0.0).next_after() > 0.0);
}

#[test]
fn ef32_previous_before_neg_zero() {
    assert!((-0.0).previous_before() < 0.0);
}

#[test]
fn ef32_next_float_up_inf() {
    assert_eq!((INFINITY).next_after(), INFINITY);
    assert!((-INFINITY).next_after() > -INFINITY);
}

#[test]
fn ef32_next_float_down_inf() {
    assert!((INFINITY).previous_before() < INFINITY);
    assert_eq!((-INFINITY).previous_before(), -INFINITY);
}

fn gen_f32<T: RngCore>(rng: &mut T, min_exp: f32, max_exp: f32) -> f32 {
    let log = lerp(rng.gen_range(0.0, 1.0), min_exp, max_exp);
    let sign = if rng.gen_range(0.0, 1.0) < 0.5 {
        -1.0 as f32
    } else {
        1.0 as f32
    };
    sign * (10.0 as f32).powf(log)
}

#[test]
fn ef32_precision_longrun() {
    // Shamelessly copied from pbrt-v3. It's their float model so I'm gonna port it with their tests
    // (Even though it's not rocket science, just random operations and check that precision holds)
    let mut rng = Xoshiro256StarStar::seed_from_u64(FIXED_SEED);
    for _ in 0..RANDOM_ITERATIONS {
        //gen random float
        let float1 = gen_f32(&mut rng, -4.0, 4.0);
        let mut precise1 = float1 as f64;
        let mut tracked1 = Ef32::new(float1, 0.0);
        for _ in 0..100 {
            if !precise1.is_finite() || !tracked1.float().is_finite() {
                break;
            }
            // actual assert that the previous operation was fine
            assert!(precise1 >= tracked1.lower() as f64);
            assert!(precise1 <= tracked1.upper() as f64);
            //gen second operand
            let float2;
            let precise2;
            let tracked2;
            match rng.gen_range(0, 3) {
                0 => {
                    //random
                    float2 = gen_f32(&mut rng, -4.0, 4.0);
                    tracked2 = Ef32::new(float2, 0.0);
                    precise2 = float2 as f64;
                }
                1 => {
                    //same
                    precise2 = precise1;
                    tracked2 = tracked1;
                }
                2 => {
                    // random with small error
                    float2 = gen_f32(&mut rng, -4.0, 4.0);
                    let err = (gen_f32(&mut rng, -8.0, -2.0) * float2).abs();
                    tracked2 = Ef32::new(float2, err);
                    let offset = rng.gen_range(0.0, 1.0);
                    let precise_tmp =
                        ((1.0 - offset) * tracked2.lower() + offset * tracked2.upper()) as f64;
                    precise2 = if precise_tmp < tracked2.upper() as f64
                        && precise_tmp > tracked2.lower() as f64
                    {
                        precise_tmp
                    } else {
                        tracked2.float() as f64
                    };
                }
                _ => panic!(),
            }
            // random operation
            match rng.gen_range(0, 1) {
                0 => {
                    tracked1 = -tracked1;
                    precise1 = -precise1;
                }
                1 => {
                    tracked1 = tracked1 + tracked2;
                    precise1 = precise1 + precise2;
                }
                2 => {
                    tracked1 = tracked1 - tracked2;
                    precise1 = precise1 - precise2;
                }
                3 => {
                    tracked1 = tracked1 * tracked2;
                    precise1 = precise1 * precise2;
                }
                4 => {
                    if tracked1.lower() * tracked1.upper() > 0.0
                        && tracked2.lower() * tracked1.upper() > 0.0
                    {
                        tracked1 = tracked1 / tracked2;
                        precise1 = precise1 / precise2;
                    }
                }
                5 => {
                    if precise1 >= 0.0 && tracked1.lower() > 0.0 {
                        tracked1 = tracked1.sqrt();
                        precise1 = precise1.sqrt();
                    }
                }
                6 => {
                    tracked1 = tracked1.abs();
                    precise1 = precise1.abs()
                }
                _ => panic!(),
            }
        }
    }
}
