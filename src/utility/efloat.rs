use overload::overload;
use std::fmt::Formatter;
use std::ops;

pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

/// Trait used to implement methods to get the next or the previous representable float.
pub trait NextRepresentable {
    /// Returns the next representable float.
    fn next_after(self) -> Self;

    /// Returns the previous representable float.
    fn previous_before(self) -> Self;
}

impl NextRepresentable for f32 {
    fn next_after(self) -> Self {
        if self.is_infinite() && self > 0.0 {
            self
        } else {
            let v = if self == -0.0 { 0.0 } else { self };
            let mut ui = v.to_bits();
            ui = if v >= 0.0 { ui + 1 } else { ui - 1 };
            f32::from_bits(ui)
        }
    }

    fn previous_before(self) -> Self {
        if self.is_infinite() && self < 0.0 {
            self
        } else {
            let v = if self == 0.0 { -0.0 } else { self };
            let mut ui = v.to_bits();
            ui = if v > 0.0 { ui - 1 } else { ui + 1 };
            f32::from_bits(ui)
        }
    }
}

/// Function to track the accumulated fp error as described by *Higham N. J.* in
/// *Accuracy and Stability of Numerical Algorithms*, section 3.1.
/// For a precise description of the function or the input value check the above source (it's long).
/// Intuitively this function is the accumulated rounding error, the input is the nth operation
/// performed.
#[inline]
pub(crate) fn gamma(n: i32) -> f32 {
    let n_machine = n as f32 * MACHINE_EPSILON;
    n_machine / 1.0 - n_machine
}

/// Float (f32) value keeping track of the error accumulation.
///
/// This type represents the actual f32 (with the accumulated error), along with a lower and a
/// higher bounds representing what the maximum and minimum value could be without any error
/// involved.
///
/// Given that lower and higher bounds itself could exhibit errors, they are chosen in a
/// conservative way such that the real value is always between the bounds.
///
/// Using this type is 1/3 faster than using a double.
#[derive(Copy, Clone)]
pub struct Ef32 {
    // actual value
    val: f32,
    // minimum possible value without error
    low: f32,
    // maximum possible value without error
    high: f32,
}

impl Ef32 {
    /// Creates a new type with the given error.
    ///
    /// Usually for set values, the error is 0.0 and increases with each computation.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::utility::Ef32;
    ///
    /// let tracked = Ef32::new(0.1, 0.0);
    /// assert_eq!(tracked.value(), 0.1);
    /// assert_eq!(tracked.upper(), 0.1);
    /// assert_eq!(tracked.lower(), 0.1);
    /// ```
    pub fn new(val: f32, err: f32) -> Ef32 {
        if err == 0.0 {
            Ef32 {
                val,
                low: val,
                high: val,
            }
        } else {
            Ef32 {
                val,
                low: (val - err).previous_before(),
                high: (val + err).next_after(),
            }
        }
    }

    /// Retrieves the actual computed value (involving errors)
    /// # Examples
    /// Basic usage:
    /// ```
    /// use assert_approx_eq::assert_approx_eq;
    /// use glaze::utility::Ef32;
    ///
    /// let val0 = Ef32::new(0.1, 0.0);
    /// let val1 = Ef32::new(0.2, 0.0);
    /// let res = (val0 + val1).value();
    ///
    /// assert_approx_eq!(res, 0.3);
    /// ```
    pub fn value(&self) -> f32 {
        self.val
    }

    /// Retrieves the value of the float, assuming any approximation resulted in rounding towards
    /// `-∞`.
    /// This is the highest value the float can assume with the given operations.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::utility::Ef32;
    ///
    /// let val0 = Ef32::new(0.1, 0.0);
    /// let val1 = Ef32::new(0.2, 0.0);
    /// let res = (val0 + val1);
    ///
    /// assert!(res.upper() >= 0.3);
    /// ```
    pub fn upper(&self) -> f32 {
        self.high
    }

    /// Retrieves the value of the float, assuming any approximation resulted in rounding towards
    /// `+∞`.
    /// This is the lowest value the float can assume with the given operations.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::utility::Ef32;
    ///
    /// let val0 = Ef32::new(0.1, 0.0);
    /// let val1 = Ef32::new(0.2, 0.0);
    /// let res = (val0 + val1);
    ///
    /// assert!(res.lower() <= 0.3);
    /// ```
    pub fn lower(&self) -> f32 {
        self.low
    }

    /// Performs a square root on this float and updates the error accordingly.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use assert_approx_eq::assert_approx_eq;
    /// use glaze::utility::Ef32;
    /// use std::f32;
    ///
    /// let val0 = Ef32::new(2.0, 0.0);
    /// let root = val0.sqrt();
    ///
    /// assert_approx_eq!(root.value(), f32::consts::SQRT_2);
    /// assert!(root.upper() >= f32::consts::SQRT_2);
    /// assert!(root.lower() <= f32::consts::SQRT_2);
    /// ```
    pub fn sqrt(&self) -> Ef32 {
        Ef32 {
            val: self.val.sqrt(),
            low: (self.low.sqrt()).previous_before(),
            high: (self.high.sqrt()).next_after(),
        }
    }

    /// Calculates the absolute value of a float and updates the error accordingly.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use assert_approx_eq::assert_approx_eq;
    /// use glaze::utility::Ef32;
    ///
    /// let val0 = Ef32::new(-0.1, 0.0);
    /// let abs = val0.abs();
    ///
    /// assert_approx_eq!(abs.value(), 0.1);
    /// ```
    pub fn abs(&self) -> Ef32 {
        if self.low >= 0.0 {
            *self
        } else if self.high <= 0.0 {
            -self
        } else {
            Ef32 {
                val: self.val.abs(),
                low: 0.0,
                high: self.high.max(-self.low),
            }
        }
    }
}

overload!((a: ?Ef32) + (b: ?Ef32) -> Ef32 {
Ef32 {
    val: a.val+b.val,
    low: (a.low + b.low).previous_before(),
    high: (a.high + b.high).next_after(),
}
});

overload!((a: ?Ef32) - (b: ?Ef32) -> Ef32 {
Ef32 {
    val: a.val-b.val,
    low: (a.low - b.high).previous_before(),
    high: (a.high - b.low).next_after(),
}
});

overload!((a: ?Ef32) * (b: ?Ef32) -> Ef32 {
let prods = [a.low*b.low, a.high*b.low, a.low*b.high, a.high*b.high];
Ef32 {
    val: a.val*b.val,
    low: ((prods[0].min(prods[1])).min(prods[2].min(prods[3]))).previous_before(),
    high: ((prods[0].max(prods[1])).max(prods[2].max(prods[3]))).next_after(),
}
});

overload!((a: ?Ef32) / (b: ?Ef32) -> Ef32 {
let divs = [a.low/b.low, a.high/b.low, a.low/b.high, a.high/b.high];
let error = if b.low < 0.0 && b.high > 0.0 {
     //dividing by 0.0±𝜀 means the resulting number is completely fucked up
    (-std::f32::INFINITY, std::f32::INFINITY)
} else {
((divs[0].min(divs[1])).min(divs[2].min(divs[3])),(divs[0].max(divs[1])).max(divs[2].max(divs[3])))
};
Ef32 {
    val: a.val/b.val,
    low: (error.0).previous_before(),
    high: (error.1).next_after(),
}
});

overload!(- (a: ?Ef32) -> Ef32 {
Ef32 {
    val: -a.val,
    low: -a.high,
    high: -a.low,
}
});

impl std::fmt::Display for Ef32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "EF~{} [{} - {}]", self.val, self.low, self.high)
    }
}

/// Solves a quadratic equation using error tracking floats.
/// Equation is in the form `ax^2+bx+c=0`.
/// If `b^2-4ac >= 0` returns the two results, otherwise None
pub(crate) fn quadratic(a: Ef32, b: Ef32, c: Ef32) -> Option<(Ef32, Ef32)> {
    let discriminant = b.val * b.val - 4. * a.val * c.val;
    if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        let root_ef = Ef32::new(root, std::f32::EPSILON * 0.5 * root);
        let q = if b.val < 0.0 {
            (b - root_ef) * Ef32::new(-0.5, 0.0)
        } else {
            (b + root_ef) * Ef32::new(-0.5, 0.0)
        };
        let t0 = q / a;
        let t1 = c / q;
        if t0.val <= t1.val {
            Some((t0, t1))
        } else {
            Some((t1, t0))
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::utility::efloat::{quadratic, NextRepresentable};
    use crate::utility::{lerp, Ef32};
    use assert_approx_eq::assert_approx_eq;
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
        // Shamelessly copied from pbrt-v3. Their float model, Their tests.
        // (it's not rocket science, just random operations and check that precision holds)
        let mut rng = Xoshiro256StarStar::seed_from_u64(FIXED_SEED);
        for _ in 0..RANDOM_ITERATIONS {
            //gen random float
            let float1 = gen_f32(&mut rng, -4.0, 4.0);
            let mut precise1 = float1 as f64;
            let mut tracked1 = Ef32::new(float1, 0.0);
            for _ in 0..100 {
                if !precise1.is_finite() || !tracked1.value().is_finite() {
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
                            tracked2.value() as f64
                        };
                    }
                    _ => panic!(),
                }
                // random operation
                match rng.gen_range(0, 7) {
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
                        precise1 = precise1.abs();
                    }
                    _ => panic!(),
                }
            }
        }
    }

    #[test]
    fn efloat_display() {
        let two = Ef32::new(2.0, 0.0);
        let sqrt = two.sqrt();
        let str = format!("{}", &sqrt);
        assert_eq!(str, "EF~1.4142135 [1.4142134 - 1.4142137]");
    }

    #[test]
    // Δ > 0
    fn quadratic_two_sol() {
        let a = Ef32::new(2.0, 0.0);
        let b = Ef32::new(1.0, 0.0);
        let c = Ef32::new(-1.0, 0.0);
        let sol = quadratic(a, b, c);
        assert!(sol.is_some());
        let q = sol.unwrap();
        assert_approx_eq!((q.0).value(), -1.0);
        assert_approx_eq!((q.1).value(), 0.5);
    }

    #[test]
    // Δ = 0, very rare but could happen
    fn quadratic_one_sol() {
        let a = Ef32::new(2.0, 0.0);
        let b = Ef32::new(1.0, 0.0);
        let c = Ef32::new(0.125, 0.0);
        let sol = quadratic(a, b, c);
        assert!(sol.is_some());
        let q = sol.unwrap();
        assert_approx_eq!((q.0).value(), -0.25);
        assert_approx_eq!((q.1).value(), -0.25);
    }

    #[test]
    // Δ < 0
    fn quadratic_zero_sol() {
        let a = Ef32::new(2.0, 0.0);
        let b = Ef32::new(1.0, 0.0);
        let c = Ef32::new(1.0, 0.0);
        let sol = quadratic(a, b, c);
        assert!(sol.is_none());
    }

    #[test]
    // negative b performs a different algorithm (to minimize errors)
    fn quadratic_neg_b() {
        let a = Ef32::new(1.0, 0.0);
        let b = Ef32::new(-2.0, 0.0);
        let c = Ef32::new(0.75, 0.0);
        let sol = quadratic(a, b, c);
        assert!(sol.is_some());
        let q = sol.unwrap();
        assert_approx_eq!((q.0).value(), 0.5);
        assert_approx_eq!((q.1).value(), 1.5);
    }
}
