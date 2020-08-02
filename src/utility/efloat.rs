use overload::overload;
use std::fmt::Formatter;
use std::ops;

pub trait NextRepresentable {
    fn next_after(self) -> Self;
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

#[derive(Copy, Clone)]
pub(crate) struct Ef32 {
    val: f32,
    low: f32,
    high: f32,
}

impl Ef32 {
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

    pub fn float(&self) -> f32 {
        self.val
    }

    pub fn upper(&self) -> f32 {
        self.high
    }

    pub fn lower(&self) -> f32 {
        self.low
    }

    pub fn sqrt(&self) -> Ef32 {
        Ef32 {
            val: self.val.sqrt(),
            low: (self.low.sqrt()).previous_before(),
            high: (self.high.sqrt()).next_after(),
        }
    }

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
    low: (a.low - b.low).previous_before(),
    high: (a.high - b.high).next_after(),
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
    low: -a.low,
    high: -a.high,
}
});

impl std::fmt::Display for Ef32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - [{}, {}]", self.val, self.low, self.high)
    }
}

pub(crate) fn quadratic(a: Ef32, b: Ef32, c: Ef32) -> Option<(Ef32, Ef32)> {
    let discriminant = b.val * b.val - 4. * a.val * c.val;
    if discriminant > 0.0 {
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
