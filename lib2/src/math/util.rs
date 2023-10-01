/// Returns the result of `ceil(log2(val))`
pub fn ilog2_ceil(val: u64) -> u8 {
    if val == 1 {
        0
    } else {
        (u64::ilog2(val - 1) + 1) as u8
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Stats {
    avg: f32,
    var: f32,
    min: f32,
    max: f32,
}

impl Stats {
    pub fn new(values: &[f32]) -> Self {
        if !values.is_empty() {
            let avg = values.iter().sum::<f32>() / values.len() as f32;
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            let mut var = 0.0;
            for &val in values {
                // cant use iterators because f32 does not support ord
                min = min.min(val);
                max = max.max(val);
                var += (val - avg) * (val - avg);
            }
            var /= values.len() as f32;
            Self { avg, var, min, max }
        } else {
            Default::default()
        }
    }

    pub fn mean(&self) -> f32 {
        self.avg
    }

    pub fn variance(&self) -> f32 {
        self.var
    }

    pub fn standard_deviation(&self) -> f32 {
        self.var.sqrt()
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn max(&self) -> f32 {
        self.max
    }
}
