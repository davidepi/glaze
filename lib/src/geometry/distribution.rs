use num::{Float, Zero};

// based on pbrtv3 distribution class

pub struct Distribution1D<T: Float + Zero> {
    cdf: Vec<T>,
    values: Vec<T>,
    integral: T,
}

impl<T: Float + Zero> Distribution1D<T> {
    pub fn new(values: impl Iterator<Item = T>) -> Self {
        let values = values.collect::<Vec<_>>();
        let n = values.len();
        let n_as_t = T::from(n).unwrap();
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(T::zero());
        for i in 1..(n + 1) {
            cdf.push(cdf[i - 1] + values[i - 1] / n_as_t);
        }
        let integral = cdf[n];
        if integral.is_zero() {
            for (i, item) in cdf.iter_mut().enumerate().skip(1) {
                *item = T::from(i).unwrap() / n_as_t;
            }
        } else {
            for item in cdf.iter_mut().skip(1) {
                *item = *item / integral;
            }
        }
        Self {
            values,
            cdf,
            integral,
        }
    }

    pub fn cdf(&self) -> &[T] {
        &self.cdf
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn integral(&self) -> T {
        self.integral
    }
}

pub struct Distribution2D<T: Float + Zero> {
    conditional: Vec<Distribution1D<T>>,
    marginal: Distribution1D<T>,
}

impl<T: Float + Zero> Distribution2D<T> {
    pub fn new(values: impl Iterator<Item = T>, row_size: usize) -> Self {
        let vals = values.collect::<Vec<T>>();
        let conditional = vals
            .chunks_exact(row_size)
            .map(|s| Distribution1D::new(s.iter().copied()))
            .collect::<Vec<_>>();
        let marginal = Distribution1D::new(conditional.iter().map(Distribution1D::integral));
        Self {
            conditional,
            marginal,
        }
    }

    pub fn conditional(&self) -> &[Distribution1D<T>] {
        &self.conditional
    }

    pub fn marginal(&self) -> &Distribution1D<T> {
        &self.marginal
    }

    pub fn dimensions_values(&self) -> (usize, usize) {
        (self.conditional[0].values.len(), self.conditional.len())
    }

    pub fn dimensions_cdf(&self) -> (usize, usize) {
        (self.conditional[0].cdf.len(), self.conditional.len())
    }
}
