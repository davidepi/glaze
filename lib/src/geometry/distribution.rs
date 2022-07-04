use num::{Float, Zero};

// based on pbrtv3 distribution class

/// A 1-dimensional distribution of values.
pub struct Distribution1D<T: Float + Zero> {
    cdf: Vec<T>,
    values: Vec<T>,
    integral: T,
}

impl<T: Float + Zero> Distribution1D<T> {
    /// Creates a new distribution with the current values.
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

    /// Returns the Cumulative Distribution Function for the current distribution.
    pub fn cdf(&self) -> &[T] {
        &self.cdf
    }

    /// Returns all the values of the current distribution.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the integral of the current distribution.
    pub fn integral(&self) -> T {
        self.integral
    }
}

/// A 2-dimensional distribution of values.
pub struct Distribution2D<T: Float + Zero> {
    conditional: Vec<Distribution1D<T>>,
    marginal: Distribution1D<T>,
}

impl<T: Float + Zero> Distribution2D<T> {
    /// Creates a new distribution with the current values and the amount of elements in the X
    /// dimension.
    pub fn new(values: impl Iterator<Item = T>, x_size: usize) -> Self {
        let vals = values.collect::<Vec<T>>();
        let conditional = vals
            .chunks_exact(x_size)
            .map(|s| Distribution1D::new(s.iter().copied()))
            .collect::<Vec<_>>();
        let marginal = Distribution1D::new(conditional.iter().map(Distribution1D::integral));
        Self {
            conditional,
            marginal,
        }
    }

    /// Returns every conditional distribution.
    ///
    /// There is a conditional distribution for each row in the X dimension of the original 2D
    /// distribution.
    pub fn conditional(&self) -> &[Distribution1D<T>] {
        &self.conditional
    }

    /// Returns the marginal distribution.
    /// There is a marginal distribution value for each row in the X dimension of the original 2D
    /// distribution.
    pub fn marginal(&self) -> &Distribution1D<T> {
        &self.marginal
    }

    /// Returns the dimensions of the 2D distribution values, in form `(x_size, y_size)`
    pub fn dimensions_values(&self) -> (usize, usize) {
        (self.conditional[0].values.len(), self.conditional.len())
    }

    /// Returns the dimensions of the 2D distribution cumulative distribution function,
    /// in form `(x_size, y_size)`
    pub fn dimensions_cdf(&self) -> (usize, usize) {
        (self.conditional[0].cdf.len(), self.conditional.len())
    }
}
