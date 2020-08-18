/// Performs an equality comparison between two single precision floats with a given margin of
/// error.
///
/// This error is represented by `epsilon` and is the maximum allowed difference between the
/// two numbers to still be counted as equality. A negative value will result in every finite number
/// being equals to any other.
/// # Examples
/// Basic usage:
/// ```
/// use glaze::utility::float_eq;
///
/// assert_ne!(0.1 + 0.2, 0.3);
/// assert!(float_eq(0.1 + 0.2, 0.3, 1E-5));
/// ```
#[inline]
pub fn float_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() <= epsilon
}

/// Performs a linear interpolation.
///
/// Maps a value between [`0.0`, `1.0`] in an interval [`min`, `max`].
/// # Examples
/// Basic usage:
/// ```
/// use glaze::utility::lerp;
///
/// let res = lerp(0.2, 0.0, 10.0);
/// assert_eq!(res, 2.0);
/// ```
#[inline]
pub fn lerp(value: f32, min: f32, max: f32) -> f32 {
    (1.0 - value) * min + value * max
}

/// Restricts a value to a certain interval.
///
/// Used as a replacement while
/// [the std version](https://doc.rust-lang.org/std/primitive.f32.html#method.clamp)
/// is nightly-only
/// # Examples
/// Basic usage:
/// ```
/// // ez example stolen from the above link
/// use glaze::utility::clamp;
///
/// assert_eq!(clamp(-3.0, -2.0, 1.0), -2.0);
/// assert_eq!(clamp(0.0, -2.0, 1.0), 0.0);
/// assert_eq!(clamp(2.0, -2.0, 1.0), 1.0);
/// ```
//TODO: replace all occurrences with std version when it will be stabilized
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value <= min {
        min
    } else if value >= max {
        max
    } else {
        value
    }
}