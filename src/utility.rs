/// Performs an equality comparison between two single precision floats with a given margin of
/// error. This error is represented by `epsilon` and is the maximum allowed difference between the
/// two numbers to still be counted as equality. A negative value will result in every finite number
/// being equals to any other.
/// # Examples
/// ```
/// use glaze::utility::float_eq;
///
/// assert_ne!(0.1+0.2, 0.3);
/// assert!(float_eq(0.1+0.2, 0.3, 1E-5));
/// ```
pub fn float_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() <= epsilon
}
