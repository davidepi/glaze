mod inlines;
pub use self::inlines::float_eq;
pub use self::inlines::lerp;
mod efloat;
pub(crate) use self::efloat::quadratic;
pub(crate) use self::efloat::Ef32;

#[cfg(test)]
mod tests;
