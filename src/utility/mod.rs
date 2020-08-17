mod inlines;
pub use self::inlines::float_eq;
pub use self::inlines::lerp;
mod efloat;
pub(crate) use self::efloat::gamma;
pub use self::efloat::Ef32;
pub use self::efloat::NextRepresentable;

#[cfg(test)]
mod tests;
