mod inlines;
pub use self::inlines::clamp;
pub use self::inlines::float_eq;
pub use self::inlines::lerp;
mod efloat;
pub(crate) use self::efloat::gamma;
pub(crate) use self::efloat::quadratic;
pub use self::efloat::Ef32;
pub use self::efloat::NextRepresentable;
