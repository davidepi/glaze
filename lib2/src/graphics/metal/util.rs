use crate::math::Extent2D;
use core_graphics::geometry::CGSize;
use num_traits::AsPrimitive;

impl<T: AsPrimitive<f64>> Extent2D<T> {
    pub fn to_cgsize(self) -> CGSize {
        CGSize {
            width: self.x.as_(),
            height: self.y.as_(),
        }
    }
}
