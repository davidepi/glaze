use crate::shapes::Shape;

pub trait Accelerator: Shape {
    type Item;

    #[must_use]
    fn build(self, elements: Vec<Self::Item>) -> Self;
}
