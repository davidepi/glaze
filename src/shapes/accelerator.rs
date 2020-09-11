use crate::shapes::Shape;
use std::slice::Iter;

/// A trait for structs used to speed up Ray - Shape intersections.
///
/// The purpose of these structures is to reduces the number of intersection required by discarding
/// groups of shapes that the ray will never intersect without trying the actual intersection.
pub trait Accelerator: Shape {
    /// The types contained in the accelerator structure.
    ///
    /// **Must** implement Shape for obvious reason, as any structure implementing Accelerator will
    /// rely on the intersection of underlying shapes.
    type Item: Shape;

    /// Builds the acceleration structure and add takes ownership of the input elements.
    ///
    /// This method returns a newly built acceleration structure that should replace/invalidate the
    /// old one.
    /// # Examples
    /// Basic usage with Kd-trees:
    /// ```
    /// use glaze::geometry::{Point3, Ray, Vec3};
    /// use glaze::shapes::{Accelerator, KdTree, Shape, Sphere};
    ///
    /// let shapes = vec![Sphere::new()];
    /// let ray = Ray::new(&Point3::new(-10.0, 0.0, 0.0), &Vec3::new(1.0, 0.0, 0.0));
    ///
    /// assert!(shapes[0].intersect_fast(&ray));
    ///
    /// let kdtree = KdTree::default().build(shapes);
    ///
    /// assert!(kdtree.intersect_fast(&ray));
    /// ```
    #[must_use]
    fn build(self, elements: Vec<Self::Item>) -> Self;

    /// Iterates all the elements contained inside the accelerator structure.
    ///
    /// No particular order is guaranteed.
    /// # Examples
    /// Basic usage with Kd-trees:
    /// ```
    /// use glaze::shapes::{Accelerator, KdTree, Sphere};
    ///
    /// let shapes = vec![Sphere::new()];
    /// let kdtree = KdTree::default().build(shapes);
    /// let mut iterator = kdtree.iter();
    ///
    /// assert!(iterator.next().is_some());
    /// assert!(iterator.next().is_none());
    /// ```
    fn iter(&self) -> Iter<Self::Item>;
}
