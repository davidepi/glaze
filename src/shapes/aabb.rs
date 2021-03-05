use crate::linear::{Matrix4, Point3, Transform3};
use overload::overload;
use std::f32::INFINITY;
use std::ops;

/// An axis aligned bounding box.
///
/// This class represents a bounding volume, a volume that contains a specific set of objects, in
/// this case points belonging to the Point3 class.
/// This bounding volume is a Box, as the name implies, and it's aligned with the axis of the scene.
///
/// This bounding box is defined by two points defined as private variables the point on the front
/// bottom left of the box and the one on the back top right.
#[derive(Copy, Clone)]
pub struct AABB {
    /// The front bottom left point of the AABB.
    pub(super) bot: Point3,
    /// The back top right point of the AABB.
    pub(super) top: Point3,
}

impl AABB {
    /// Constructs a bounding box with the bottom left pointing to Infinity and the top right
    /// pointing to -Infinity.
    ///
    /// This bounding box is degenerated thus it works exactly as an empty box, as any point can be
    /// added to it, however, some methods may return unexpected values.
    /// # Examples
    /// ```
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::zero();
    ///
    /// assert_eq!(aabb.surface(), std::f32::INFINITY);
    /// ```
    pub fn zero() -> AABB {
        AABB {
            bot: Point3::new(INFINITY, INFINITY, INFINITY),
            top: Point3::new(-INFINITY, -INFINITY, -INFINITY),
        }
    }

    /// Constructs a zero-sized bounding box composed solely by one, given, point.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::point(&Point3::zero());
    ///
    /// assert_eq!(aabb.surface(), 0.0);
    /// ```
    pub fn point(point: &Point3) -> AABB {
        AABB {
            bot: *point,
            top: *point,
        }
    }

    /// Constructs a bounding box given two points in the space.
    ///
    /// No restriction over the points order or their minimum or maximum component is imposed.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    ///
    /// assert_eq!(aabb.surface(), 6.0);
    /// ```
    pub fn new(point_a: &Point3, point_b: &Point3) -> AABB {
        AABB {
            bot: Point3::min(&point_a, &point_b),
            top: Point3::max(&point_a, &point_b),
        }
    }

    /// Expands the bounding box by a given amount in every axis, both positive and negative.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    /// assert_eq!(aabb.surface(), 6.0); //each face has surface == 1.0
    ///
    /// let expanded = aabb.expand(1.0);
    /// assert_eq!(expanded.surface(), 54.0); // each face has surface == 9.0
    /// ```
    pub fn expand(&self, value: f32) -> AABB {
        AABB {
            bot: Point3::new(self.bot.x - value, self.bot.y - value, self.bot.z - value),
            top: Point3::new(self.top.x + value, self.top.y + value, self.top.z + value),
        }
    }

    /// Creates a new, bigger, bounding box that encloses both the old bounding box and the given
    /// point.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    /// assert_eq!(aabb.surface(), 6.0);
    ///
    /// let new_point = Point3::new(2.0, 2.0, 2.0);
    /// let bigger_aabb = aabb.engulf(&new_point);
    /// assert_eq!(bigger_aabb.surface(), 24.0); // each face has surface == 4.0
    /// ```
    pub fn engulf(&self, point: &Point3) -> AABB {
        AABB {
            bot: Point3::min(&self.bot, point),
            top: Point3::max(&self.top, point),
        }
    }

    /// Creates a new, bigger, bounding box that encloses two bounding boxes: the current one and
    /// the one passed as input parameter.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb0 = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    /// let aabb1 = AABB::new(&Point3::new(3.0, 3.0, 3.0), &Point3::new(4.0, 4.0, 4.0));
    /// let bigger = aabb0.merge(&aabb1);
    ///
    /// assert_eq!(bigger.surface(), 96.0); // each face has surface == 16.0
    /// ```
    pub fn merge(&self, aabb: &AABB) -> AABB {
        AABB {
            bot: Point3::min(&self.bot, &aabb.bot),
            top: Point3::max(&self.top, &aabb.top),
        }
    }

    /// Returns true if the given point is inside the bounding box, false otherwise.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    /// let point = Point3::new(0.5, 0.7, 0.3);
    ///
    /// assert!(aabb.contains(&point));
    /// ```
    pub fn contains(&self, point: &Point3) -> bool {
        point.x >= self.bot.x
            && point.x <= self.top.x
            && point.y >= self.bot.y
            && point.y <= self.top.y
            && point.z >= self.bot.z
            && point.z <= self.top.z
    }

    /// Returns the total surface of the bounding box.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    ///
    /// assert_eq!(aabb.surface(), 6.0);
    /// ```
    pub fn surface(&self) -> f32 {
        let diagonal = self.top - self.bot;
        2.0 * (diagonal.x * diagonal.y + diagonal.x * diagonal.z + diagonal.y * diagonal.z)
    }

    /// Returns the total volume of the bounding box.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(2.0, 2.0, 2.0));
    ///
    /// assert_eq!(aabb.volume(), 8.0);
    /// ```
    pub fn volume(&self) -> f32 {
        let diagonal = self.top - self.bot;
        diagonal.x * diagonal.y * diagonal.z
    }

    /// Returns the longest axis of the bounding box.
    ///
    /// The possible return values are:
    /// - `0` - if the `x` axis is the longest
    /// - `1` - if the `y` axis is the longest
    /// - `2` - if the `z` axis is the longest
    ///
    /// Out of two or more axes having the same length, the first appearing in the previous list is
    /// returned.
    pub fn longest_axis(&self) -> u8 {
        let diagonal = self.top - self.bot;
        if diagonal.x >= diagonal.y && diagonal.x >= diagonal.z {
            0
        } else if diagonal.y >= diagonal.z {
            1
        } else {
            2
        }
    }

    /// Returns the point at the center of the bounding box.
    /// # Examples
    /// ```
    /// use glaze::linear::Point3;
    /// use glaze::shapes::AABB;
    ///
    /// let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
    ///
    /// let centre = aabb.center();
    /// assert_eq!(centre.x, 0.5);
    /// assert_eq!(centre.y, 0.5);
    /// assert_eq!(centre.z, 0.5);
    /// ```
    pub fn center(&self) -> Point3 {
        Point3 {
            x: (self.bot.x + self.top.x) * 0.5,
            y: (self.bot.y + self.top.y) * 0.5,
            z: (self.bot.z + self.top.z) * 0.5,
        }
    }
}

impl Transform3 for AABB {
    fn transform(&self, mat: &Matrix4) -> Self {
        // transform all 8 points of the box, transforming two corners and taking their min/max
        // is not sufficient
        let p = [
            self.bot.transform(mat),
            Point3::new(self.top.x, self.bot.y, self.bot.z).transform(mat),
            Point3::new(self.top.x, self.top.y, self.bot.z).transform(mat),
            Point3::new(self.bot.x, self.top.y, self.bot.z).transform(mat),
            Point3::new(self.bot.x, self.bot.y, self.top.z).transform(mat),
            Point3::new(self.top.x, self.bot.y, self.top.z).transform(mat),
            Point3::new(self.bot.x, self.top.y, self.top.z).transform(mat),
            self.top.transform(mat),
        ];
        // take their min and max as new top/bot
        // TODO: replace with https://github.com/rust-lang/rust/issues/68125 when stable
        let bot = p.iter().fold(p[0], |bot, x| Point3::min(&bot, x));
        let top = p.iter().fold(p[0], |top, x| Point3::max(&top, x));

        AABB { bot, top }
    }
}

overload!((a: ?AABB) + (b: &Point3) -> AABB {
AABB {bot: Point3::min(&a.bot, b),top: Point3::max(&a.top, b),}});
overload!((a: ?AABB) + (b: Point3) -> AABB {
AABB {bot: Point3::min(&a.bot, &b),top: Point3::max(&a.top, &b),}});
overload!((a: ?AABB) + (b: ?AABB) -> AABB {
AABB {bot: Point3::min(&a.bot, &b.bot),top: Point3::max(&a.top, &b.top),}});
overload!((a: &mut AABB) += (b: &Point3)
{a.bot=Point3::min(&a.bot, b);a.top=Point3::max(&a.top, b);});
overload!((a: &mut AABB) += (b: Point3)
{a.bot=Point3::min(&a.bot, &b);a.top=Point3::max(&a.top, &b);});
overload!((a: &mut AABB) += (b: ?AABB)
{a.bot=Point3::min(&a.bot, &b.bot);a.top=Point3::max(&a.top, &b.top);});

#[cfg(test)]
mod tests {
    use crate::linear::{Matrix4, Point3, Transform3};
    use crate::shapes::AABB;
    use assert_approx_eq::assert_approx_eq;
    use std::f32::INFINITY;

    #[test]
    fn aabb_zero() {
        let aabb = AABB::zero();
        assert!((aabb.bot.x).is_infinite());
        assert!((aabb.bot.y).is_infinite());
        assert!((aabb.bot.z).is_infinite());
        assert!((aabb.top.x).is_infinite());
        assert!((aabb.top.y).is_infinite());
        assert!((aabb.top.z).is_infinite());
    }

    #[test]
    fn aabb_new() {
        let pmin = Point3::new(60.06347, 33.59238, -37.23738);
        let pmax = Point3::new(-21.18293, 50.33405, 9.33384);
        let aabb = AABB::new(&pmin, &pmax);
        assert_eq!(aabb.bot.x, pmax.x);
        assert_eq!(aabb.bot.y, pmin.y);
        assert_eq!(aabb.bot.z, pmin.z);
        assert_eq!(aabb.top.x, pmin.x);
        assert_eq!(aabb.top.y, pmax.y);
        assert_eq!(aabb.top.z, pmax.z);
    }

    #[test]
    fn aabb_from_point() {
        let p = Point3::new(76.9907, -31.02559, 64.63251);
        let aabb = AABB::point(&p);
        assert_eq!(aabb.bot.x, p.x);
        assert_eq!(aabb.bot.y, p.y);
        assert_eq!(aabb.bot.z, p.z);
        assert_eq!(aabb.top.x, p.x);
        assert_eq!(aabb.top.y, p.y);
        assert_eq!(aabb.top.z, p.z);
    }

    #[test]
    fn aabb_expand_positive() {
        let zero = Point3::zero();
        let one = Point3::new(1.0, 1.0, 1.0);
        let aabb = AABB::new(&zero, &one).expand(50.50);
        assert_approx_eq!(aabb.bot.x, -50.5, 1e-5);
        assert_approx_eq!(aabb.bot.y, -50.5, 1e-5);
        assert_approx_eq!(aabb.bot.z, -50.5, 1e-5);
        assert_approx_eq!(aabb.top.x, 51.5, 1e-5);
        assert_approx_eq!(aabb.top.y, 51.5, 1e-5);
        assert_approx_eq!(aabb.top.z, 51.5, 1e-5);
    }

    #[test]
    fn aabb_expand_negative() {
        let min = Point3::new(-50.50, -50.50, -50.50);
        let max = Point3::new(50.50, 50.50, 50.50);
        let aabb = AABB::new(&min, &max).expand(-0.50);
        assert_approx_eq!(aabb.bot.x, -50.0, 1e-5);
        assert_approx_eq!(aabb.bot.y, -50.0, 1e-5);
        assert_approx_eq!(aabb.bot.z, -50.0, 1e-5);
        assert_approx_eq!(aabb.top.x, 50.0, 1e-5);
        assert_approx_eq!(aabb.top.y, 50.0, 1e-5);
        assert_approx_eq!(aabb.top.z, 50.0, 1e-5);
    }

    #[test]
    fn aabb_engulf_point_outside() {
        let aabb = AABB::point(&Point3::zero());
        let addme = Point3::new(10.0, -15.0, 20.0);
        let merged = aabb.engulf(&addme);
        assert_eq!(merged.bot.x, 0.0);
        assert_eq!(merged.bot.y, -15.0);
        assert_eq!(merged.bot.z, 0.0);
        assert_eq!(merged.top.x, 10.0);
        assert_eq!(merged.top.y, 0.0);
        assert_eq!(merged.top.z, 20.0);
    }

    #[test]
    fn aabb_engulf_point_inside() {
        let aabb = AABB::new(
            &Point3::new(-5.0, -5.0, -5.0),
            &Point3::new(30.0, 30.0, 30.0),
        );
        let addme = Point3::new(2.0, -3.0, 15.0);
        let merged = aabb.engulf(&addme);
        assert_eq!(merged.bot.x, aabb.bot.x);
        assert_eq!(merged.bot.y, aabb.bot.y);
        assert_eq!(merged.bot.z, aabb.bot.z);
        assert_eq!(merged.top.x, aabb.top.x);
        assert_eq!(merged.top.y, aabb.top.y);
        assert_eq!(merged.top.z, aabb.top.z);
    }

    #[test]
    fn aabb_engulf_point_infinite() {
        let aabb = AABB::point(&Point3::zero());
        let addme = Point3::new(INFINITY, -INFINITY, 0.0);
        let merged = aabb.engulf(&addme);
        assert_approx_eq!(merged.bot.x, 0.0, 1e-5);
        assert!(merged.bot.y.is_infinite());
        assert_approx_eq!(merged.bot.z, 0.0, 1e-5);
        assert!(merged.top.x.is_infinite());
        assert_approx_eq!(merged.top.y, 0.0, 1e-5);
        assert_approx_eq!(merged.top.z, 0.0, 1e-5);
    }

    #[test]
    fn aabb_merge_aabb() {
        let aabb = AABB::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 1.0, 1.0));
        let outside = AABB::new(
            &Point3::new(59.28244, -3.01509, 47.61078),
            &Point3::new(67.30925, 53.29163, 82.07844),
        );
        let merged = aabb.merge(&outside);
        assert_eq!(merged.bot.x, aabb.bot.x);
        assert_eq!(merged.bot.y, outside.bot.y);
        assert_eq!(merged.bot.z, aabb.bot.z);
        assert_eq!(merged.top.x, outside.top.x);
        assert_eq!(merged.top.y, outside.top.y);
        assert_eq!(merged.top.z, outside.top.z);
    }
    #[test]
    fn aabb_merge_aabb_inside() {
        let aabb = AABB::new(
            &Point3::new(-100.0, -100.0, -100.0),
            &Point3::new(100.0, 100.0, 100.0),
        );
        let point_inside = AABB::new(
            &Point3::new(-9.30374, 8.49896, -35.41399),
            &Point3::new(58.56126, 18.59649, 37.76507),
        );
        let merged = aabb.merge(&point_inside);
        assert_eq!(merged.bot.x, aabb.bot.x);
        assert_eq!(merged.bot.y, aabb.bot.y);
        assert_eq!(merged.bot.z, aabb.bot.z);
        assert_eq!(merged.top.x, aabb.top.x);
        assert_eq!(merged.top.y, aabb.top.y);
        assert_eq!(merged.top.z, aabb.top.z);
    }

    #[test]
    fn aabb_merge_aabb_infinite() {
        let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
        let target = AABB::new(
            &Point3::new(-INFINITY, -73.22298, 53.70019),
            &Point3::new(-138.73003, INFINITY, INFINITY),
        );
        let res = aabb.merge(&target);
        assert!(res.bot.x.is_infinite());
        assert_approx_eq!(res.bot.y, target.bot.y, 1e-5);
        assert_approx_eq!(res.bot.z, aabb.bot.z, 1e-5);
        assert_approx_eq!(res.top.x, aabb.top.x, 1e-5);
        assert!(res.top.y.is_infinite());
        assert!(res.top.z.is_infinite());
    }

    #[test]
    fn aabb_contains_point_inside() {
        let aabb = AABB::new(
            &Point3::new(-10.0, -10.0, -10.0),
            &Point3::new(10.0, 10.0, 10.0),
        );
        let ins = Point3::new(-5.0, -5.0, -5.0);
        assert!(aabb.contains(&ins));
    }

    #[test]
    fn aabb_contains_point_on_border() {
        let aabb = AABB::new(
            &Point3::new(-10.0, -10.0, -10.0),
            &Point3::new(10.0, 10.0, 10.0),
        );
        let border = Point3::new(10.0, 10.0, 10.0);
        assert!(aabb.contains(&border));
    }

    #[test]
    fn aabb_contains_point_outside() {
        let aabb = AABB::new(
            &Point3::new(-10.0, -10.0, -10.0),
            &Point3::new(10.0, 10.0, 10.0),
        );
        let only_x = Point3::new(-10.000001, -5.0, -5.0);
        let only_y = Point3::new(-5.0, -10.000001, -5.0);
        let only_z = Point3::new(-5.0, -5.0, -10.000001);
        let all = Point3::new(11.0, 11.0, 11.0);
        assert!(!aabb.contains(&only_x));
        assert!(!aabb.contains(&only_y));
        assert!(!aabb.contains(&only_z));
        assert!(!aabb.contains(&all));
    }
    #[test]
    fn aabb_contains_point_infinite() {
        let aabb = AABB::new(
            &Point3::new(-10.0, -10.0, -10.0),
            &Point3::new(10.0, 10.0, 10.0),
        );
        let inf = Point3::new(INFINITY, 0.0, 0.0);
        let minf = Point3::new(0.0, -INFINITY, 0.0);
        let inf2 = Point3::new(0.0, 0.0, INFINITY);
        assert!(!aabb.contains(&inf));
        assert!(!aabb.contains(&minf));
        assert!(!aabb.contains(&inf2));
    }

    #[test]
    fn aabb_surface_zero() {
        //corner case, zero length
        let p = Point3::new(-0.53123, -24.29362, 84.26433);
        let aabb = AABB::point(&p);
        assert_eq!(aabb.surface(), 0.0);
    }

    #[test]
    fn aabb_surface() {
        //normal length
        let aabb = AABB::new(&Point3::new(-1.0, -1.0, -1.0), &Point3::new(3.0, 4.0, 5.0));
        assert_eq!(aabb.surface(), 148.0);
    }

    #[test]
    fn aabb_surface_infinite() {
        //infinite length
        let aabb = AABB::new(
            &Point3::new(-1.0, -1.0, -1.0),
            &Point3::new(1.0, 1.0, INFINITY),
        );
        assert_eq!(aabb.surface(), INFINITY);
        let aabb2 = AABB::new(
            &Point3::new(-INFINITY, -INFINITY, -INFINITY),
            &Point3::new(INFINITY, INFINITY, INFINITY),
        );
        assert_eq!(aabb2.surface(), INFINITY);
    }

    #[test]
    fn aabb_volume_zero() {
        let p = Point3::new(-0.53123, -24.29362, 84.26433);
        let aabb = AABB::point(&p);
        assert_eq!(aabb.volume(), 0.0);
    }

    #[test]
    fn aabb_volume() {
        let aabb = AABB::new(&Point3::new(-1.0, -1.0, -1.0), &Point3::new(3.0, 4.0, 5.0));
        assert_eq!(aabb.volume(), 120.0);
    }

    #[test]
    fn aabb_volume_infinite() {
        let aabb = AABB::new(
            &Point3::new(-1.0, -1.0, -1.0),
            &Point3::new(1.0, 1.0, INFINITY),
        );
        assert_eq!(aabb.volume(), INFINITY);
        let aabb2 = AABB::new(
            &Point3::new(-INFINITY, -INFINITY, -INFINITY),
            &Point3::new(INFINITY, INFINITY, INFINITY),
        );
        assert_eq!(aabb2.volume(), INFINITY);
    }

    #[test]
    fn aabb_longest_axis_zero() {
        //zero length -> return x
        let p = Point3::new(-0.53123, -24.29362, 84.26433);
        let aabb = AABB::point(&p);
        assert_eq!(aabb.longest_axis(), 0);
    }

    #[test]
    fn aabb_longest_axis() {
        //longest x - non inf
        let aabbx = AABB::new(
            &Point3::new(-85.77731, 5.98468, -10.75332),
            &Point3::new(74.13619, 99.79995, 37.72758),
        );
        assert_eq!(aabbx.longest_axis(), 0);
        //longest y - non inf
        let aabby = AABB::new(
            &Point3::new(-27.68684, -73.58186, -69.54105),
            &Point3::new(65.46841, 95.43746, -51.04507),
        );
        assert_eq!(aabby.longest_axis(), 1);
        //longest z - non inf
        let aabbz = AABB::new(
            &Point3::new(17.90233, -6.71415, -88.93419),
            &Point3::new(76.75507, 50.73106, 95.81359),
        );
        assert_eq!(aabbz.longest_axis(), 2);
    }

    #[test]
    fn aabb_longest_axis_infinite() {
        //longest x - inf
        let aabb = AABB::new(
            &Point3::new(-INFINITY, 5.98468, -10.75332),
            &Point3::new(74.13619, 99.79995, 37.72758),
        );
        assert_eq!(aabb.longest_axis(), 0);
        //longest y - inf
        let aabb = AABB::new(
            &Point3::new(-27.68684, -73.58186, -69.54105),
            &Point3::new(65.46841, INFINITY, -51.04507),
        );
        assert_eq!(aabb.longest_axis(), 1);
        //longest z - inf
        let aabb = AABB::new(
            &Point3::new(17.90233, -46.71415, -INFINITY),
            &Point3::new(76.75507, 90.73106, 95.81359),
        );
        assert_eq!(aabb.longest_axis(), 2);
        //everything infinite
        let aabb = AABB::new(
            &Point3::new(-INFINITY, -INFINITY, -INFINITY),
            &Point3::new(INFINITY, INFINITY, INFINITY),
        );
        assert_eq!(aabb.longest_axis(), 0);
    }

    #[test]
    fn aabb_center_zero() {
        let p = Point3::new(-0.53123, -24.29362, 84.26433);
        let aabb = AABB::point(&p);
        let center = aabb.center();
        assert_approx_eq!(center.x, p.x);
        assert_approx_eq!(center.y, p.y);
        assert_approx_eq!(center.z, p.z);
    }

    #[test]
    fn aabb_center() {
        //normal aabb
        let aabb = AABB::new(&Point3::new(-1.0, -1.0, -1.0), &Point3::new(1.0, 1.0, 1.0));
        let center = aabb.center();
        assert_approx_eq!(center.x, 0.0, 1e-5);
        assert_approx_eq!(center.y, 0.0, 1e-5);
        assert_approx_eq!(center.z, 0.0, 1e-5);
    }

    #[test]
    fn aaabb_center_infinite() {
        //bot is infinite
        let aabb = AABB::new(
            &Point3::new(-1.0, -INFINITY, -1.0),
            &Point3::new(1.0, -1.0, 1.0),
        );
        let center = aabb.center();
        assert_approx_eq!(center.x, 0.0, 1e-5);
        assert!(center.y.is_infinite());
        assert_approx_eq!(center.z, 0.0, 1e-5);
        //top is infinite
        let aabb2 = AABB::new(
            &Point3::new(-1.0, -1.0, -1.0),
            &Point3::new(1.0, 1.0, INFINITY),
        );
        let center2 = aabb2.center();
        assert_approx_eq!(center2.x, 0.0, 1e-5);
        assert_approx_eq!(center2.y, 0.0, 1e-5);
        assert!(center2.z.is_infinite());
        //all infinite aabb
        let aabb3 = AABB::new(
            &Point3::new(-INFINITY, -INFINITY, -INFINITY),
            &Point3::new(INFINITY, INFINITY, INFINITY),
        );
        let center3 = aabb3.center();
        assert!((center3.x).is_nan());
        assert!((center3.y).is_nan());
        assert!((center3.z).is_nan());
    }

    #[test]
    fn aabb_sum_point() {
        let aabb = AABB::point(&Point3::zero());
        let addme = Point3::new(10.0, -15.0, 20.0);
        let merged = aabb + addme;
        assert_eq!(merged.bot.x, 0.0);
        assert_eq!(merged.bot.y, -15.0);
        assert_eq!(merged.bot.z, 0.0);
        assert_eq!(merged.top.x, 10.0);
        assert_eq!(merged.top.y, 0.0);
        assert_eq!(merged.top.z, 20.0);
    }

    #[test]
    fn aabb_sum_point_inside() {
        let aabb = AABB::new(
            &Point3::new(-5.0, -5.0, -5.0),
            &Point3::new(30.0, 30.0, 30.0),
        );
        let addme = Point3::new(2.0, -3.0, 15.0);
        let merged = aabb + addme;
        assert_eq!(merged.bot.x, aabb.bot.x);
        assert_eq!(merged.bot.y, aabb.bot.y);
        assert_eq!(merged.bot.z, aabb.bot.z);
        assert_eq!(merged.top.x, aabb.top.x);
        assert_eq!(merged.top.y, aabb.top.y);
        assert_eq!(merged.top.z, aabb.top.z);
    }

    #[test]
    fn aabb_sum_point_infinite() {
        let aabb = AABB::point(&Point3::zero());
        let addme = Point3::new(INFINITY, -INFINITY, 0.0);
        let merged = aabb + addme;
        assert_approx_eq!(merged.bot.x, 0.0, 1e-5);
        assert!(merged.bot.y.is_infinite());
        assert_approx_eq!(merged.bot.z, 0.0, 1e-5);
        assert!(merged.top.x.is_infinite());
        assert_approx_eq!(merged.top.y, 0.0, 1e-5);
        assert_approx_eq!(merged.top.z, 0.0, 1e-5);
    }

    #[test]
    fn aabb_sum_aabb() {
        let aabb = AABB::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 1.0, 1.0));
        let outside = AABB::new(
            &Point3::new(59.28244, -3.01509, 47.61078),
            &Point3::new(67.30925, 53.29163, 82.07844),
        );
        let merged = aabb + outside;
        assert_eq!(merged.bot.x, aabb.bot.x);
        assert_eq!(merged.bot.y, outside.bot.y);
        assert_eq!(merged.bot.z, aabb.bot.z);
        assert_eq!(merged.top.x, outside.top.x);
        assert_eq!(merged.top.y, outside.top.y);
        assert_eq!(merged.top.z, outside.top.z);
    }
    #[test]
    fn aabb_sum_aabb_inside() {
        let aabb = AABB::new(
            &Point3::new(-100.0, -100.0, -100.0),
            &Point3::new(100.0, 100.0, 100.0),
        );
        let inside = AABB::new(
            &Point3::new(-9.30374, 8.49896, -35.41399),
            &Point3::new(58.56126, 18.59649, 37.76507),
        );
        let merged = aabb + inside;
        assert_eq!(merged.bot.x, aabb.bot.x);
        assert_eq!(merged.bot.y, aabb.bot.y);
        assert_eq!(merged.bot.z, aabb.bot.z);
        assert_eq!(merged.top.x, aabb.top.x);
        assert_eq!(merged.top.y, aabb.top.y);
        assert_eq!(merged.top.z, aabb.top.z);
    }

    #[test]
    fn aabb_sum_aabb_infinite() {
        let aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
        let target = AABB::new(
            &Point3::new(-INFINITY, -73.22298, 53.70019),
            &Point3::new(-138.73003, INFINITY, INFINITY),
        );
        let res = aabb + target;
        assert!(res.bot.x.is_infinite());
        assert_approx_eq!(res.bot.y, target.bot.y, 1e-5);
        assert_approx_eq!(res.bot.z, aabb.bot.z, 1e-5);
        assert_approx_eq!(res.top.x, aabb.top.x, 1e-5);
        assert!(res.top.y.is_infinite());
        assert!(res.top.z.is_infinite());
    }

    #[test]
    fn aabb_sum_assign_point() {
        let mut aabb = AABB::point(&Point3::zero());
        let addme = Point3::new(10.0, -15.0, 20.0);
        aabb += addme;
        assert_eq!(aabb.bot.x, 0.0);
        assert_eq!(aabb.bot.y, -15.0);
        assert_eq!(aabb.bot.z, 0.0);
        assert_eq!(aabb.top.x, 10.0);
        assert_eq!(aabb.top.y, 0.0);
        assert_eq!(aabb.top.z, 20.0);
    }

    #[test]
    fn aabb_sum_assign_point_inside() {
        let mut aabb = AABB::new(
            &Point3::new(-5.0, -5.0, -5.0),
            &Point3::new(30.0, 30.0, 30.0),
        );
        let old = aabb;
        let addme = Point3::new(2.0, -3.0, 15.0);
        aabb += addme;
        assert_eq!(aabb.bot.x, old.bot.x);
        assert_eq!(aabb.bot.y, old.bot.y);
        assert_eq!(aabb.bot.z, old.bot.z);
        assert_eq!(aabb.top.x, old.top.x);
        assert_eq!(aabb.top.y, old.top.y);
        assert_eq!(aabb.top.z, old.top.z);
    }

    #[test]
    fn aabb_sum_assign_point_infinite() {
        let mut aabb = AABB::point(&Point3::zero());
        let addme = Point3::new(INFINITY, -INFINITY, 0.0);
        aabb += addme;
        assert_approx_eq!(aabb.bot.x, 0.0, 1e-5);
        assert!(aabb.bot.y.is_infinite());
        assert_approx_eq!(aabb.bot.z, 0.0, 1e-5);
        assert!(aabb.top.x.is_infinite());
        assert_approx_eq!(aabb.top.y, 0.0, 1e-5);
        assert_approx_eq!(aabb.top.z, 0.0, 1e-5);
    }

    #[test]
    fn aabb_sum_assign_aabb() {
        let mut aabb = AABB::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 1.0, 1.0));
        let old = aabb;
        let outside = AABB::new(
            &Point3::new(59.28244, -3.01509, 47.61078),
            &Point3::new(67.30925, 53.29163, 82.07844),
        );
        aabb += outside;
        assert_eq!(aabb.bot.x, old.bot.x);
        assert_eq!(aabb.bot.y, outside.bot.y);
        assert_eq!(aabb.bot.z, old.bot.z);
        assert_eq!(aabb.top.x, outside.top.x);
        assert_eq!(aabb.top.y, outside.top.y);
        assert_eq!(aabb.top.z, outside.top.z);
    }
    #[test]
    fn aabb_sum_assign_aabb_inside() {
        let mut aabb = AABB::new(
            &Point3::new(-100.0, -100.0, -100.0),
            &Point3::new(100.0, 100.0, 100.0),
        );
        let old = aabb;
        let inside = AABB::new(
            &Point3::new(-9.30374, 8.49896, -35.41399),
            &Point3::new(58.56126, 18.59649, 37.76507),
        );
        aabb += inside;
        assert_eq!(aabb.bot.x, old.bot.x);
        assert_eq!(aabb.bot.y, old.bot.y);
        assert_eq!(aabb.bot.z, old.bot.z);
        assert_eq!(aabb.top.x, old.top.x);
        assert_eq!(aabb.top.y, old.top.y);
        assert_eq!(aabb.top.z, old.top.z);
    }

    #[test]
    fn aabb_sum_assign_aabb_infinite() {
        let mut aabb = AABB::new(&Point3::zero(), &Point3::new(1.0, 1.0, 1.0));
        let old = aabb;
        let target = AABB::new(
            &Point3::new(-INFINITY, -73.22298, 53.70019),
            &Point3::new(-138.73003, INFINITY, INFINITY),
        );
        aabb += target;
        assert!(aabb.bot.x.is_infinite());
        assert_approx_eq!(aabb.bot.y, target.bot.y, 1e-5);
        assert_approx_eq!(aabb.bot.z, old.bot.z, 1e-5);
        assert_approx_eq!(aabb.top.x, old.top.x, 1e-5);
        assert!(aabb.top.y.is_infinite());
        assert!(aabb.top.z.is_infinite());
    }

    #[test]
    fn aabb_transform() {
        let one = Point3::new(1.0, 1.0, 1.0);
        let aabb = AABB::new(&-one, &one);
        let matrix = Matrix4::rotate_z((-45.0 as f32).to_radians());
        let transformed = aabb.transform(&matrix);

        assert_approx_eq!(transformed.bot.x, -1.414213);
        assert_approx_eq!(transformed.bot.y, -1.414213);
        assert_eq!(transformed.bot.z, -1.0);
        assert_approx_eq!(transformed.top.x, 1.414213);
        assert_approx_eq!(transformed.top.y, 1.414213);
        assert_eq!(transformed.top.z, 1.0);
    }
}
