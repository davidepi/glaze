use crate::geometry::{Matrix4, Point3, Transform3};
use overload::overload;
use std::f32::INFINITY;
use std::ops;

/// An axis aligned bounding box
///
/// This class represents a bounding volume, a volume that contains a specific set of objects, in
/// this case points belonging to the Point3 class.
/// This bounding volume is a Box, as the name implies, and it's aligned with the axis of the scene.
///
/// This bounding box is defined by two points defined as private variables the point on the front
/// bottom left of the box and the one on the back top right.
#[derive(Copy, Clone)]
pub struct AABB {
    /// The front bottom left point of the AABB
    pub(super) bot: Point3,
    /// The back top right point of the AABB
    pub(super) top: Point3,
}

impl AABB {
    /// Constructs a bounding box with the bottom left pointing to Infinity and the top right
    /// pointing to -Infinity. This bounding box is degenerated thus it works exactly as an empty
    /// box, as any point can be added to it, however, some methods may return unexpected values.
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
    /// use glaze::geometry::Point3;
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

    /// Constructs a bounding box given two points in the space. No restriction over the points
    /// order or their minimum or maximum component is imposed.
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
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
    /// use glaze::geometry::Point3;
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
    /// point
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
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
    /// use glaze::geometry::Point3;
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
    /// use glaze::geometry::Point3;
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

    /// Returns the total surface of the bounding box
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
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

    /// Returns the total volume of the bounding box
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
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

    /// Returns the longest axis of the bounding box. The possible return values are:
    /// - `0` - if the `x` axis is the longest
    /// - `1` - if the `y` axis is the longest
    /// - `2` - if the `z` axis is the longest
    ///
    /// Out of two or more axes having the same length, the first appearing in the previous list is
    /// returned
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

    /// Returns the point at the center of the bounding box
    /// # Examples
    /// ```
    /// use glaze::geometry::Point3;
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
