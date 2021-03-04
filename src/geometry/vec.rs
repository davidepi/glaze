use crate::geometry::matrix::{Matrix4, Transform3};
use log::warn;
use overload::overload;
use std::f32;
use std::fmt::Formatter;
use std::ops;
use std::ops::{Index, IndexMut};

/// A vector of two components.
///
/// Vec2 class represents a vector in a 2D space.
///
/// A Vec2 consist of two coordinates, usually called `x`, and `y`.
#[derive(Clone, Copy)]
pub struct Vec2 {
    /// Single precision floating point representing the `x` component of the vector. Indexed by 0.
    pub x: f32,
    /// Single precision floating point representing the `y` component of the vector. Indexed by 1.
    pub y: f32,
}

impl Vec2 {
    /// Constructs a 2D zero vector, a vector in the form `(0.0, 0.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::zero();
    ///
    /// assert_eq!(v.x, 0.0);
    /// assert_eq!(v.y, 0.0);
    /// ```
    pub fn zero() -> Vec2 {
        Vec2 { x: 0., y: 0. }
    }

    /// Constructs a vector with the given `(x, y)` components.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::new(3.5, -2.2);
    ///
    /// assert_eq!(v.x, 3.5);
    /// assert_eq!(v.y, -2.2);
    /// ```
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    /// Constructs a normalized vector pointing *right*, towards `(1.0, 0.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::right();
    ///
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 0.0);
    /// ```
    pub fn right() -> Vec2 {
        Vec2 { x: 1.0, y: 0.0 }
    }

    /// Constructs a normalized vector pointing *up*, towards `(0.0, 1.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::up();
    ///
    /// assert_eq!(v.x, 0.0);
    /// assert_eq!(v.y, 1.0);
    /// ```
    pub fn up() -> Vec2 {
        Vec2 { x: 0.0, y: 1.0 }
    }

    /// Returns the euclidean length (or magnitude) of the vector.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::new(3.0, 4.0);
    ///
    /// assert_eq!(v.length(), 5.0);
    /// ```
    pub fn length(&self) -> f32 {
        ((self.x * self.x) + (self.y * self.y)).sqrt()
    }

    /// Returns the squared euclidean length (or Manhattan length) of the vector.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::new(3.0, 4.0);
    ///
    /// assert_eq!(v.length2(), 25.0);
    /// ```
    pub fn length2(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Returns the normalized version of the current vector. A vector is normalized if its
    /// euclidean length is equal to 1.0.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::new(5.0, 5.0);
    /// let normalized = v.normalize();
    ///
    /// assert_eq!(normalized.length(), 1.0);
    /// ```
    /// # Panics
    /// Panics if compiled with debug assertions and the vector has a length of zero. Otherwise the
    /// resulting vector will be `(inf, inf)`.
    #[must_use = "Vec2::normalize() does not act in place!"]
    pub fn normalize(&self) -> Vec2 {
        let len = self.length();
        #[cfg(debug_assertions)]
        {
            if len == 0. {
                panic!("Can't normalize a zero-length vector");
            }
        }
        let inverse = 1. / len;
        Vec2 {
            x: self.x * inverse,
            y: self.y * inverse,
        }
    }

    /// Checks whether the vector is normalized or not.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let normalized = Vec2::new(1.0, 0.0);
    /// let not_normalized = Vec2::new(1.5, 0.5);
    ///
    /// assert!(normalized.is_normalized());
    /// assert!(!not_normalized.is_normalized());
    /// ```
    pub fn is_normalized(&self) -> bool {
        let len = self.length();
        len > 1. - f32::EPSILON && len < 1. + f32::EPSILON
    }

    /// Returns the component-wise absolute value for this vector.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let negative = Vec2::new(-1.0, -2.0);
    /// let positive = negative.abs();
    ///
    /// assert_eq!(positive.x, 1.0);
    /// assert_eq!(positive.y, 2.0);
    /// ```
    pub fn abs(&self) -> Vec2 {
        Vec2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns the current vector restricted between two boundaries.
    ///
    /// The lower bound is defined by the `min` parameter, while the upper bound is defined by the
    /// `max` one. If the current vector is in-between, it is left unchanged.
    ///
    /// Clamping is performed component-wise.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::new(1.5, 0.5);
    /// let max = Vec2::new(1.0, 1.0);
    /// let min = Vec2::zero();
    /// let clamped = v.clamp(&min, &max);
    ///
    /// assert_eq!(clamped.x, 1.0);
    /// assert_eq!(clamped.y, 0.5);
    /// ```
    pub fn clamp(&self, min: &Vec2, max: &Vec2) -> Vec2 {
        let x = if self.x < min.x {
            min.x
        } else if self.x > max.x {
            max.x
        } else {
            self.x
        };
        let y = if self.y < min.y {
            min.y
        } else if self.y > max.y {
            max.y
        } else {
            self.y
        };
        Vec2 { x, y }
    }

    /// Performs the dot product between two vectors.
    ///
    /// Given two vectors `a` and `b` the dot product is defined as ‖`a`‖*‖`b`‖ cos`θ`, where ‖`x`‖
    /// represents the euclidean length of the vector `x`, and `θ` represents the angle between the
    /// two vectors.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec2;
    ///
    /// let v = Vec2::new(1.0, 2.0);
    /// let v2 = Vec2::new(4.0, -5.0);
    ///
    /// assert_eq!(Vec2::dot(&v, &v2), -6.0);
    /// ```
    pub fn dot(vec_a: &Vec2, vec_b: &Vec2) -> f32 {
        vec_a.x * vec_b.x + vec_a.y * vec_b.y
    }
}

impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec2[{}, {}]", self.x, self.y)
    }
}

impl Index<u8> for Vec2 {
    type Output = f32;

    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Out of bounds index {} for Vec2", index),
        }
    }
}

impl IndexMut<u8> for Vec2 {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Out of bounds index {} for Vec2", index),
        }
    }
}

overload!((a: ?Vec2) + (b: ?Vec2) -> Vec2 {Vec2{x:a.x+b.x,y:a.y+b.y}});
overload!((a: ?Vec2) - (b: ?Vec2) -> Vec2 {Vec2{x:a.x-b.x,y:a.y-b.y}});
overload!((a: ?Vec2) + (b: f32) -> Vec2 {Vec2{x:a.x+b,y:a.y+b}});
overload!((a: ?Vec2) - (b: f32) -> Vec2 {Vec2{x:a.x-b,y:a.y-b}});
overload!((a: ?Vec2) * (b: f32) -> Vec2 {Vec2{x:a.x*b,y:a.y*b}});
overload!(- (a: ?Vec2) -> Vec2 {Vec2{x:-a.x,y:-a.y}});
overload!((a: &mut Vec2) += (b: ?Vec2){a.x+=b.x;a.y+=b.y;});
overload!((a: &mut Vec2) -= (b: ?Vec2){a.x-=b.x;a.y-=b.y;});
overload!((a: &mut Vec2) += (b: f32){a.x+=b;a.y+=b;});
overload!((a: &mut Vec2) -= (b: f32){a.x-=b;a.y-=b;});
overload!((a: &mut Vec2) *= (b: f32){a.x*=b;a.y*=b;});

/// A vector of three components.
///
/// Vec3 class represents a vector in a 3D space.
///
/// A Vec3 consist of three coordinates, usually called `x`, `y` and `z`.
#[derive(Clone, Copy)]
pub struct Vec3 {
    /// Single precision floating point representing the `x` component of the vector. Indexed by 0.
    pub x: f32,
    /// Single precision floating point representing the `y` component of the vector. Indexed by 1.
    pub y: f32,
    /// Single precision floating point representing the `z` component of the vector. Indexed by 2.
    pub z: f32,
}

/// A Normal can be represented as a Vec3 and behaves almost identically, except during its
/// transformation.
pub type Normal = Vec3;

impl Vec3 {
    /// Constructs a 3D zero vector, a vector in the form `(0.0, 0.0, 0.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::zero();
    ///
    /// assert_eq!(v.x, 0.0);
    /// assert_eq!(v.y, 0.0);
    /// assert_eq!(v.z, 0.0);
    /// ```
    pub fn zero() -> Vec3 {
        Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    /// Constructs a vector with the given `(x, y, z)` components.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(3.5, -2.2, 1.0);
    ///
    /// assert_eq!(v.x, 3.5);
    /// assert_eq!(v.y, -2.2);
    /// assert_eq!(v.z, 1.0);
    /// ```
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Constructs a normalized vector pointing *right*, towards `(1.0, 0.0, 0.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::right();
    ///
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 0.0);
    /// assert_eq!(v.z, 0.0);
    /// ```
    pub fn right() -> Vec3 {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Constructs a normalized vector pointing *up*, towards `(0.0, 1.0, 0.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::up();
    ///
    /// assert_eq!(v.x, 0.0);
    /// assert_eq!(v.y, 1.0);
    /// assert_eq!(v.z, 0.0);
    /// ```
    pub fn up() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    }

    /// Constructs a normalized vector pointing *front* in a Left-Handed system, towards
    /// `(0.0, 0.0, 1.0)`.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::front();
    ///
    /// assert_eq!(v.x, 0.0);
    /// assert_eq!(v.y, 0.0);
    /// assert_eq!(v.z, 1.0);
    /// ```
    pub fn front() -> Vec3 {
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    /// Returns the euclidean length (or magnitude) of the vector.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(2.0, 3.0, 6.0);
    ///
    /// assert_eq!(v.length(), 7.0);
    /// ```
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns the squared euclidean length (or Manhattan length) of the vector.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(2.0, 3.0, 6.0);
    ///
    /// assert_eq!(v.length2(), 49.0);
    /// ```
    pub fn length2(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns the normalized version of the current vector.
    ///
    /// A vector is normalized if its euclidean length is equal to 1.0.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(5.0, 0.0, 0.0);
    /// let normalized = v.normalize();
    ///
    /// assert_eq!(normalized.length(), 1.0);
    /// ```
    /// # Panics
    /// Panics if compiled with debug assertions and the vector has a length of zero. Otherwise the
    /// resulting vector will be `(inf, inf)`.
    #[must_use = "Vec3::normalize() does not act in place!"]
    pub fn normalize(&self) -> Vec3 {
        let len = self.length();
        if cfg!(debug_assertions) && len == 0.0 {
            panic!("Can't normalize a zero-length vector");
        }
        let inverse = 1. / len;
        Vec3 {
            x: self.x * inverse,
            y: self.y * inverse,
            z: self.z * inverse,
        }
    }

    /// Checks whether the vector is normalized or not.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let normalized = Vec3::new(0.0, 1.0, 0.0);
    /// let not_normalized = Vec3::new(1.5, 0.5, 25.0);
    ///
    /// assert!(normalized.is_normalized());
    /// assert!(!not_normalized.is_normalized());
    /// ```
    pub fn is_normalized(&self) -> bool {
        let len = self.length();
        len > 1. - f32::EPSILON && len < 1. + f32::EPSILON
    }

    /// Returns the component-wise absolute value for this vector.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let negative = Vec3::new(-1.0, -2.0, -3.0);
    /// let positive = negative.abs();
    ///
    /// assert_eq!(positive.x, 1.0);
    /// assert_eq!(positive.y, 2.0);
    /// assert_eq!(positive.z, 3.0);
    /// ```
    pub fn abs(&self) -> Vec3 {
        Vec3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Returns the current vector restricted between two boundaries.
    ///
    /// The lower bound is defined by the `min` parameter, while the upper bound is defined by the
    /// `max` one. If the current vector is in-between, it is left unchanged.
    ///
    /// Clamping is performed component-wise.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(1.5, 0.5, -2.5);
    /// let max = Vec3::new(1.0, 1.0, 1.0);
    /// let min = Vec3::zero();
    /// let clamped = v.clamp(&min, &max);
    ///
    /// assert_eq!(clamped.x, 1.0);
    /// assert_eq!(clamped.y, 0.5);
    /// assert_eq!(clamped.z, 0.0);
    /// ```
    pub fn clamp(&self, min: &Vec3, max: &Vec3) -> Vec3 {
        let x = if self.x < min.x {
            min.x
        } else if self.x > max.x {
            max.x
        } else {
            self.x
        };
        let y = if self.y < min.y {
            min.y
        } else if self.y > max.y {
            max.y
        } else {
            self.y
        };
        let z = if self.z < min.z {
            min.z
        } else if self.z > max.z {
            max.z
        } else {
            self.z
        };
        Vec3 { x, y, z }
    }

    /// Reflects a vector around a centre of reflection.
    ///
    /// The centre of reflection is represented by the `centre` parameter and should be normalized.
    ///
    /// If debug assertions are enabled, a warning is issued in case the centre of reflection is
    /// not normalized.
    /// # Example
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(0.5, 0.3, -0.5);
    /// let centre = Vec3::new(0., 0., 1.);
    /// let reflected = v.reflect(&centre);
    ///
    /// assert_eq!(reflected.x, v.x);
    /// assert_eq!(reflected.y, v.y);
    /// assert_eq!(reflected.z, -v.z);
    /// ```
    pub fn reflect(&self, centre: &Normal) -> Vec3 {
        #[cfg(debug_assertions)]
        {
            if !centre.is_normalized() {
                warn!("Reflecting vector around non-normalized centre");
            }
        }
        self - (centre * (2. * Vec3::dot(self, centre)))
    }

    /// Refracts a vector passing through an interface.
    ///
    /// The interface is represented by the `interface` parameter and should be normalized.
    /// The magnitude of refraction is controlled by the `eta` parameter, representing the ration
    /// between the two materials' Index of Refraction η  = η1/η2.
    ///
    /// If debug assertions are enabled, a warning is issued in case the centre of reflection is
    /// not normalized.
    ///
    /// In case of Total Internal Reflection, None is returned.
    /// # Example
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v = Vec3::new(0.3, 0.1, 0.8);
    /// let interface = Vec3::new(0., 0., 1.);
    /// let eta = 1.45;
    /// let refracted = v.refract(&interface, eta);
    ///
    /// assert!(refracted.is_some());
    /// ```
    pub fn refract(&self, interface: &Vec3, eta: f32) -> Option<Vec3> {
        #[cfg(debug_assertions)]
        {
            if !interface.is_normalized() {
                warn!("Refracting vector around non-normalized interface");
            }
        }
        let cosi = Vec3::dot(self, interface); //cos incident
        let sin2i = (0.0_f32).max(1. - cosi * cosi);
        let sin2t = sin2i * eta * eta;
        if sin2t <= 1. {
            let cos2t = 1. - sin2t; //cos2t transmitted
            let cost = cos2t.sqrt();
            let ret = -self * eta;
            Some(ret + (interface * (cosi * eta - cost)))
        } else {
            None
        }
    }

    /// Performs the dot product between two vectors.
    ///
    /// Given two vectors `a` and `b` the dot product `a`·`b` is defined as ‖`a`‖ * ‖`b`‖ * cos`θ`,
    /// where:
    /// - ‖`x`‖ represents the euclidean length of the vector `x`.
    /// - `θ` represents the angle between the two vectors.
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v0 = Vec3::new(1.0, 2.0, 3.0);
    /// let v1 = Vec3::new(4.0, -5.0, 6.0);
    ///
    /// assert_eq!(Vec3::dot(&v0, &v1), 12.0);
    /// ```
    pub fn dot(vec_a: &Vec3, vec_b: &Vec3) -> f32 {
        vec_a.x * vec_b.x + vec_a.y * vec_b.y + vec_a.z * vec_b.z
    }

    /// Performs the cross product between two vectors.
    ///
    /// Given two vectors `a` and `b` the cross product `a` ⨯ `b` represents the vector
    /// perpendicular to both `a` and `b` defined as `n` * ‖`a`‖ * ‖`b`‖ * sin`θ` where:
    ///- ‖`x`‖ represents the euclidean length of the vector `x`.
    ///- `θ` represents the angle between the two vectors.
    ///- `n` represents an unit vector.
    /// The unit vector is necessary as two different results, having opposite directions, exists.
    ///
    /// In this implementation, however, the `n` vector is not supplied, as the cross product is
    /// anti-commutative and `a ⨯ b = -(b ⨯ a)`. The wanted direction can thus be obtained by
    /// changing the order of the parameters.
    ///
    /// In this implementation, a Right-Handed System is assumed. If a Left-Handed one is used,
    /// the functions should be called as `Vec3::cross(&vec_b, &vec_a);`
    /// # Examples
    /// ```
    /// use glaze::geometry::Vec3;
    ///
    /// let v0 = Vec3::new(3.0, -3.0, 1.0);
    /// let v1 = Vec3::new(4.0, 9.0, 2.0);
    /// let cross = Vec3::cross(&v0, &v1);
    ///
    /// assert_eq!(cross.x, -15.0);
    /// assert_eq!(cross.y, -2.0);
    /// assert_eq!(cross.z, 39.0);
    /// ```
    pub fn cross(vec_a: &Vec3, vec_b: &Vec3) -> Vec3 {
        Vec3 {
            x: vec_a.y * vec_b.z - vec_a.z * vec_b.y,
            y: vec_a.z * vec_b.x - vec_a.x * vec_b.z,
            z: vec_a.x * vec_b.y - vec_a.y * vec_b.x,
        }
    }
}

impl Transform3 for Vec3 {
    fn transform(&self, mat: &Matrix4) -> Self {
        let x = mat.m[00] * self.x + mat.m[01] * self.y + mat.m[02] * self.z;
        let y = mat.m[04] * self.x + mat.m[05] * self.y + mat.m[06] * self.z;
        let z = mat.m[08] * self.x + mat.m[09] * self.y + mat.m[10] * self.z;
        Vec3 { x, y, z }
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec3[{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl Index<u8> for Vec3 {
    type Output = f32;

    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Out of bounds index {} for Vec3", index),
        }
    }
}

impl IndexMut<u8> for Vec3 {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Out of bounds index {} for Vec3", index),
        }
    }
}

overload!((a: ?Vec3) + (b: ?Vec3) -> Vec3 {Vec3{x:a.x+b.x,y:a.y+b.y,z:a.z+b.z}});
overload!((a: ?Vec3) - (b: ?Vec3) -> Vec3 {Vec3{x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}});
overload!((a: ?Vec3) + (b: f32) -> Vec3 {Vec3{x:a.x+b,y:a.y+b,z:a.z+b}});
overload!((a: ?Vec3) - (b: f32) -> Vec3 {Vec3{x:a.x-b,y:a.y-b,z:a.z-b}});
overload!((a: ?Vec3) * (b: f32) -> Vec3 {Vec3{x:a.x*b,y:a.y*b,z:a.z*b}});
overload!(- (a: ?Vec3) -> Vec3 {Vec3{x:-a.x,y:-a.y,z:-a.z}});
overload!((a: &mut Vec3) += (b: ?Vec3){a.x+=b.x;a.y+=b.y;a.z+=b.z;});
overload!((a: &mut Vec3) -= (b: ?Vec3){a.x-=b.x;a.y-=b.y;a.z-=b.z;});
overload!((a: &mut Vec3) += (b: f32){a.x+=b;a.y+=b;a.z+=b;});
overload!((a: &mut Vec3) -= (b: f32){a.x-=b;a.y-=b;a.z-=b;});
overload!((a: &mut Vec3) *= (b: f32){a.x*=b;a.y*=b;a.z*=b;});

#[cfg(test)]
mod tests {
    use crate::geometry::matrix::Transform3;
    use crate::geometry::{Matrix4, Vec2, Vec3};
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn vec2_zero_constructor() {
        let v = Vec2::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn vec2_components_constructor() {
        let v = Vec2::new(1.0, 0.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn vec2_right_constructor() {
        let v = Vec2::right();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn vec2_up_constructor() {
        let v = Vec2::up();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
    }

    #[test]
    fn vec2_clone() {
        let v = Vec2::new(-83.27705, 79.29129);
        let v2 = v.clone();
        assert_eq!(v.x, v2.x);
        assert_eq!(v.y, v2.y);
    }

    #[test]
    fn vec2_dot() {
        let v = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, -5.0);
        assert_eq!(Vec2::dot(&v, &v2), -6.0);

        let v3 = Vec2::new(6.0, -1.0);
        let v4 = Vec2::new(4.0, 18.0);
        assert_eq!(Vec2::dot(&v3, &v4), 6.0);

        let v5 = Vec2::new(6.0, -1.0);
        let v6 = Vec2::new(4.0, 24.0);
        assert_eq!(Vec2::dot(&v5, &v6), 0.0);
    }

    #[test]
    fn vec2_length() {
        let v = Vec2::new(-15.0, -2.0);
        let length = v.length();
        assert_approx_eq!(length, 15.132745);

        let v1 = Vec2::new(-3.0, 2.0);
        assert_approx_eq!(v1.length2(), 13.0);

        assert_approx_eq!(v1.length() * v1.length(), v1.length2());

        let v2 = Vec2::zero();
        assert_eq!(v2.length(), 0.0);
    }

    #[test]
    fn vec2_normalize() {
        let v1 = Vec2::new(3.0, 1.0);
        assert_approx_eq!(v1.length(), 3.162277);
        assert!(!v1.is_normalized());

        let v2 = v1.normalize();
        assert_approx_eq!(v2.x, 0.948683);
        assert_approx_eq!(v2.y, 0.316228);
        assert!(v2.is_normalized());
    }

    #[test]
    #[should_panic]
    fn vec2_normalize_zero() {
        let _ = Vec2::zero().normalize();
    }

    #[test]
    fn vec2_abs() {
        let negative = Vec2::new(-1.5, -2.5);
        let positive = Vec2::new(1.5, 2.5);
        let neg_abs = negative.abs();
        let pos_abs = positive.abs();

        assert_eq!(neg_abs.x, 1.5);
        assert_eq!(neg_abs.y, 2.5);
        assert_eq!(pos_abs.x, 1.5);
        assert_eq!(pos_abs.y, 2.5);
    }

    #[test]
    fn vec2_clamp() {
        let sample = Vec2::new(4.9, -5.8);
        let mut clamped;
        let min = Vec2::new(-10.0, -10.0);
        let max = Vec2::new(10.0, 10.0);

        //in range
        clamped = sample.clamp(&min, &max);
        assert_eq!(clamped.x, sample.x);
        assert_eq!(clamped.y, sample.y);
        //min x
        let new_min = Vec2::new(5.0, 5.0);
        clamped = sample.clamp(&new_min, &max);
        assert_eq!(clamped.x, new_min.x);
        assert_eq!(clamped.y, new_min.y);
        //max x
        let new_max = Vec2::new(3.0, -6.0);
        clamped = sample.clamp(&min, &new_max);
        assert_eq!(clamped.x, new_max.x);
        assert_eq!(clamped.y, new_max.y);
    }

    #[test]
    fn vec2_index() {
        let v = Vec2::new(0.5, 0.7);
        assert_eq!(v[0], 0.5);
        assert_eq!(v[1], 0.7);
    }

    #[test]
    fn vec2_index_mut() {
        let mut v = Vec2::new(0.8, 0.4);
        v[0] = 0.4;
        v[1] = 0.9;
        assert_eq!(v.x, 0.4);
        assert_eq!(v.y, 0.9);
    }

    #[test]
    fn vec2_add_vec2() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, 5.5);
        let res = v1 + v2;
        assert_approx_eq!(res.x, 5.0);
        assert_approx_eq!(res.y, 7.5);
    }

    #[test]
    fn vec2_add_float() {
        let v1 = Vec2::new(1.0, 2.65);
        let f = -0.75;
        let res = v1 + f;
        assert_approx_eq!(res.x, 0.25);
        assert_approx_eq!(res.y, 1.9);
    }

    #[test]
    fn vec2_add_assign_vec2() {
        let mut v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(4.0, 5.5);
        v1 += v2;
        assert_approx_eq!(v1.x, 5.0);
        assert_approx_eq!(v1.y, 7.5);
    }

    #[test]
    fn vec2_add_assign_float() {
        let mut v1 = Vec2::new(3.3, 1.2);
        let f = 0.5;
        v1 += f;
        assert_approx_eq!(v1.x, 3.8);
        assert_approx_eq!(v1.y, 1.7);
    }

    #[test]
    fn vec2_sub_vec2() {
        let v1 = Vec2::new(7.0, 6.5);
        let v2 = Vec2::new(7.0, 5.3);
        let res = v1 - v2;
        assert_approx_eq!(res.x, 0.0);
        assert_approx_eq!(res.y, 1.2);
    }

    #[test]
    fn vec2_sub_float() {
        let v1 = Vec2::new(3.3, 1.2);
        let f = 0.5;
        let res = v1 - f;
        assert_approx_eq!(res.x, 2.8);
        assert_approx_eq!(res.y, 0.7);
    }

    #[test]
    fn vec2_sub_assign_vec2() {
        let mut v1 = Vec2::new(5.0, 3.0);
        let v2 = Vec2::new(7.0, 3.3);
        v1 -= v2;
        assert_approx_eq!(v1.x, -2.0);
        assert_approx_eq!(v1.y, -0.3);
    }

    #[test]
    fn vec2_sub_assign_float() {
        let mut v1 = Vec2::new(3.3, 1.2);
        let f = 0.5;
        v1 -= f;
        assert_approx_eq!(v1.x, 2.8);
        assert_approx_eq!(v1.y, 0.7);
    }

    #[test]
    fn vec2_mul_float() {
        let v1 = Vec2::new(3.3, 1.2);
        let f = 7.33;
        let res = v1 * f;
        assert_approx_eq!(res.x, 24.189);
        assert_approx_eq!(res.y, 8.796);
    }

    #[test]
    fn vec2_mul_assign_float() {
        let mut v1 = Vec2::new(3.3, 1.2);
        let f = -53.477;
        v1 *= f;
        assert_approx_eq!(v1.x, -176.4741);
        assert_approx_eq!(v1.y, -64.1724);
    }

    #[test]
    fn vec2_neg() {
        let v1 = Vec2::new(6.0, 8.5);
        let res = -v1;
        assert_eq!(res.x, -v1.x);
        assert_eq!(res.y, -v1.y);
    }

    #[test]
    fn vec2_display() {
        let v = Vec2::new(0.1, 1.2);
        let str = format!("{}", &v);
        assert_eq!(str, "Vec2[0.1, 1.2]");
    }

    #[test]
    fn vec3_zero_constructor() {
        let v = Vec3::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vec3_components_constructor() {
        let v = Vec3::new(1.0, 0.0, -1.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, -1.0);
    }

    #[test]
    fn vec3_right_constructor() {
        let v = Vec3::right();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vec3_up_constructor() {
        let v = Vec3::up();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vec3_front_constructor() {
        let v = Vec3::front();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 1.0);
    }

    #[test]
    fn vec3_clone() {
        let v = Vec3::new(-83.27705, 79.29129, -51.32018);
        let v2 = v;

        assert_eq!(v.x, v2.x);
        assert_eq!(v.y, v2.y);
        assert_eq!(v.z, v2.z);
    }

    #[test]
    fn vec3_dot() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, -5.0, 6.0);
        assert_eq!(Vec3::dot(&v, &v2), 12.0);

        let v3 = Vec3::new(6.0, -1.0, 3.0);
        let v4 = Vec3::new(4.0, 18.0, -2.0);
        assert_eq!(Vec3::dot(&v3, &v4), 0.0);
    }

    #[test]
    fn vec3_cross() {
        let v0 = Vec3::new(3.0, -3.0, 1.0);
        let v1 = Vec3::new(4.0, 9.0, 2.0);
        let outv = Vec3::cross(&v0, &v1);

        assert_eq!(outv.x, -15.0);
        assert_eq!(outv.y, -2.0);
        assert_eq!(outv.z, 39.0);
    }

    #[test]
    fn vec3_length() {
        let v = Vec3::new(-15.0, -2.0, 39.0);
        let length = v.length();
        assert_approx_eq!(length, 41.833);

        let v1 = Vec3::new(-32.0, -53.0, 23.0);
        assert_eq!(v1.length2(), 4362.0);
        assert_approx_eq!(v1.length() * v1.length(), v1.length2());

        let v2 = Vec3::zero();
        assert_eq!(v2.length(), 0.0);
    }

    #[test]
    fn vec3_normalize() {
        let v1 = Vec3::new(3.0, 1.0, 2.0);
        assert_approx_eq!(v1.length(), 3.741657);
        assert!(!v1.is_normalized());

        let normalized = v1.normalize();
        assert_approx_eq!(normalized.x, 0.801783);
        assert_approx_eq!(normalized.y, 0.267261);
        assert_approx_eq!(normalized.z, 0.534522);
        assert!(normalized.is_normalized());
    }

    #[test]
    #[should_panic]
    fn vec3_normalize_zero() {
        let _ = Vec3::zero().normalize();
    }

    #[test]
    fn vec3_abs() {
        let negative = Vec3::new(-1.5, -2.5, -3.5);
        let positive = Vec3::new(1.5, 2.5, 3.5);
        let neg_abs = negative.abs();
        let pos_abs = positive.abs();

        assert_eq!(neg_abs.x, 1.5);
        assert_eq!(neg_abs.y, 2.5);
        assert_eq!(neg_abs.z, 3.5);
        assert_eq!(pos_abs.x, 1.5);
        assert_eq!(pos_abs.y, 2.5);
        assert_eq!(pos_abs.z, 3.5);
    }

    #[test]
    fn vec3_clamp() {
        let sample = Vec3::new(4.9, -5.8, 3.6);
        let mut clamped;
        let min = Vec3::new(-10.0, -10.0, -10.0);
        let max = Vec3::new(10.0, 10.0, 10.0);
        //in range
        clamped = sample.clamp(&min, &max);
        assert_eq!(clamped.x, sample.x);
        assert_eq!(clamped.y, sample.y);
        assert_eq!(clamped.z, sample.z);
        //min
        let new_min = Vec3::new(5.0, 5.0, 5.0);
        clamped = sample.clamp(&new_min, &max);
        assert_eq!(clamped.x, new_min.x);
        assert_eq!(clamped.y, new_min.y);
        assert_eq!(clamped.z, new_min.z);
        //max
        let new_max = Vec3::new(4.0, -6.0, 3.0);
        clamped = sample.clamp(&min, &new_max);
        assert_eq!(clamped.x, new_max.x);
        assert_eq!(clamped.y, new_max.y);
        assert_eq!(clamped.z, new_max.z);
    }

    #[test]
    fn vec3_reflect() {
        let sample = Vec3::new(0.5, 0.3, -0.5);
        let centre = Vec3::new(0.0, 0.0, 1.0);
        let reflected = sample.reflect(&centre);
        assert_eq!(reflected.x, sample.x);
        assert_eq!(reflected.y, sample.y);
        assert_eq!(reflected.z, -sample.z);
    }

    #[test]
    fn vec3_refract_tir() {
        let v = Vec3::new(0.5, 0.3, -0.5);
        let interface = Vec3::new(0.0, 0.0, 1.0);
        let eta = 1.45;
        let refracted = v.refract(&interface, eta);
        assert!(refracted.is_none());
    }

    #[test]
    fn vec3_refract_no_tir() {
        let v = Vec3::new(0.3, 0.1, 0.8);
        let interface = Vec3::new(0.0, 0.0, 1.0);
        let eta = 1.45;
        let refracted = v.refract(&interface, eta).unwrap();
        assert_approx_eq!(refracted.x, -0.435);
        assert_approx_eq!(refracted.y, -0.145);
        assert_approx_eq!(refracted.z, -0.493051767);
    }

    #[test]
    fn vec3_index() {
        let v = Vec3::new(0.3, 0.1, 0.8);
        assert_eq!(v[0], 0.3);
        assert_eq!(v[1], 0.1);
        assert_eq!(v[2], 0.8);
    }

    #[test]
    fn vec3_index_mut() {
        let mut v = Vec3::new(0.3, 0.1, 0.8);
        v[0] = 0.1;
        v[1] = 0.8;
        v[2] = 0.3;
        assert_eq!(v.x, 0.1);
        assert_eq!(v.y, 0.8);
        assert_eq!(v.z, 0.3);
    }

    #[test]
    fn vec3_add_vec3() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.5, -3.0);
        let res = v1 + v2;
        assert_approx_eq!(res.x, 5.0);
        assert_approx_eq!(res.y, 7.50);
        assert_approx_eq!(res.z, 0.0);
    }

    #[test]
    fn vec3_add_float() {
        let v1 = Vec3::new(1.0, 2.0, -3.0);
        let f = -0.75;
        let res = v1 + f;
        assert_approx_eq!(res.x, 0.25);
        assert_approx_eq!(res.y, 1.25);
        assert_approx_eq!(res.z, -3.75);
    }

    #[test]
    fn vec3_add_assign_vec3() {
        let mut v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.5, -3.0);
        v1 += v2;
        assert_approx_eq!(v1.x, 5.0);
        assert_approx_eq!(v1.y, 7.5);
        assert_approx_eq!(v1.z, 0.0);
    }

    #[test]
    fn vec3_add_assign_float() {
        let mut v1 = Vec3::new(3.3, 1.2, -1.5);
        let f = 0.5;
        v1 += f;
        assert_approx_eq!(v1.x, 3.8);
        assert_approx_eq!(v1.y, 1.7);
        assert_approx_eq!(v1.z, -1.0);
    }

    #[test]
    fn vec3_sub_vec3() {
        let v1 = Vec3::new(7.0, 6.5, -5.0);
        let v2 = Vec3::new(7.0, 5.3, -4.0);

        let res = v1 - v2;
        assert_approx_eq!(res.x, 0.0);
        assert_approx_eq!(res.y, 1.2);
        assert_approx_eq!(res.z, -1.0);
    }

    #[test]
    fn vec3_sub_float() {
        let v1 = Vec3::new(3.3, 1.2, -1.5);
        let f = 0.5;
        let res = v1 - f;
        assert_approx_eq!(res.x, 2.8);
        assert_approx_eq!(res.y, 0.7);
        assert_approx_eq!(res.z, -2.0);
    }

    #[test]
    fn vec3_sub_assign_vec3() {
        let mut v1 = Vec3::new(5.0, 3.0, 1.0);
        let v2 = Vec3::new(7.0, 3.3, 4.0);
        v1 -= v2;
        assert_approx_eq!(v1.x, -2.0);
        assert_approx_eq!(v1.y, -0.3);
        assert_approx_eq!(v1.z, -3.0);
    }

    #[test]
    fn vec3_sub_assign_float() {
        let mut v1 = Vec3::new(3.3, 1.2, -1.5);
        let f = 0.5;
        v1 -= f;
        assert_approx_eq!(v1.x, 2.8);
        assert_approx_eq!(v1.y, 0.7);
        assert_approx_eq!(v1.z, -2.0);
    }

    #[test]
    fn vec3_mul_float() {
        let v1 = Vec3::new(3.3, 1.2, -1.5);
        let f = 7.33;
        let res = v1 * f;
        assert_approx_eq!(res.x, 24.189);
        assert_approx_eq!(res.y, 8.796);
        assert_approx_eq!(res.z, -10.995);
    }

    #[test]
    fn vec3_mul_assign_float() {
        let mut v1 = Vec3::new(3.3, 1.2, -1.5);
        let f = -53.477;
        v1 *= f;
        assert_approx_eq!(v1.x, -176.4741);
        assert_approx_eq!(v1.y, -64.1724);
        assert_approx_eq!(v1.z, 80.2155);
    }

    #[test]
    fn vec3_neg() {
        let v1 = Vec3::new(6.0, 8.5, -3.76);
        let res = -v1;

        assert_approx_eq!(res.x, -6.0);
        assert_approx_eq!(res.y, -8.5);
        assert_approx_eq!(res.z, 3.76);
    }

    #[test]
    fn vec3_display() {
        let v = Vec3::new(0.1, 1.2, 2.3);
        let str = format!("{}", &v);
        assert_eq!(str, "Vec3[0.1, 1.2, 2.3]");
    }

    #[test]
    //scale should alter the vector
    fn vec3_transform_scale() {
        //scale
        let v = Vec3::new(1.0, 1.0, 1.0);
        let m = Matrix4::scale(&Vec3::new(3.0, 3.0, 3.0));
        let transformed = v.transform(&m);
        assert_eq!(transformed.x, 3.0);
        assert_eq!(transformed.y, 3.0);
        assert_eq!(transformed.z, 3.0);
    }

    #[test]
    //translation should leave vector unaffected
    fn vec3_transform_translation() {
        let v = Vec3::new(0.0, 1.0, 1.0);
        let m = Matrix4::translation(&Vec3::new(0.0, -1.0, 2.5));
        let transformed = v.transform(&m);
        assert_eq!(transformed.x, 0.0);
        assert_eq!(transformed.y, 1.0);
        assert_eq!(transformed.z, 1.0);
    }
}
