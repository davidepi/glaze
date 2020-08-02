use crate::geometry::point::Point3;
use crate::geometry::vec::Vec3;
use crate::utility::float_eq;
use overload::overload;
use std::ops;

//indices for an easier indexing (don't want to use multidimensional array as I don't know the
//memory layout and this will be a really low level class)
const M00: usize = 00;
const M01: usize = 01;
const M02: usize = 02;
const M03: usize = 03;
const M10: usize = 04;
const M11: usize = 05;
const M12: usize = 06;
const M13: usize = 07;
const M20: usize = 08;
const M21: usize = 09;
const M22: usize = 10;
const M23: usize = 11;
const M30: usize = 12;
const M31: usize = 13;
const M32: usize = 14;
const M33: usize = 15;

/// A 4x4 Matrix.
///
/// The Matrix4 class represents a 4x4 transformation matrix in 3D space. A 4x4 Matrix is usually
/// used to perform transformations such as scaling, rotations or translations of a model.
/// The matrix is 4x4 instead of 3x3 because some transformations requires an homogeneus space
/// instead of a cartesian one.
pub struct Matrix4 {
    pub(super) m: [f32; 16],
}

impl Matrix4 {
    /// Creates a zero matrix, a matrix where every value is `(0.0)`.
    pub fn zero() -> Matrix4 {
        Matrix4 { m: [0.0; 16] }
    }

    /// Creates an identity matrix.
    ///
    /// The identity matrix is filled with `1.0` values in the diagonal and `0.0` values everywhere
    /// else.
    pub fn identity() -> Matrix4 {
        Matrix4 {
            m: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ],
        }
    }

    /// Creates a matrix with the given values.
    ///
    /// This is intended for debug purposes only, and thus only an array is accepted, as opposed to
    /// a slice.
    #[cfg(test)]
    pub(crate) fn new(values: &[f32; 16]) -> Matrix4 {
        Matrix4 { m: *values }
    }

    /// Creates a transformation matrix representing a translation.
    ///
    /// The input vector `dir` defines the magnitude and direction of the translation.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Matrix4, Point3, Vec3};
    ///
    /// let dir = Vec3::new(0.0, 1.0, 0.0);
    /// let translate = Matrix4::translation(&dir);
    /// let original = Point3::zero();
    /// let transformed = original.transform(&translate);
    ///
    /// assert_eq!(transformed.x, 0.0);
    /// assert_eq!(transformed.y, 1.0);
    /// assert_eq!(transformed.z, 0.0);
    /// ```
    pub fn translation(dir: &Vec3) -> Matrix4 {
        Matrix4 {
            m: [
                1.000, 0.000, 0.000, dir.x, //
                0.000, 1.000, 0.000, dir.y, //
                0.000, 0.000, 1.000, dir.z, //
                0.000, 0.000, 0.000, 1.000,
            ],
        }
    }

    /// Extracts the translation component from the matrix.
    ///
    /// If the matrix was generated by a composition of scales, rotations and translations, this
    /// method extracts the translation component.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Matrix4, Vec3};
    ///
    /// let dir = Vec3::new(1.0, 2.0, -3.0);
    /// let translate = Matrix4::translation(&dir);
    /// let extracted = translate.get_translation();
    ///
    /// assert_eq!(extracted.x, dir.x);
    /// assert_eq!(extracted.y, dir.y);
    /// assert_eq!(extracted.z, dir.z);
    /// ```
    pub fn get_translation(&self) -> Vec3 {
        Vec3 {
            x: self.m[M03],
            y: self.m[M13],
            z: self.m[M23],
        }
    }

    /// Creates a transformation matrix representing a scaling.
    ///
    /// The input vector `magnitude` defines the magnitude of the scaling and its component should
    /// be strictly positives.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Matrix4, Point3, Vec3};
    ///
    /// let magnitude = Vec3::new(5.0, 1.0, 1.0);
    /// let scale = Matrix4::scale(&magnitude);
    /// let original = Point3::new(1.0, 1.0, 1.0);
    /// let transformed = original.transform(&scale);
    ///
    /// assert_eq!(transformed.x, 5.0);
    /// assert_eq!(transformed.y, 1.0);
    /// assert_eq!(transformed.z, 1.0);
    /// ```
    pub fn scale(magnitude: &Vec3) -> Matrix4 {
        //otherwise fmt refactors the matrix because it's too large with `magnitude`
        //but I want magnitude as the name shown in the signature
        let value = magnitude;
        Matrix4 {
            m: [
                value.x, 0.00000, 0.00000, 0.00000, //
                0.00000, value.y, 0.00000, 0.00000, //
                0.00000, 0.00000, value.z, 0.00000, //
                0.00000, 0.00000, 0.00000, 1.00000, //
            ],
        }
    }

    /// Extracts the scale component from the matrix.
    ///
    /// If the matrix was generated by a composition of scales, rotations and translations, this
    /// method extracts the scale component.
    /// # Examples
    /// ```
    /// use glaze::geometry::{Matrix4, Vec3};
    ///
    /// let magnitude = Vec3::new(1.0, 2.0, 3.0);
    /// let translate = Matrix4::scale(&magnitude);
    /// let extracted = translate.get_scale();
    ///
    /// assert_eq!(extracted.x, magnitude.x);
    /// assert_eq!(extracted.y, magnitude.y);
    /// assert_eq!(extracted.z, magnitude.z);
    /// ```
    pub fn get_scale(&self) -> Vec3 {
        let x = Vec3::new(self.m[M00], self.m[M10], self.m[M20]).length();
        let y = Vec3::new(self.m[M01], self.m[M11], self.m[M21]).length();
        let z = Vec3::new(self.m[M02], self.m[M12], self.m[M22]).length();
        Vec3 { x, y, z }
    }

    /// Creates a rotation matrix around the `x` axis.
    ///
    /// Sets this matrix to a transformation matrix responsible of the rotation around the `x` axis.
    /// This action is also called roll. The input float defines the angle of rotation in *radians*.
    /// # Examples
    /// ```
    /// use assert_approx_eq::assert_approx_eq;
    /// use glaze::geometry::{Matrix4, Point3, Vec3};
    ///
    /// let roll = 90.0_f32.to_radians();
    /// let rot = Matrix4::rotate_x(roll);
    /// let original = Point3::new(0.0, 1.0, 0.0);
    /// let transformed = original.transform(&rot);
    ///
    /// assert_approx_eq!(transformed.x, 0.0);
    /// assert_approx_eq!(transformed.y, 0.0);
    /// assert_approx_eq!(transformed.z, 1.0);
    /// ```
    pub fn rotate_x(roll: f32) -> Matrix4 {
        let sint = roll.sin();
        let cost = roll.cos();
        Matrix4 {
            m: [
                1.00, 0.00, 0.00, 0.00, //
                0.00, cost, -sint, 0.00, //
                0.00, sint, cost, 0.00, //
                0.00, 0.00, 0.00, 1.00, //
            ],
        }
    }

    /// Creates a rotation matrix around the `y` axis.
    ///
    /// Sets this matrix to a transformation matrix responsible of the rotation around the `y` axis.
    /// This action is also called pitch. The input float defines the angle of rotation in
    /// *radians*.
    /// # Examples
    /// ```
    /// use assert_approx_eq::assert_approx_eq;
    /// use glaze::geometry::{Matrix4, Point3, Vec3};
    ///
    /// let pitch = 90.0_f32.to_radians();
    /// let rot = Matrix4::rotate_y(pitch);
    /// let original = Point3::new(1.0, 0.0, 0.0);
    /// let transformed = original.transform(&rot);
    ///
    /// assert_approx_eq!(transformed.x, 0.0);
    /// assert_approx_eq!(transformed.y, 0.0);
    /// assert_approx_eq!(transformed.z, -1.0);
    /// ```
    pub fn rotate_y(pitch: f32) -> Matrix4 {
        let sint = pitch.sin();
        let cost = pitch.cos();
        Matrix4 {
            m: [
                cost, 0.00, sint, 0.00, //
                0.00, 1.00, 0.00, 0.00, //
                -sint, 0.00, cost, 0.00, //
                0.00, 0.00, 0.00, 1.00, //
            ],
        }
    }

    /// Creates a rotation matrix around the `z` axis.
    ///
    /// Sets this matrix to a transformation matrix responsible of the rotation around the `z` axis.
    /// This action is also called yaw. The input float defines the angle of rotation in *radians*.
    /// # Examples
    /// ```
    /// use assert_approx_eq::assert_approx_eq;
    /// use glaze::geometry::{Matrix4, Point3, Vec3};
    ///
    /// let yaw = 90.0_f32.to_radians();
    /// let rot = Matrix4::rotate_z(yaw);
    /// let original = Point3::new(1.0, 0.0, 0.0);
    /// let transformed = original.transform(&rot);
    ///
    /// assert_approx_eq!(transformed.x, 0.0);
    /// assert_approx_eq!(transformed.y, 1.0);
    /// assert_approx_eq!(transformed.z, 0.0);
    /// ```
    pub fn rotate_z(yaw: f32) -> Matrix4 {
        let sint = yaw.sin();
        let cost = yaw.cos();
        Matrix4 {
            m: [
                cost, -sint, 0.00, 0.00, //
                sint, cost, 0.00, 0.00, //
                0.00, 0.00, 1.00, 0.00, //
                0.00, 0.00, 0.00, 1.00, //
            ],
        }
    }

    /// Creates a look-at matrix.
    ///
    /// Sets this matrix to a transformation Look-At matrix in a Left Handed System.
    /// This matrix can then be used to transform coordinates from camera-space to world-space.
    /// In order to obtain a world-space to camera-space transformation, the matrix can be easily
    /// inverted.
    ///
    /// This method takes the following parameters to represent a camera:
    /// - `pos`: The position of the camera in world-space coordinates.
    /// - `target`: The point where the camera is looking at, in world-space coordinates.
    /// - `up`: A unit-vector indicating which direction is the upper side of the camera in
    /// world-space coordinate (used to achieve effects such as camera orientation and angles).
    ///
    /// In camera space, instead, the camera is centered in `(0.0, 0.0, 0.0)`, points towards +z
    /// and has an `up` vector of `(0.0, 1.0, 0.0)`.
    ///
    /// # Panics
    /// If debug assertions are enabled, this method panics if the vector input `up` is not
    /// normalized.
    pub fn camera_to_world(pos: &Point3, target: &Point3, up: &Vec3) -> Matrix4 {
        #[cfg(debug_assertions)]
        {
            if !up.is_normalized() {
                panic!("Creating a camera to world transformation with a non-normalized vector");
            }
        }
        let dir = (target - pos).normalize();
        let right = Vec3::cross(&up, &dir).normalize();
        let newup = Vec3::cross(&dir, &right).normalize();
        Matrix4 {
            m: [
                right.x, newup.x, dir.x, pos.x, //
                right.y, newup.y, dir.y, pos.y, //
                right.z, newup.z, dir.z, pos.z, //
                0.00000, 0.00000, 0.000, 1.000, //
            ],
        }
    }

    /// Returns a new matrix that is the transpose of the current matrix.
    pub fn transpose(&self) -> Matrix4 {
        Matrix4 {
            m: [
                self.m[M00],
                self.m[M10],
                self.m[M20],
                self.m[M30],
                self.m[M01],
                self.m[M11],
                self.m[M21],
                self.m[M31],
                self.m[M02],
                self.m[M12],
                self.m[M22],
                self.m[M32],
                self.m[M03],
                self.m[M13],
                self.m[M23],
                self.m[M33],
            ],
        }
    }

    /// Returns a new matrix that is the inverse of the current matrix.
    /// If the original matrix is not invertible, `None` is returned.
    pub fn inverse(&self) -> Option<Matrix4> {
        let inv00 = self.m[M11] * self.m[M22] * self.m[M33]
            - self.m[M11] * self.m[M23] * self.m[M32]
            - self.m[M21] * self.m[M12] * self.m[M33]
            + self.m[M21] * self.m[M13] * self.m[M32]
            + self.m[M31] * self.m[M12] * self.m[M23]
            - self.m[M31] * self.m[M13] * self.m[M22];

        let inv04 = -self.m[M10] * self.m[M22] * self.m[M33]
            + self.m[M10] * self.m[M23] * self.m[M32]
            + self.m[M20] * self.m[M12] * self.m[M33]
            - self.m[M20] * self.m[M13] * self.m[M32]
            - self.m[M30] * self.m[M12] * self.m[M23]
            + self.m[M30] * self.m[M13] * self.m[M22];

        let inv08 = self.m[M10] * self.m[M21] * self.m[M33]
            - self.m[M10] * self.m[M23] * self.m[M31]
            - self.m[M20] * self.m[M11] * self.m[M33]
            + self.m[M20] * self.m[M13] * self.m[M31]
            + self.m[M30] * self.m[M11] * self.m[M23]
            - self.m[M30] * self.m[M13] * self.m[M21];

        let inv12 = -self.m[M10] * self.m[M21] * self.m[M32]
            + self.m[M10] * self.m[M22] * self.m[M31]
            + self.m[M20] * self.m[M11] * self.m[M32]
            - self.m[M20] * self.m[M12] * self.m[M31]
            - self.m[M30] * self.m[M11] * self.m[M22]
            + self.m[M30] * self.m[M12] * self.m[M21];

        let det =
            self.m[M00] * inv00 + self.m[M01] * inv04 + self.m[M02] * inv08 + self.m[M03] * inv12;

        if float_eq(det, 0.0, 1E-5) {
            return None;
        }

        let inv01 = -self.m[M01] * self.m[M22] * self.m[M33]
            + self.m[M01] * self.m[M23] * self.m[M32]
            + self.m[M21] * self.m[M02] * self.m[M33]
            - self.m[M21] * self.m[M03] * self.m[M32]
            - self.m[M31] * self.m[M02] * self.m[M23]
            + self.m[M31] * self.m[M03] * self.m[M22];

        let inv05 = self.m[M00] * self.m[M22] * self.m[M33]
            - self.m[M00] * self.m[M23] * self.m[M32]
            - self.m[M20] * self.m[M02] * self.m[M33]
            + self.m[M20] * self.m[M03] * self.m[M32]
            + self.m[M30] * self.m[M02] * self.m[M23]
            - self.m[M30] * self.m[M03] * self.m[M22];

        let inv09 = -self.m[M00] * self.m[M21] * self.m[M33]
            + self.m[M00] * self.m[M23] * self.m[M31]
            + self.m[M20] * self.m[M01] * self.m[M33]
            - self.m[M20] * self.m[M03] * self.m[M31]
            - self.m[M30] * self.m[M01] * self.m[M23]
            + self.m[M30] * self.m[M03] * self.m[M21];

        let inv13 = self.m[M00] * self.m[M21] * self.m[M32]
            - self.m[M00] * self.m[M22] * self.m[M31]
            - self.m[M20] * self.m[M01] * self.m[M32]
            + self.m[M20] * self.m[M02] * self.m[M31]
            + self.m[M30] * self.m[M01] * self.m[M22]
            - self.m[M30] * self.m[M02] * self.m[M21];

        let inv02 = self.m[M01] * self.m[M12] * self.m[M33]
            - self.m[M01] * self.m[M13] * self.m[M32]
            - self.m[M11] * self.m[M02] * self.m[M33]
            + self.m[M11] * self.m[M03] * self.m[M32]
            + self.m[M31] * self.m[M02] * self.m[M13]
            - self.m[M31] * self.m[M03] * self.m[M12];

        let inv06 = -self.m[M00] * self.m[M12] * self.m[M33]
            + self.m[M00] * self.m[M13] * self.m[M32]
            + self.m[M10] * self.m[M02] * self.m[M33]
            - self.m[M10] * self.m[M03] * self.m[M32]
            - self.m[M30] * self.m[M02] * self.m[M13]
            + self.m[M30] * self.m[M03] * self.m[M12];

        let inv10 = self.m[M00] * self.m[M11] * self.m[M33]
            - self.m[M00] * self.m[M13] * self.m[M31]
            - self.m[M10] * self.m[M01] * self.m[M33]
            + self.m[M10] * self.m[M03] * self.m[M31]
            + self.m[M30] * self.m[M01] * self.m[M13]
            - self.m[M30] * self.m[M03] * self.m[M11];

        let inv14 = -self.m[M00] * self.m[M11] * self.m[M32]
            + self.m[M00] * self.m[M12] * self.m[M31]
            + self.m[M10] * self.m[M01] * self.m[M32]
            - self.m[M10] * self.m[M02] * self.m[M31]
            - self.m[M30] * self.m[M01] * self.m[M12]
            + self.m[M30] * self.m[M02] * self.m[M11];

        let inv03 = -self.m[M01] * self.m[M12] * self.m[M23]
            + self.m[M01] * self.m[M13] * self.m[M22]
            + self.m[M11] * self.m[M02] * self.m[M23]
            - self.m[M11] * self.m[M03] * self.m[M22]
            - self.m[M21] * self.m[M02] * self.m[M13]
            + self.m[M21] * self.m[M03] * self.m[M12];

        let inv07 = self.m[M00] * self.m[M12] * self.m[M23]
            - self.m[M00] * self.m[M13] * self.m[M22]
            - self.m[M10] * self.m[M02] * self.m[M23]
            + self.m[M10] * self.m[M03] * self.m[M22]
            + self.m[M20] * self.m[M02] * self.m[M13]
            - self.m[M20] * self.m[M03] * self.m[M12];

        let inv11 = -self.m[M00] * self.m[M11] * self.m[M23]
            + self.m[M00] * self.m[M13] * self.m[M21]
            + self.m[M10] * self.m[M01] * self.m[M23]
            - self.m[M10] * self.m[M03] * self.m[M21]
            - self.m[M20] * self.m[M01] * self.m[M13]
            + self.m[M20] * self.m[M03] * self.m[M11];

        let inv15 = self.m[M00] * self.m[M11] * self.m[M22]
            - self.m[M00] * self.m[M12] * self.m[M21]
            - self.m[M10] * self.m[M01] * self.m[M22]
            + self.m[M10] * self.m[M02] * self.m[M21]
            + self.m[M20] * self.m[M01] * self.m[M12]
            - self.m[M20] * self.m[M02] * self.m[M11];

        let invdet = 1.0 / det;
        Some(Matrix4 {
            m: [
                inv00 * invdet,
                inv01 * invdet,
                inv02 * invdet,
                inv03 * invdet,
                inv04 * invdet,
                inv05 * invdet,
                inv06 * invdet,
                inv07 * invdet,
                inv08 * invdet,
                inv09 * invdet,
                inv10 * invdet,
                inv11 * invdet,
                inv12 * invdet,
                inv13 * invdet,
                inv14 * invdet,
                inv15 * invdet,
            ],
        })
    }
}

overload!((a: ?Matrix4) + (b: ?Matrix4) -> Matrix4 {
Matrix4{
    m: [
    a.m[M00]+b.m[M00], a.m[M01]+b.m[M01], a.m[M02]+b.m[M02], a.m[M03]+b.m[M03],
    a.m[M10]+b.m[M10], a.m[M11]+b.m[M11], a.m[M12]+b.m[M12], a.m[M13]+b.m[M13],
    a.m[M20]+b.m[M20], a.m[M21]+b.m[M21], a.m[M22]+b.m[M22], a.m[M23]+b.m[M23],
    a.m[M30]+b.m[M30], a.m[M31]+b.m[M31], a.m[M32]+b.m[M32], a.m[M33]+b.m[M33],
    ]}}
    );

overload!((a: ?Matrix4) - (b: ?Matrix4) -> Matrix4 {
Matrix4{
    m: [
    a.m[M00]-b.m[M00], a.m[M01]-b.m[M01], a.m[M02]-b.m[M02], a.m[M03]-b.m[M03],
    a.m[M10]-b.m[M10], a.m[M11]-b.m[M11], a.m[M12]-b.m[M12], a.m[M13]-b.m[M13],
    a.m[M20]-b.m[M20], a.m[M21]-b.m[M21], a.m[M22]-b.m[M22], a.m[M23]-b.m[M23],
    a.m[M30]-b.m[M30], a.m[M31]-b.m[M31], a.m[M32]-b.m[M32], a.m[M33]-b.m[M33],
    ]}}
    );

overload!((a: ?Matrix4) * (b: ?Matrix4) -> Matrix4 {
 Matrix4 {
 m: [
    (a.m[M00]*b.m[M00])+(a.m[M01]*b.m[M10])+(a.m[M02]*b.m[M20])+(a.m[M03]*b.m[M30]),
    (a.m[M00]*b.m[M01])+(a.m[M01]*b.m[M11])+(a.m[M02]*b.m[M21])+(a.m[M03]*b.m[M31]),
    (a.m[M00]*b.m[M02])+(a.m[M01]*b.m[M12])+(a.m[M02]*b.m[M22])+(a.m[M03]*b.m[M32]),
    (a.m[M00]*b.m[M03])+(a.m[M01]*b.m[M13])+(a.m[M02]*b.m[M23])+(a.m[M03]*b.m[M33]),
    (a.m[M10]*b.m[M00])+(a.m[M11]*b.m[M10])+(a.m[M12]*b.m[M20])+(a.m[M13]*b.m[M30]),
    (a.m[M10]*b.m[M01])+(a.m[M11]*b.m[M11])+(a.m[M12]*b.m[M21])+(a.m[M13]*b.m[M31]),
    (a.m[M10]*b.m[M02])+(a.m[M11]*b.m[M12])+(a.m[M12]*b.m[M22])+(a.m[M13]*b.m[M32]),
    (a.m[M10]*b.m[M03])+(a.m[M11]*b.m[M13])+(a.m[M12]*b.m[M23])+(a.m[M13]*b.m[M33]),
    (a.m[M20]*b.m[M00])+(a.m[M21]*b.m[M10])+(a.m[M22]*b.m[M20])+(a.m[M23]*b.m[M30]),
    (a.m[M20]*b.m[M01])+(a.m[M21]*b.m[M11])+(a.m[M22]*b.m[M21])+(a.m[M23]*b.m[M31]),
    (a.m[M20]*b.m[M02])+(a.m[M21]*b.m[M12])+(a.m[M22]*b.m[M22])+(a.m[M23]*b.m[M32]),
    (a.m[M20]*b.m[M03])+(a.m[M21]*b.m[M13])+(a.m[M22]*b.m[M23])+(a.m[M23]*b.m[M33]),
    (a.m[M30]*b.m[M00])+(a.m[M31]*b.m[M10])+(a.m[M32]*b.m[M20])+(a.m[M33]*b.m[M30]),
    (a.m[M30]*b.m[M01])+(a.m[M31]*b.m[M11])+(a.m[M32]*b.m[M21])+(a.m[M33]*b.m[M31]),
    (a.m[M30]*b.m[M02])+(a.m[M31]*b.m[M12])+(a.m[M32]*b.m[M22])+(a.m[M33]*b.m[M32]),
    (a.m[M30]*b.m[M03])+(a.m[M31]*b.m[M13])+(a.m[M32]*b.m[M23])+(a.m[M33]*b.m[M33]),
    ]}}
);

overload!((a: &mut Matrix4) += (b: ?Matrix4) {
a.m[M00]+=b.m[M00]; a.m[M01]+=b.m[M01]; a.m[M02]+=b.m[M02]; a.m[M03]+=b.m[M03];
a.m[M10]+=b.m[M10]; a.m[M11]+=b.m[M11]; a.m[M12]+=b.m[M12]; a.m[M13]+=b.m[M13];
a.m[M20]+=b.m[M20]; a.m[M21]+=b.m[M21]; a.m[M22]+=b.m[M22]; a.m[M23]+=b.m[M23];
a.m[M30]+=b.m[M30]; a.m[M31]+=b.m[M31]; a.m[M32]+=b.m[M32]; a.m[M33]+=b.m[M33];
});

overload!((a: &mut Matrix4) -= (b: ?Matrix4) {
a.m[M00]-=b.m[M00]; a.m[M01]-=b.m[M01]; a.m[M02]-=b.m[M02]; a.m[M03]-=b.m[M03];
a.m[M10]-=b.m[M10]; a.m[M11]-=b.m[M11]; a.m[M12]-=b.m[M12]; a.m[M13]-=b.m[M13];
a.m[M20]-=b.m[M20]; a.m[M21]-=b.m[M21]; a.m[M22]-=b.m[M22]; a.m[M23]-=b.m[M23];
a.m[M30]-=b.m[M30]; a.m[M31]-=b.m[M31]; a.m[M32]-=b.m[M32]; a.m[M33]-=b.m[M33];
});

overload!((a: &mut Matrix4) *= (b: ?Matrix4) {
let m00=(a.m[M00]*b.m[M00])+(a.m[M01]*b.m[M10])+(a.m[M02]*b.m[M20])+(a.m[M03]*b.m[M30]);
let m01=(a.m[M00]*b.m[M01])+(a.m[M01]*b.m[M11])+(a.m[M02]*b.m[M21])+(a.m[M03]*b.m[M31]);
let m02=(a.m[M00]*b.m[M02])+(a.m[M01]*b.m[M12])+(a.m[M02]*b.m[M22])+(a.m[M03]*b.m[M32]);
let m03=(a.m[M00]*b.m[M03])+(a.m[M01]*b.m[M13])+(a.m[M02]*b.m[M23])+(a.m[M03]*b.m[M33]);
let m04=(a.m[M10]*b.m[M00])+(a.m[M11]*b.m[M10])+(a.m[M12]*b.m[M20])+(a.m[M13]*b.m[M30]);
let m05=(a.m[M10]*b.m[M01])+(a.m[M11]*b.m[M11])+(a.m[M12]*b.m[M21])+(a.m[M13]*b.m[M31]);
let m06=(a.m[M10]*b.m[M02])+(a.m[M11]*b.m[M12])+(a.m[M12]*b.m[M22])+(a.m[M13]*b.m[M32]);
let m07=(a.m[M10]*b.m[M03])+(a.m[M11]*b.m[M13])+(a.m[M12]*b.m[M23])+(a.m[M13]*b.m[M33]);
let m08=(a.m[M20]*b.m[M00])+(a.m[M21]*b.m[M10])+(a.m[M22]*b.m[M20])+(a.m[M23]*b.m[M30]);
let m09=(a.m[M20]*b.m[M01])+(a.m[M21]*b.m[M11])+(a.m[M22]*b.m[M21])+(a.m[M23]*b.m[M31]);
let m10=(a.m[M20]*b.m[M02])+(a.m[M21]*b.m[M12])+(a.m[M22]*b.m[M22])+(a.m[M23]*b.m[M32]);
let m11=(a.m[M20]*b.m[M03])+(a.m[M21]*b.m[M13])+(a.m[M22]*b.m[M23])+(a.m[M23]*b.m[M33]);
let m12=(a.m[M30]*b.m[M00])+(a.m[M31]*b.m[M10])+(a.m[M32]*b.m[M20])+(a.m[M33]*b.m[M30]);
let m13=(a.m[M30]*b.m[M01])+(a.m[M31]*b.m[M11])+(a.m[M32]*b.m[M21])+(a.m[M33]*b.m[M31]);
let m14=(a.m[M30]*b.m[M02])+(a.m[M31]*b.m[M12])+(a.m[M32]*b.m[M22])+(a.m[M33]*b.m[M32]);
let m15=(a.m[M30]*b.m[M03])+(a.m[M31]*b.m[M13])+(a.m[M32]*b.m[M23])+(a.m[M33]*b.m[M33]);
a.m[00]=m00;a.m[01]=m01;a.m[02]=m02;a.m[03]=m03;a.m[04]=m04;a.m[05]=m05;a.m[06]=m06;a.m[07]=m07;
a.m[08]=m08;a.m[09]=m09;a.m[10]=m10;a.m[11]=m11;a.m[12]=m12;a.m[13]=m13;a.m[14]=m14;a.m[15]=m15;
});
