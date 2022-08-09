use cgmath::{
    ortho, perspective, InnerSpace, Matrix3, Matrix4, Point3, Rad, Transform, Vector2 as Vec2,
    Vector3 as Vec3,
};

/// A camera exhibiting a perspective projection.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PerspectiveCam {
    /// Position of the camera.
    pub position: Point3<f32>,
    /// Target (focus) of the camera.
    pub target: Point3<f32>,
    /// Vector pointing upwards.
    pub up: Vec3<f32>,
    /// Horizontal field of view in radians.
    pub fovy: f32,
    /// Near clippling plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl PerspectiveCam {
    /// Converts a vertical field of view (rad) into a horizontal field of view (rad).
    ///
    /// Both field of views are expected in radians.
    pub fn fovy_to_fovx(fovy_rad: f32, aspect_ratio: f32) -> f32 {
        2.0 * f32::atan(f32::tan(fovy_rad * 0.5) * aspect_ratio)
    }

    /// Converts a horizontal field of view (rad) into a vertical field of view (rad).
    ///
    /// The Perspective camera requires a vertical field of view, this method is provided as
    /// convenience to perform the conversion.
    ///
    /// Both field of views are expected in radians.
    pub fn fovx_to_fovy(fovx_rad: f32, aspect_ratio: f32) -> f32 {
        2.0 * f32::atan(f32::tan(fovx_rad * 0.5) / aspect_ratio)
    }
}

impl Default for PerspectiveCam {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 100.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fovy: f32::to_radians(75.0),
            near: 1E-3,
            far: 1E3,
        }
    }
}

/// A camera exhibiting an orthographic projection.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct OrthographicCam {
    /// Position of the camera.
    pub position: Point3<f32>,
    /// Target (focus) of the camera.
    pub target: Point3<f32>,
    /// Vector pointing upwards.
    pub up: Vec3<f32>,
    /// Left/Right/Top/Bottom absolute values of the clipping plane.
    /// Sign will be adjusted accordingly.
    pub scale: f32,
    /// Near clippling plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Default for OrthographicCam {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 100.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            scale: 1.0,
            near: 1E-3,
            far: 1E3,
        }
    }
}

/// A projective camera in 3D space.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Camera {
    Perspective(PerspectiveCam),
    Orthographic(OrthographicCam),
}

impl Camera {
    /// Returns the position of the camera.
    pub fn position(&self) -> Point3<f32> {
        match self {
            Camera::Perspective(cam) => cam.position,
            Camera::Orthographic(cam) => cam.position,
        }
    }

    /// Returns the target of the camera.
    pub fn target(&self) -> Point3<f32> {
        match self {
            Camera::Perspective(cam) => cam.target,
            Camera::Orthographic(cam) => cam.target,
        }
    }

    /// Returns the `up` vector of the camera.
    pub fn up(&self) -> Vec3<f32> {
        match self {
            Camera::Perspective(cam) => cam.up,
            Camera::Orthographic(cam) => cam.up,
        }
    }

    /// Returns the value of the camera near clipping plane.
    pub fn near_plane(&self) -> f32 {
        match self {
            Camera::Perspective(cam) => cam.near,
            Camera::Orthographic(cam) => cam.near,
        }
    }

    /// Returns the value of the camera far clipping plane.
    pub fn far_plane(&self) -> f32 {
        match self {
            Camera::Perspective(cam) => cam.far,
            Camera::Orthographic(cam) => cam.far,
        }
    }

    /// Returns the right handed (i.e. OpenGL style) view matrix for the current camera.
    pub fn look_at_rh(&self) -> Matrix4<f32> {
        match self {
            Camera::Perspective(cam) => Matrix4::look_at_rh(cam.position, cam.target, cam.up),
            Camera::Orthographic(cam) => Matrix4::look_at_rh(cam.position, cam.target, cam.up),
        }
    }

    /// Returns the projection matrix for the current camera.
    pub fn projection(&self, width: u32, height: u32) -> Matrix4<f32> {
        match self {
            Camera::Perspective(cam) => {
                let ar = width as f32 / height as f32;
                perspective(Rad(cam.fovy), ar, cam.near, cam.far)
            }

            Camera::Orthographic(cam) => ortho(
                -cam.scale, cam.scale, -cam.scale, cam.scale, -cam.far, cam.far,
            ),
        }
    }

    /// Returns a ray in screen space, given a coordinate in the screen from (-1, -1) to (1, 1).
    fn ray_screen_space(&self, ndc: Vec2<f32>) -> (Point3<f32>, Vec3<f32>) {
        match self {
            Camera::Perspective(_) => (
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(ndc.x, ndc.y, 1.0).normalize(),
            ),
            Camera::Orthographic(_) => (Point3::new(ndc.x, ndc.y, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        }
    }

    /// Returns a ray in world space, given a coordinare in the screen from (-1, -1) to (1, 1).
    ///
    /// The screen2camera matrix is the inverse of the camera projection matrix, and the
    /// camera2world matrix is the inverse of the look_at matrix.
    pub fn ray_world_space(
        &self,
        ndc: Vec2<f32>,
        screen2camera: Matrix4<f32>,
        camera2world: Matrix4<f32>,
    ) -> (Point3<f32>, Vec3<f32>) {
        let (origin_ss, direction_ss) = self.ray_screen_space(ndc);
        let screen2world = screen2camera * camera2world;
        match self {
            Camera::Perspective(_) => (
                camera2world.transform_point(origin_ss),
                screen2world.transform_vector(direction_ss).normalize(),
            ),
            Camera::Orthographic(_) => (
                screen2world.transform_point(origin_ss),
                camera2world.transform_vector(direction_ss).normalize(),
            ),
        }
    }

    /// Moves the camera by strafing left/right by the given magnitude.
    ///
    /// The left/right direction is determined by the magnitude sign.
    pub fn strafe(&mut self, magnitude: f32) {
        let position;
        let target;
        let up;
        match self {
            Camera::Perspective(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
                up = &mut cam.up;
            }
            Camera::Orthographic(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
                up = &mut cam.up;
            }
        }
        // extract the movement axis ("right" vector)
        let dir = (*target - *position).normalize();
        let right = dir.cross(*up).normalize();
        let mov_vec = right * magnitude;
        *position += mov_vec;
        *target += mov_vec;
    }

    /// Advance the camera towards its target by the given magnitude.
    ///
    /// If the magnitude is negative, the camera will move away from the target.
    pub fn advance(&mut self, magnitude: f32) {
        let position;
        let target;
        match self {
            Camera::Perspective(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
            }
            Camera::Orthographic(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
            }
        }
        if target == position {
            target.z += f32::EPSILON;
        }
        // no need to extract the rotation matrix
        let mov_vec = (*target - *position).normalize();
        *position += mov_vec * magnitude;
        *target += mov_vec * magnitude;
    }

    /// Move the camera along its vertical axis.
    pub fn elevate(&mut self, magnitude: f32) {
        let position;
        let target;
        let up;
        match self {
            Camera::Perspective(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
                up = &mut cam.up;
            }
            Camera::Orthographic(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
                up = &mut cam.up;
            }
        }
        // move along the up vector
        *position += magnitude * *up;
        *target += magnitude * *up;
    }

    /// Move the camera target while keeping the same camera position.
    /// The horizontal angle of movement is `theta`, and the vertical angle is `phi`.
    /// Both angles are expressed in radians.
    pub fn look_around(&mut self, theta: f32, phi: f32) {
        let position;
        let target;
        let up;
        match self {
            Camera::Perspective(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
                up = &mut cam.up;
            }
            Camera::Orthographic(cam) => {
                position = &mut cam.position;
                target = &mut cam.target;
                up = &mut cam.up;
            }
        }
        let mut direction = *target - *position;
        let radius = direction.magnitude();
        direction = direction.normalize();
        let right = direction.cross(*up).normalize();
        let h_rot = Matrix3::from_axis_angle(*up, Rad(theta));
        let v_rot = Matrix3::from_axis_angle(right, Rad(phi));
        let rotation = h_rot * v_rot;
        *target = *position + radius * rotation * direction;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera::Perspective(PerspectiveCam::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::PerspectiveCam;
    use float_cmp::assert_approx_eq;

    #[test]
    pub fn fovx_to_fovy() {
        let fovx = f32::to_radians(91.0);
        let fovy = PerspectiveCam::fovx_to_fovy(fovx, 1.453);
        assert_approx_eq!(f32, fovy, f32::to_radians(70.0), epsilon = 1e-3);
    }

    #[test]
    pub fn fovy_to_fovx() {
        let fovy = f32::to_radians(70.0);
        let fovx = PerspectiveCam::fovy_to_fovx(fovy, 1.453);
        assert_approx_eq!(f32, fovx, f32::to_radians(91.0), epsilon = 1e-3);
    }
}
