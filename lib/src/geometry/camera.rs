use cgmath::{ortho, perspective, InnerSpace, Matrix3, Matrix4, Point3, Rad, Vector3 as Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct PerspectiveCam {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vec3<f32>,
    pub fovx: f32, // in radians
}

impl PerspectiveCam {
    pub fn fovy(&self, aspect_ratio: f32) -> f32 {
        2.0 * f32::atan(f32::tan(self.fovx * 0.5) / aspect_ratio)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrthographicCam {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vec3<f32>,
    pub scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Camera {
    Perspective(PerspectiveCam),
    Orthographic(OrthographicCam),
}

impl Camera {
    pub fn position(&self) -> Point3<f32> {
        match self {
            Camera::Perspective(cam) => cam.position,
            Camera::Orthographic(cam) => cam.position,
        }
    }
    pub fn target(&self) -> Point3<f32> {
        match self {
            Camera::Perspective(cam) => cam.target,
            Camera::Orthographic(cam) => cam.target,
        }
    }
    pub fn up(&self) -> Vec3<f32> {
        match self {
            Camera::Perspective(cam) => cam.up,
            Camera::Orthographic(cam) => cam.up,
        }
    }

    pub fn look_at_rh(&self) -> Matrix4<f32> {
        match self {
            Camera::Perspective(cam) => Matrix4::look_at_rh(cam.position, cam.target, cam.up),
            Camera::Orthographic(cam) => Matrix4::look_at_rh(cam.position, cam.target, cam.up),
        }
    }

    pub fn projection(&self, aspect_ratio: f32) -> Matrix4<f32> {
        match self {
            Camera::Perspective(cam) => {
                perspective(Rad(cam.fovy(aspect_ratio)), aspect_ratio, 0.1, 250.0)
            }

            Camera::Orthographic(cam) => {
                ortho(-cam.scale, cam.scale, -cam.scale, cam.scale, 0.1, 250.0)
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use crate::PerspectiveCam;
    use cgmath::{Point3, Vector3 as Vec3};
    use float_cmp::assert_approx_eq;

    #[test]
    pub fn fovx_to_fovy() {
        let camera = PerspectiveCam {
            position: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fovx: f32::to_radians(91.0),
        };
        let fovy = camera.fovy(1.453);
        assert_approx_eq!(f32, fovy, f32::to_radians(70.0), epsilon = 1e-3);
    }
}
