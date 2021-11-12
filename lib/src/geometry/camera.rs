use cgmath::{ortho, perspective, Deg, Matrix4, Point3, Rad, Vector3 as Vec3};

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
    pub fn look_at_rh(&self) -> Matrix4<f32> {
        match self {
            Camera::Perspective(cam) => Matrix4::look_at_rh(cam.position, cam.target, cam.up),
            Camera::Orthographic(cam) => Matrix4::look_at_rh(cam.position, cam.target, cam.up),
        }
    }

    pub fn projection(&self, aspect_ratio: f32) -> Matrix4<f32> {
        match self {
            Camera::Perspective(cam) => {
                perspective(Rad(cam.fovy(aspect_ratio)), aspect_ratio, 1.0, 100.0)
            }

            Camera::Orthographic(cam) => {
                ortho(-cam.scale, cam.scale, -cam.scale, cam.scale, 1.0, 100.0)
            }
        }
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
