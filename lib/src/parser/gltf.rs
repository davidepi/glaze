use crate::{
    Camera, Light, Material, Mesh, MeshInstance, Meta, OrthographicCam, ParsedScene,
    PerspectiveCam, Texture, Transform, Vertex,
};
use cgmath::{InnerSpace, Matrix4, Point3, Vector3 as Vec3};
use gltf::camera::Projection::{Orthographic, Perspective};
use gltf::Gltf;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Seek};
use std::path::Path;

struct ContentGLTF {
    gltf: Gltf,
}

impl ContentGLTF {
    /// Parses a glTF or binary glTF file
    pub fn parse_gltf<P: AsRef<Path>>(path: P, mut file: File) -> Result<Self, Error> {
        file.rewind()?;
        let reader = BufReader::new(file);
        let gltf = Gltf::from_reader(reader);
        match gltf {
            Ok(gltf) => Ok(ContentGLTF { gltf }),
            Err(e) => Err(Error::new(ErrorKind::InvalidData, e.to_string())),
        }
    }
}

impl ParsedScene for ContentGLTF {
    fn vertices(&self) -> Result<Vec<Vertex>, Error> {
        todo!()
    }

    fn meshes(&self) -> Result<Vec<Mesh>, Error> {
        todo!()
    }

    fn transforms(&self) -> Result<Vec<Transform>, Error> {
        todo!()
    }

    fn instances(&self) -> Result<Vec<MeshInstance>, Error> {
        todo!()
    }

    fn cameras(&self) -> Result<Vec<Camera>, Error> {
        if let Some(scene) = self.gltf.scenes().next() {
            let mut cameras = Vec::new();
            let camera_nodes = scene
                .nodes()
                .into_iter()
                .filter(|n| n.camera().is_some())
                .map(|n| (n.camera().unwrap(), n.transform()));
            for (camera, transform) in camera_nodes {
                let (position, target, up) = camera_trans_to_vec(transform);
                let converted_camera = match camera.projection() {
                    Orthographic(ortho) => Camera::Orthographic(OrthographicCam {
                        position,
                        target,
                        up,
                        scale: ortho.xmag(),
                        near: ortho.znear(),
                        far: ortho.zfar(),
                    }),
                    Perspective(persp) => Camera::Perspective(PerspectiveCam {
                        position,
                        target,
                        up,
                        fovy: persp.yfov(),
                        near: persp.znear(),
                        far: persp.zfar().unwrap_or(1E3),
                    }),
                };
                cameras.push(converted_camera);
            }
            Ok(cameras)
        } else {
            Err(Error::new(
                ErrorKind::NotFound,
                "No scene was found in the file",
            ))
        }
    }

    fn textures(&self) -> Result<Vec<Texture>, Error> {
        todo!()
    }

    fn materials(&self) -> Result<Vec<Material>, Error> {
        todo!()
    }

    fn lights(&self) -> Result<Vec<Light>, Error> {
        todo!()
    }

    fn meta(&self) -> Result<Meta, Error> {
        todo!()
    }

    fn update(
        &mut self,
        _: Option<&[Camera]>,
        _: Option<&[Material]>,
        _: Option<&[Light]>,
        _: Option<&[Texture]>,
        _: Option<&Meta>,
    ) -> Result<(), Error> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "glTF file can not be updated. Use \"save as...\" to create a new file.",
        ))
    }
}
// Transform a camera scene node into (position, target, up) vectors
fn camera_trans_to_vec(trans: gltf::scene::Transform) -> (Point3<f32>, Point3<f32>, Vec3<f32>) {
    use cgmath::Transform;

    let trans_mat = Matrix4::from(trans.matrix());
    let default_position = Point3::new(0.0, 0.0, 0.0);
    let default_up = Vec3::new(0.0, 1.0, 0.0);
    let default_dir = Vec3::new(0.0, 0.0, -1.0); // default gltf dir for cameras is 0,0,-1
    let position = trans_mat.transform_point(default_position);
    let up = trans_mat.transform_vector(default_up).normalize();
    let dir = trans_mat.transform_vector(default_dir).normalize();
    // using an arbitrary distance for target, this is not provided by gltf
    let target = position + dir * 100.0;
    (position, target, up)
}
