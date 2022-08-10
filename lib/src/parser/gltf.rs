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

#[cfg(test)]
mod tests {
    use super::ContentGLTF;
    use crate::ParsedScene;
    use cgmath::Point3;
    use curl::easy::Easy;
    use sha2::{Digest, Sha256};
    use std::fs::{self, File};
    use std::io::{Error as IOError, ErrorKind, Read, Write};
    use std::path::Path;

    const RES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../resources/gltf");
    // downloads a gltf assets from the github repository KhronosGroup/glTF-Sample-Models
    //
    // using commit 16e8034
    fn download_gltf_asset(url: &'static str, sha256: &'static str) -> Result<(), IOError> {
        // creates the folder if not existing, extract filename
        fs::create_dir_all(RES_DIR)?;
        let filename = Path::new(&url).file_name().unwrap().to_str().unwrap();
        const BASE_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/16e803435fca5b07dde3dbdc5bd0e9b8374b2750/";
        let download_url = format!("{}{}", BASE_URL, url);
        let file_path = format!("{}/{}", RES_DIR, filename);
        let mut data = Vec::new();
        // try to read the file, if existing
        if let Ok(mut existing_file) = File::open(file_path.clone()) {
            if existing_file.read_to_end(&mut data).is_ok() {
                // check sha1 and return if correct, otherwise download a new file
                let mut hasher = Sha256::new();
                hasher.update(&data);
                let result = format!("{:x}", hasher.finalize());
                if result == sha256 {
                    return Ok(());
                }
            }
        }
        // download file
        let mut easy = Easy::new();
        easy.url(&download_url)?;
        easy.follow_location(true)?;
        {
            let mut transfer = easy.transfer();
            transfer.write_function(|c| {
                data.extend_from_slice(c);
                Ok(c.len())
            })?;
            transfer.perform()?;
        }
        // ensures the sha matches
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = format!("{:x}", hasher.finalize());
        if result == sha256 {
            let mut file = File::create(file_path)?;
            file.write_all(&data)
        } else {
            Err(IOError::new(
                ErrorKind::InvalidData,
                "Downloaded file with wrong sha256",
            ))
        }
    }

    fn open_file(name: &'static str) -> Result<ContentGLTF, IOError> {
        let path = format!("{}/{}", RES_DIR, name);
        let file = File::open(path.clone())?;
        ContentGLTF::parse_gltf(path, file)
    }

    #[test]
    fn camera_test() -> Result<(), IOError> {
        download_gltf_asset(
            "2.0/Cameras/glTF-Embedded/Cameras.gltf",
            "601cd3af3728de69b45aa3a91f8a6849799479ae41eede3912ee0ac92190b2f2",
        )?;
        let gltf = open_file("Cameras.gltf")?;
        let cameras = gltf.cameras()?;
        assert_eq!(cameras.len(), 2);
        println!("{:?}", cameras[0].up());
        match cameras[0] {
            crate::Camera::Perspective(p) => {
                assert_eq!(p.position, Point3::new(0.5, 0.5, 3.0));
                assert_eq!(p.fovy, 0.7);
                assert_eq!(p.far, 100.0);
                assert_eq!(p.near, 0.01);
            }
            crate::Camera::Orthographic(_) => panic!("Wrong type of camera"),
        }
        match cameras[1] {
            crate::Camera::Orthographic(o) => {
                assert_eq!(o.position, Point3::new(0.5, 0.5, 3.0));
                assert_eq!(o.scale, 1.0);
                assert_eq!(o.far, 100.0);
                assert_eq!(o.near, 0.01);
            }
            crate::Camera::Perspective(_) => panic!("Wrong type of camera"),
        }
        Ok(())
    }
}
