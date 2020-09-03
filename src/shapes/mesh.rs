use crate::geometry::{Normal, Point2, Point3, Ray, Vec3};
use crate::shapes::{HitPoint, Intersection, KdTree, Shape, AABB};
use crate::utility::gamma;

struct VertexIndex {
    p: u32,
    n: u32,
    t: u32,
}

struct Triangle<'a> {
    mesh: &'a Mesh<'a>,
    a: VertexIndex,
    b: VertexIndex,
    c: VertexIndex,
}

impl Triangle<'_> {
    fn get_partial_derivatives(&self) -> (Vec3, Vec3) {
        let uv = [
            self.mesh.t[self.a.t as usize],
            self.mesh.t[self.b.t as usize],
            self.mesh.t[self.c.t as usize],
        ];
        let p = [
            self.mesh.p[self.a.p as usize],
            self.mesh.p[self.b.p as usize],
            self.mesh.p[self.c.p as usize],
        ];
        let delta = [uv[0] - uv[2], uv[1] - uv[2]];
        let determinant = delta[0].x * delta[1].y - delta[0].y * delta[1].x;
        if determinant == 0.0 {
            // This happens when user supplies degenerates UVs
            panic!("Triangle UV determinant should never be zero!")
        } else {
            let invdet = 1.0 / determinant;
            let dp = [p[0] - p[2], p[1] - p[2]];
            let dpdu = (dp[0] * delta[1].y - dp[1] * delta[1].x) * invdet;
            let dpdv = (dp[1] * delta[0].x - dp[0] * delta[0].y) * invdet;
            (dpdu, dpdv)
        }
    }

    fn object_to_ray(&self, ray: &Ray) -> ([Point3; 3], Vec3) {
        let a = self.mesh.p[self.a.p as usize];
        let b = self.mesh.p[self.b.p as usize];
        let c = self.mesh.p[self.c.p as usize];
        let origin_as_vec = Vec3::new(ray.origin.x, ray.origin.y, ray.origin.z);
        let mut a_translated = a - origin_as_vec;
        let mut b_translated = b - origin_as_vec;
        let mut c_translated = c - origin_as_vec;
        let abs_dir = ray.direction.abs();
        let z_dim = index_max_component(&abs_dir);
        let x_dim = if z_dim + 1 != 3 { z_dim + 1 } else { 0 };
        let y_dim = if x_dim + 1 != 3 { x_dim + 1 } else { 0 };
        let dir = Vec3::new(
            ray.direction[x_dim],
            ray.direction[y_dim],
            ray.direction[z_dim],
        );
        a_translated = Point3::new(
            a_translated[x_dim],
            a_translated[y_dim],
            a_translated[z_dim],
        );
        b_translated = Point3::new(
            b_translated[x_dim],
            b_translated[y_dim],
            b_translated[z_dim],
        );
        c_translated = Point3::new(
            c_translated[x_dim],
            c_translated[y_dim],
            c_translated[z_dim],
        );
        let invdir_z = 1.0 / dir.z;
        let shear = Vec3::new(-dir.x * invdir_z, -dir.y * invdir_z, invdir_z);
        a_translated.x += shear.x * a_translated.z;
        a_translated.y += shear.y * a_translated.z;
        b_translated.x += shear.x * b_translated.z;
        b_translated.y += shear.y * b_translated.z;
        c_translated.x += shear.x * c_translated.z;
        c_translated.y += shear.y * c_translated.z;
        ([a_translated, b_translated, c_translated], shear)
    }

    fn hit_barycentric_coords(&self, ray: &Ray) -> Option<(f32, Point3)> {
        let data = self.object_to_ray(&ray);
        let a = data.0[0];
        let b = data.0[1];
        let c = data.0[2];
        let shear = data.1;
        let mut e0 = b.x * c.y - b.y * c.x;
        let mut e1 = c.x * a.y - c.y * a.x;
        let mut e2 = a.x * b.y - a.y * b.x;
        if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
            e0 = (b.x as f64 * c.y as f64 - b.y as f64 * c.x as f64) as f32;
            e1 = (c.x as f64 * a.y as f64 - c.y as f64 * a.x as f64) as f32;
            e2 = (a.x as f64 * b.y as f64 - a.y as f64 * b.x as f64) as f32;
        }
        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            None
        } else {
            let det = e0 + e1 + e2;
            if det == 0.0 {
                None
            } else {
                let az_scaled = a.z * shear.z;
                let bz_scaled = b.z * shear.z;
                let cz_scaled = c.z * shear.z;
                let distance_scaled = e0 * az_scaled + e1 * bz_scaled + e2 * cz_scaled;
                if det < 0.0 && distance_scaled >= 0.0 || det > 0.0 && distance_scaled <= 0.0 {
                    None
                } else {
                    let inv_det = 1.0 / det;
                    let distance = distance_scaled * inv_det;

                    let zt = index_max_component(&Vec3::new(az_scaled, bz_scaled, cz_scaled).abs());
                    let yt = index_max_component(&Vec3::new(a.y, b.y, c.y).abs());
                    let xt = index_max_component(&Vec3::new(a.x, b.x, c.x).abs());
                    let et = index_max_component(&Vec3::new(e0, e1, e2).abs());
                    let delta_x = gamma(5) * (xt + zt) as f32;
                    let delta_y = gamma(5) * (yt + zt) as f32;
                    let delta_z = gamma(3) * zt as f32;
                    let delta_e = 2.0
                        * (gamma(2) * (xt * yt) as f32 + delta_y * xt as f32 + delta_x * yt as f32);
                    let delta_t = 3.0
                        * (gamma(3) * (et * zt) as f32 + delta_e * zt as f32 + delta_z * et as f32)
                        * inv_det.abs();
                    if distance <= delta_t {
                        None
                    } else {
                        Some((
                            distance,
                            Point3::new(e0 * inv_det, e1 * inv_det, e2 * inv_det),
                        ))
                    }
                }
            }
        }
    }
}

impl Shape for Triangle<'_> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        match self.hit_barycentric_coords(&ray) {
            Some(data) => {
                let distance = data.0;
                let b = data.1;
                let p = [
                    self.mesh.p[self.a.p as usize],
                    self.mesh.p[self.b.p as usize],
                    self.mesh.p[self.c.p as usize],
                ];
                let n = [
                    self.mesh.n[self.a.n as usize],
                    self.mesh.n[self.b.n as usize],
                    self.mesh.n[self.c.n as usize],
                ];
                let uv = [
                    self.mesh.t[self.a.t as usize],
                    self.mesh.t[self.b.t as usize],
                    self.mesh.t[self.c.t as usize],
                ];
                let derivatives = self.get_partial_derivatives();
                let hit = HitPoint {
                    point: Point3::new(
                        p[0].x * b.x + p[1].x * b.y + p[2].x * b.z,
                        p[0].y * b.x + p[1].y * b.y + p[2].y * b.z,
                        p[0].z * b.x + p[1].z * b.y + p[2].z * b.z,
                    ),
                    normal: (n[0] * b.x + n[1] * b.y + n[2] * b.z).normalize(),
                    dpdu: derivatives.0,
                    dpdv: derivatives.1,
                    uv: Point2::new(
                        uv[0].x * b.x + uv[1].x * b.y + uv[2].x * b.z,
                        uv[0].y * b.x + uv[1].y * b.y + uv[2].y * b.z,
                    ),
                };
                Some(Intersection { distance, hit })
            }
            None => None,
        }
    }

    fn intersect_fast(&self, ray: &Ray) -> bool {
        self.hit_barycentric_coords(&ray).is_some()
    }

    fn bounding_box(&self) -> AABB {
        let a = self.mesh.p[self.a.p as usize];
        let b = self.mesh.p[self.b.p as usize];
        let c = self.mesh.p[self.c.p as usize];
        let bot = Point3::min(&Point3::min(&a, &b), &c);
        let top = Point3::max(&Point3::max(&a, &b), &c);
        AABB { bot, top }
    }
}

struct Mesh<'a> {
    p: Vec<Point3>,
    n: Vec<Normal>,
    t: Vec<Point2>,
    f: KdTree<Triangle<'a>>,
}

impl Shape for Mesh<'_> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        self.f.intersect(&ray)
    }

    fn intersect_fast(&self, ray: &Ray) -> bool {
        self.f.intersect_fast(&ray)
    }

    fn bounding_box(&self) -> AABB {
        self.f.bounding_box()
    }
}

#[inline]
fn index_max_component(vec: &Vec3) -> u8 {
    if vec.x > vec.y {
        if vec.x > vec.z {
            0 as u8
        } else {
            2 as u8
        }
    } else if vec.y > vec.z {
        1 as u8
    } else {
        2 as u8
    }
}
