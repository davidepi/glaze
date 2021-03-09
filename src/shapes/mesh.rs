use crate::linear::{Point2, Point3, Ray, Vec3};
use crate::parser::ParsedGeometry;
use crate::shapes::{Accelerator, HitPoint, Intersection, KdTree, Shape, VertexBuffer, AABB};
use crate::utility::gamma;
use std::cmp::max;

/// Struct representing a vertex.
///
/// The vertex is indexed over some buffers (not contained in this struct).
struct VertexIndex<T>
where
    T: Into<usize> + Copy,
{
    /// Index for the vertex coordinate.
    p: T,
    /// Index for the vertex normal.
    n: T,
    /// Index for the vertex texture coordinate.
    t: T,
}

/// Struct representing a triangle.
///
/// The triangle is closely coupled with the mesh containing it. Its object space is the same of the
/// Mesh object space (the triangle coordinates won't change when transforming from the Mesh object
/// space to the Triangle object space).
struct Triangle<T>
where
    T: Into<usize> + Copy,
{
    /// Struct containing indices for the `a` vertex, in the Mesh buffers.
    a: VertexIndex<T>,
    /// Struct containing indices for the `b` vertex, in the Mesh buffers.
    b: VertexIndex<T>,
    /// Struct containing indices for the `c` vertex, in the Mesh buffers.
    c: VertexIndex<T>,
}

impl<T> Triangle<T>
where
    T: Into<usize> + Copy,
{
    /// Calculates the partial derivatives ∂p/∂u and ∂p/∂v.
    ///
    /// Note that this value is always the same for each triangle and does not depend on a specific
    /// point (and thus on a specific Ray).
    fn get_partial_derivatives(&self, vb: &VertexBuffer) -> (Vec3, Vec3) {
        let uv = [
            vb.texture_buffer[self.a.t.into()],
            vb.texture_buffer[self.b.t.into()],
            vb.texture_buffer[self.c.t.into()],
        ];
        let p = [
            vb.point_buffer[self.a.p.into()],
            vb.point_buffer[self.b.p.into()],
            vb.point_buffer[self.c.p.into()],
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

    /// Coverts the triangle points from object space to ray space, with the ray origin in (0,0,0).
    fn object_to_ray(&self, ray: &Ray, vb: &VertexBuffer) -> [Point3; 3] {
        let a = vb.point_buffer[self.a.p.into()];
        let b = vb.point_buffer[self.b.p.into()];
        let c = vb.point_buffer[self.c.p.into()];
        // transform the vertices so ray origin is at (0, 0, 0)
        let origin_as_vec = Vec3::new(ray.origin.x, ray.origin.y, ray.origin.z);
        let mut a_translated = a - origin_as_vec;
        let mut b_translated = b - origin_as_vec;
        let mut c_translated = c - origin_as_vec;
        // cycle the components of the direction so z is the one with the highest absolute val
        // this ensures z is never zero (except when the entire direction is degenerate, but at
        // that point you have bigger problems)
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
        // aligns the ray direction with the new axis, scale the points correctly (shear transform)
        let shear = Vec3::new(-dir.x * invdir_z, -dir.y * invdir_z, invdir_z);
        let ret_a = Point3::new(
            a_translated.x + shear.x * a_translated.z,
            a_translated.y + shear.y * a_translated.z,
            a_translated.z * shear.z,
        );
        let ret_b = Point3::new(
            b_translated.x + shear.x * b_translated.z,
            b_translated.y + shear.y * b_translated.z,
            b_translated.z * shear.z,
        );
        let ret_c = Point3::new(
            c_translated.x + shear.x * c_translated.z,
            c_translated.y + shear.y * c_translated.z,
            c_translated.z * shear.z,
        );
        [ret_a, ret_b, ret_c]
    }

    /// Returns the point hit by a ray in barycentric coordinates, or None if no intersection
    /// happened.
    ///
    /// The returned tuple contains the distance and a Point3 with the coordinates.
    fn hit_barycentric_coords(&self, ray: &Ray, vb: &VertexBuffer) -> Option<(f32, Point3)> {
        let data = self.object_to_ray(&ray, vb);
        let a = data[0];
        let b = data[1];
        let c = data[2];
        // use edge function defined by J.Pineda in "A Parallel Algorithm for Polygon Rasterization"
        // value is negative if a point is left of the edge, zero on the edge, positive if right of
        let mut e0 = b.x * c.y - b.y * c.x;
        let mut e1 = c.x * a.y - c.y * a.x;
        let mut e2 = a.x * b.y - a.y * b.x;
        if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
            // edge function is EXACTLY zero, use double precision
            e0 = (b.x as f64 * c.y as f64 - b.y as f64 * c.x as f64) as f32;
            e1 = (c.x as f64 * a.y as f64 - c.y as f64 * a.x as f64) as f32;
            e2 = (a.x as f64 * b.y as f64 - a.y as f64 * b.x as f64) as f32;
        }
        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            // ray origin is not aligned with the triangle
            None
        } else {
            let det = e0 + e1 + e2;
            if det == 0.0 {
                // ray is approaching the triangle on the edge side
                None
            } else {
                let distance_scaled = e0 * a.z + e1 * b.z + e2 * c.z;
                if det < 0.0 && distance_scaled >= 0.0 || det > 0.0 && distance_scaled <= 0.0 {
                    // intersection exists but is not valid
                    None
                } else {
                    let inv_det = 1.0 / det;
                    let distance = distance_scaled * inv_det;
                    // check for fp errors, compare with the pbrt v3 book, section 3.9.6
                    let zt = index_max_component(&Vec3::new(a.z, b.z, c.z).abs());
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
                        // intersection due to fp error
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

impl<T> Shape for Triangle<T>
where
    T: Into<usize> + Copy,
{
    fn intersect(&self, ray: &Ray, vb: Option<&VertexBuffer>) -> Option<Intersection> {
        let buffer = vb.unwrap();
        match self.hit_barycentric_coords(&ray, buffer) {
            Some(data) => {
                let distance = data.0;
                let buffer = vb.unwrap();
                let b = data.1;
                let p = [
                    buffer.point_buffer[self.a.p.into()],
                    buffer.point_buffer[self.b.p.into()],
                    buffer.point_buffer[self.c.p.into()],
                ];
                let n = [
                    buffer.normal_buffer[self.a.n.into()],
                    buffer.normal_buffer[self.b.n.into()],
                    buffer.normal_buffer[self.c.n.into()],
                ];
                let uv = [
                    buffer.texture_buffer[self.a.t.into()],
                    buffer.texture_buffer[self.b.t.into()],
                    buffer.texture_buffer[self.c.t.into()],
                ];
                let derivatives = self.get_partial_derivatives(buffer);
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

    fn intersect_fast(&self, ray: &Ray, vb: Option<&VertexBuffer>) -> bool {
        let buffer = vb.unwrap();
        self.hit_barycentric_coords(&ray, buffer).is_some()
    }

    fn bounding_box(&self, vb: Option<&VertexBuffer>) -> AABB {
        let buffer = vb.unwrap();
        let a = buffer.point_buffer[self.a.p.into()];
        let b = buffer.point_buffer[self.b.p.into()];
        let c = buffer.point_buffer[self.c.p.into()];
        let bot = Point3::min(&Point3::min(&a, &b), &c);
        let top = Point3::max(&Point3::max(&a, &b), &c);
        AABB { bot, top }
    }
}

fn new_triangles_u8<T>(data: ParsedGeometry) -> (Vec<Triangle<T>>, VertexBuffer)
where
    T: Into<usize> + Copy,
{
    unimplemented!()
}

fn new_triangles_u16<T>(data: ParsedGeometry) -> (Vec<Triangle<T>>, VertexBuffer)
where
    T: Into<usize> + Copy,
{
    unimplemented!()
}

fn new_triangles_u32<T>(data: ParsedGeometry) -> (Vec<Triangle<T>>, VertexBuffer)
where
    T: Into<usize> + Copy,
{
    unimplemented!()
}

/// A shape formed by a collection of triangles.
///
/// A mesh uses vertex, normal and vertex texture buffers for efficient storage and an acceleration
/// structure, a KdTree in this implementation for faster intersections.
pub struct Mesh<T: Into<usize> + Copy> {
    data: VertexBuffer,
    /// Acceleration structure
    f: KdTree<Triangle<T>>,
}

impl<'a, T: Into<usize> + Copy> Mesh<T> {
    fn new(geom: ParsedGeometry) -> Mesh<T> {
        let max = max(max(geom.vv.len(), geom.vt.len()), geom.vn.len());
        let create = if max < 256 {
            new_triangles_u8
        } else if max < 65536 {
            new_triangles_u16
        } else {
            new_triangles_u32
        };
        let data = create(geom);
        let accel = KdTree::default().build(data.0, Some(&data.1));
        Mesh {
            data: data.1,
            f: accel,
        }
    }
}

impl<T: Into<usize> + Copy> Shape for Mesh<T> {
    fn intersect(&self, ray: &Ray, vb: Option<&VertexBuffer>) -> Option<Intersection> {
        self.f.intersect(&ray, vb)
    }

    fn intersect_fast(&self, ray: &Ray, vb: Option<&VertexBuffer>) -> bool {
        self.f.intersect_fast(&ray, vb)
    }

    fn bounding_box(&self, vb: Option<&VertexBuffer>) -> AABB {
        self.f.bounding_box(vb)
    }
}

/// Gets the component of a vector with the highest value:
/// - x -> 0
/// - y -> 1
/// - z -> 2
#[inline]
fn index_max_component(vec: &Vec3) -> u8 {
    if vec.x > vec.y {
        if vec.x > vec.z {
            0
        } else {
            2
        }
    } else if vec.y > vec.z {
        1
    } else {
        2
    }
}
