use crate::linear::{Normal, Point2, Point3, Ray, Vec3};
use crate::parser::ParsedGeometry;
use crate::shapes::accelerator::BufferedAccelerator;
use crate::shapes::shape::BufferedShape;
use crate::shapes::{HitPoint, Intersection, KdTree, Shape, VertexBuffer, AABB};
use crate::utility::gamma;

/// Struct representing a vertex.
///
/// The vertex is indexed over some buffers (not contained in this struct).
struct VertexIndex<T>
where
    T: Into<u32> + Copy,
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
pub(crate) struct Triangle<T>
where
    T: Into<u32> + Copy,
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
    T: Into<u32> + Copy,
{
    /// Calculates the partial derivatives ∂p/∂u and ∂p/∂v.
    ///
    /// Note that this value is always the same for each triangle and does not depend on a specific
    /// point (and thus on a specific Ray).
    fn get_partial_derivatives(&self, vb: &VertexBuffer) -> (Vec3, Vec3) {
        let uv = [
            vb.texture_buffer[self.a.t.into() as usize],
            vb.texture_buffer[self.b.t.into() as usize],
            vb.texture_buffer[self.c.t.into() as usize],
        ];
        let p = [
            vb.point_buffer[self.a.p.into() as usize],
            vb.point_buffer[self.b.p.into() as usize],
            vb.point_buffer[self.c.p.into() as usize],
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
        let a = vb.point_buffer[self.a.p.into() as usize];
        let b = vb.point_buffer[self.b.p.into() as usize];
        let c = vb.point_buffer[self.c.p.into() as usize];
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

impl<T> BufferedShape for Triangle<T>
where
    T: Into<u32> + Copy,
{
    fn intersect(&self, ray: &Ray, vb: &VertexBuffer) -> Option<Intersection> {
        match self.hit_barycentric_coords(&ray, vb) {
            Some(data) => {
                let distance = data.0;
                let b = data.1;
                let p = [
                    vb.point_buffer[self.a.p.into() as usize],
                    vb.point_buffer[self.b.p.into() as usize],
                    vb.point_buffer[self.c.p.into() as usize],
                ];
                let n = [
                    vb.normal_buffer[self.a.n.into() as usize],
                    vb.normal_buffer[self.b.n.into() as usize],
                    vb.normal_buffer[self.c.n.into() as usize],
                ];
                let uv = [
                    vb.texture_buffer[self.a.t.into() as usize],
                    vb.texture_buffer[self.b.t.into() as usize],
                    vb.texture_buffer[self.c.t.into() as usize],
                ];
                let derivatives = self.get_partial_derivatives(vb);
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

    fn intersect_fast(&self, ray: &Ray, vb: &VertexBuffer) -> bool {
        self.hit_barycentric_coords(&ray, vb).is_some()
    }

    fn bounding_box(&self, vb: &VertexBuffer) -> AABB {
        let a = vb.point_buffer[self.a.p.into() as usize];
        let b = vb.point_buffer[self.b.p.into() as usize];
        let c = vb.point_buffer[self.c.p.into() as usize];
        let bot = Point3::min(&Point3::min(&a, &b), &c);
        let top = Point3::max(&Point3::max(&a, &b), &c);
        AABB { bot, top }
    }
}

#[inline]
fn replace_neg(val: i32, len: usize) -> u32 {
    if val < 0 {
        (len as i32 - val) as u32
    } else {
        val as u32
    }
}

fn new_vertex_index_u8<T: Copy + Into<u32>>(p_idx: u32, t_idx: u32, n_idx: u32) -> VertexIndex<u8> {
    VertexIndex {
        p: p_idx as u8,
        t: t_idx as u8,
        n: n_idx as u8,
    }
}

fn new_vertex_index_u16<T: Copy + Into<u32>>(
    p_idx: u32,
    t_idx: u32,
    n_idx: u32,
) -> VertexIndex<u16> {
    VertexIndex {
        p: p_idx as u16,
        t: t_idx as u16,
        n: n_idx as u16,
    }
}

fn new_vertex_index_u32<T: Copy + Into<u32>>(
    p_idx: u32,
    t_idx: u32,
    n_idx: u32,
) -> VertexIndex<u32> {
    VertexIndex {
        p: p_idx as u32,
        t: t_idx as u32,
        n: n_idx as u32,
    }
}

fn new_vertex_buffer(data: &ParsedGeometry) -> VertexBuffer {
    VertexBuffer {
        point_buffer: data
            .vv
            .iter()
            .map(|x| Point3::new(x[0], x[1], x[2]))
            .collect(),
        texture_buffer: data.vt.iter().map(|x| Point2::new(x[0], x[1])).collect(),
        normal_buffer: data
            .vn
            .iter()
            .map(|x| Normal::new(x[0], x[1], x[2]))
            .collect(),
    }
}

fn new_triangles<T>(
    data: ParsedGeometry,
    buffer: &VertexBuffer,
    create: fn(u32, u32, u32) -> VertexIndex<T>,
) -> Vec<Triangle<T>>
where
    T: Into<u32> + Copy,
{
    let mut triangles = Vec::new();
    for face in data.ff {
        let a = create(
            replace_neg(face[0], buffer.point_buffer.len()),
            replace_neg(face[1], buffer.texture_buffer.len()),
            replace_neg(face[2], buffer.normal_buffer.len()),
        );
        let b = create(
            replace_neg(face[3], buffer.point_buffer.len()),
            replace_neg(face[4], buffer.texture_buffer.len()),
            replace_neg(face[5], buffer.normal_buffer.len()),
        );
        let c = create(
            replace_neg(face[6], buffer.point_buffer.len()),
            replace_neg(face[7], buffer.texture_buffer.len()),
            replace_neg(face[8], buffer.normal_buffer.len()),
        );
        triangles.push(Triangle { a, b, c })
    }
    triangles
}

/// A shape formed by a collection of triangles.
///
/// A mesh uses vertex, normal and vertex texture buffers for efficient storage and an acceleration
/// structure, a KdTree in this implementation for faster intersections.
pub struct Mesh<T>
where
    T: Into<u32> + Copy,
{
    data: VertexBuffer,
    /// Acceleration structure
    f: KdTree<Triangle<T>>,
}

impl<T> Mesh<T>
where
    T: Into<u32> + Copy,
{
    pub fn new_u8(geom: ParsedGeometry) -> Mesh<u8> {
        let mut mesh = Mesh {
            data: new_vertex_buffer(&geom),
            f: KdTree::default(),
        };
        let triangles = new_triangles(geom, &mesh.data, new_vertex_index_u8::<u8>);
        mesh.f = mesh.f.build(triangles, &mesh.data);
        mesh
    }

    pub fn new_u16(geom: ParsedGeometry) -> Mesh<u16> {
        let mut mesh = Mesh {
            data: new_vertex_buffer(&geom),
            f: KdTree::default(),
        };
        let triangles = new_triangles(geom, &mesh.data, new_vertex_index_u16::<u16>);
        mesh.f = mesh.f.build(triangles, &mesh.data);
        mesh
    }

    pub fn new_u32(geom: ParsedGeometry) -> Mesh<u32> {
        let mut mesh = Mesh {
            data: new_vertex_buffer(&geom),
            f: KdTree::default(),
        };
        let triangles = new_triangles(geom, &mesh.data, new_vertex_index_u32::<u32>);
        mesh.f = mesh.f.build(triangles, &mesh.data);
        mesh
    }
}

impl<T: Into<u32> + Copy> Shape for Mesh<T> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        self.f.intersect(&ray, &self.data)
    }

    fn intersect_fast(&self, ray: &Ray) -> bool {
        self.f.intersect_fast(&ray, &self.data)
    }

    fn bounding_box(&self) -> AABB {
        self.f.bounding_box(&self.data)
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
