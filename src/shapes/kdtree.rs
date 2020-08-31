use crate::geometry::{Ray, Vec3};
use crate::shapes::{Accelerator, Intersection, Shape, AABB};
use fnv::FnvHashSet;
use std::f32::NAN;

/// Maximum depth of the tree
const MAX_DEPTH: usize = 15;
/// Dimensionality of the tree. We are operating in a three dimensional space.
/// Changing this does not mean the kd-tree will work with more dimensions, the shapes are still 3D.
const DIMENSIONS: usize = 3;
/// Maximum allowed descends in the tree that are not convenient in term of cost (in the hope
/// of finding an overall better cost)
const IMPERFECT_SPLIT_TOLERANCE: usize = 2;
/// Maximum number of shapes supported. Due to how indexing works in KdNode.
const MAX_SHAPES: u32 = 0x3FFFFFFF;

/// Space-partitioning based acceleration structure.
///
/// KdTree, for K-dimensional tree, is a space partitioning data structure, used to improve the
/// speed of the intersection routines between a Ray and a Shape. It can be seen as a container
/// for Shapes.
///
/// This struct owns the Shapes it contains and implements the Shape trait itself. A call on any
/// method of the Shape trait and this class will efficiently find the result on the contained
/// data.
///
/// This struct uses a Surface Area Heuristic to build the tree: a cost is assigned to intersecting
/// the primitives or descending further down the tree. The tree is built considering the
/// probability of hitting a region of space with the Ray and the cost of the action performed after
/// hit, minimizing this cost.
pub struct KdTree<T: Shape> {
    // the memory layout of this tree is uncommon, check the finalize_rec function doc
    tree: Vec<KdNode>,
    elements: Vec<T>,
    scene_aabb: AABB,
    settings: KdTreeSettings,
}

/// Struct containing some parameters settings to build a KdTree.
#[derive(Copy, Clone)]
struct KdTreeSettings {
    /// Minimum number of shapes contained inside a leaf node.
    leaf_min: u8,
    /// Whether to search for a split in every axis (extensively) or just the longest.
    extensive_search: bool,
    /// Cost of performing an intersection.
    intersect_cost: f32,
    /// Cost of descending to the next node of the tree.
    descend_cost: f32,
}

impl<T: Shape> KdTree<T> {
    /// Creates a KdTree with customized settings.
    ///
    /// The `leaf_min` parameter controls the minimum number of shapes necessary to form a leaf.
    ///
    /// The `intersect_cost` parameter defines the cost of performing an intersection. This value
    /// should be adjusted in relation to the `descend_cost` parameters, defining the cost of
    /// traversing the tree to find the next node.
    ///
    /// The `extensive_search` parameter is used at build time to extend the search time (at
    /// least three times slower) in order to find the tree providing the fastest intersection
    /// speed.
    ///
    /// Note that this is usually not needed as good heuristics are employed in order to build the
    /// best tree without performing an extensive search.
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::geometry::{Point3, Ray, Vec3};
    /// use glaze::shapes::{KdTree, Shape, Sphere};
    ///
    /// let k: KdTree<Sphere> = KdTree::new(1, 1.0, 50.0, true);
    /// // at this point one should call `k.build(...)`, to add shapes.
    /// // this examples shows an empty tree.
    ///
    /// let ray = Ray::new(&Point3::zero(), &Vec3::up());
    /// assert!(!k.intersect_fast(&ray));
    /// ```
    pub fn new(
        leaf_min: u8,
        intersect_cost: f32,
        descend_cost: f32,
        extensive_search: bool,
    ) -> KdTree<T> {
        let settings = KdTreeSettings {
            leaf_min,
            extensive_search,
            intersect_cost,
            descend_cost,
        };
        KdTree {
            tree: Vec::new(),
            elements: Vec::new(),
            scene_aabb: AABB::zero(),
            settings,
        }
    }

    /// Creates a KdTree with default settings.
    ///
    /// The default settings are the following:
    /// - `leaf_min`: `3`,
    /// - `intersect_cost`: `50.0`,
    /// - `descend_cost`: `5.0`,
    /// - `extensive_search`: `false`
    /// # Examples
    /// Basic usage:
    /// ```
    /// use glaze::geometry::{Point3, Ray, Vec3};
    /// use glaze::shapes::{KdTree, Shape, Sphere};
    ///
    /// let k: KdTree<Sphere> = KdTree::default();
    /// // at this point one should call `k.build(...)`, to add shapes.
    /// // this examples shows an empty tree.
    ///
    /// let ray = Ray::new(&Point3::zero(), &Vec3::up());
    /// assert!(!k.intersect_fast(&ray));
    /// ```
    pub fn default() -> KdTree<T> {
        let settings = KdTreeSettings {
            leaf_min: 3,
            extensive_search: false,
            intersect_cost: 50.0,
            descend_cost: 5.0,
        };
        KdTree {
            tree: Vec::new(),
            elements: Vec::new(),
            scene_aabb: AABB::zero(),
            settings,
        }
    }
}

impl<T: Shape> Shape for KdTree<T> {
    #[allow(clippy::float_cmp)] //yep, I REALLY want a strict comparison. It's a corner case.
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        // the idea for this method is :
        // - find the order in which the ray intersects the tree children, front to back. (first the
        //   left child or the right child?)
        // - If the node contains shapes always record the closest intersection to the origin
        //   ray, until the found intersection is closer than the node bounds.
        // - Since we are processing front to back, at this point will be IMPOSSIBLE to find a
        //   closer shape intersection.
        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        let mut found = None;
        if let Some(scene_intersection) = isect(&self.scene_aabb, ray, &inv_dir) {
            let mut best = f32::INFINITY;
            // (kdtree node index, node bounding box front, node bounding box back)
            let mut jobs = vec![(0 as u32, scene_intersection.0, scene_intersection.1)];
            while let Some(node) = jobs.pop() {
                let index = node.0;
                let min_distance = node.1;
                let max_distance = node.2;
                // processing front to back, but current best is less than the node front. exit.
                // this is VERY important to cut down a lot of unneeded attempts.
                if best < min_distance {
                    break;
                }
                match &self.tree[index as usize] {
                    KdNode::Node { split, data } => {
                        let axis = (data & 0xC0000000) >> 30;
                        let origin = ray.origin[axis as u8];
                        let direction = ray.direction[axis as u8];
                        let front;
                        let back;
                        // maintain a front-to-back order of visit
                        let first_left = origin < *split || origin == *split && direction <= 0.0;
                        if first_left {
                            front = index + 1;
                            back = data & 0x3FFFFFFF;
                        } else {
                            front = data & 0x3FFFFFFF;
                            back = index + 1;
                        }
                        let split_distance = (split - origin) * inv_dir[axis as u8];
                        if split_distance > max_distance || split_distance <= 0.0 {
                            // split is intersected after the bounds exit => ray don't enter
                            // the back child
                            jobs.push((front, min_distance, max_distance));
                        } else if split_distance < min_distance {
                            // split is intersected before the bounds start => ray don't enter
                            // the front child
                            jobs.push((back, min_distance, max_distance));
                        } else {
                            jobs.push((back, split_distance, max_distance));
                            jobs.push((front, min_distance, split_distance));
                        }
                    }
                    KdNode::Leaf { offset, count } => {
                        for i in 0..*count {
                            let elem_idx = (*offset + i) as usize;
                            let shape = &self.elements[elem_idx];
                            if let Some(intersection) = shape.intersect(ray) {
                                if intersection.distance < best {
                                    best = intersection.distance;
                                    found = Some(intersection);
                                }
                            }
                        }
                        // not breaking on purpose here! I need to clear out the jobs stack!
                    }
                }
            }
        }
        found
    }

    #[allow(clippy::float_cmp)]
    fn intersect_fast(&self, ray: &Ray) -> bool {
        // mostly identical to the intersect method, but return as soon as the first intersection is
        // found
        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        if let Some(scene_intersection) = isect(&self.scene_aabb, ray, &inv_dir) {
            let mut jobs = vec![(0 as u32, scene_intersection.0, scene_intersection.1)];
            while let Some(node) = jobs.pop() {
                let index = node.0;
                let min_distance = node.1;
                let max_distance = node.2;
                match &self.tree[index as usize] {
                    KdNode::Node { split, data } => {
                        let axis = (data & 0xC0000000) >> 30;
                        let origin = ray.origin[axis as u8];
                        let direction = ray.direction[axis as u8];
                        let front;
                        let back;
                        let first_left = origin < *split || origin == *split && direction <= 0.0;
                        if first_left {
                            front = index + 1;
                            back = data & 0x3FFFFFFF;
                        } else {
                            front = data & 0x3FFFFFFF;
                            back = index + 1;
                        }
                        let split_distance = (split - origin) * inv_dir[axis as u8];
                        if split_distance > max_distance || split_distance <= 0.0 {
                            jobs.push((front, min_distance, max_distance));
                        } else if split_distance < min_distance {
                            jobs.push((back, min_distance, max_distance));
                        } else {
                            jobs.push((back, split_distance, max_distance));
                            jobs.push((front, min_distance, split_distance));
                        }
                    }
                    KdNode::Leaf { offset, count } => {
                        for i in 0..*count {
                            let elem_idx = (*offset + i) as usize;
                            let shape = &self.elements[elem_idx];
                            if shape.intersect_fast(ray) {
                                // in this variant early exit as soon as one intersection is found
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn bounding_box(&self) -> AABB {
        self.scene_aabb
    }
}

impl<T: Shape> Accelerator for KdTree<T> {
    type Item = T;
    fn build(self, elements: Vec<Self::Item>) -> Self {
        if elements.len() > MAX_SHAPES as usize {
            panic!(
                "Too many shapes! Maximum number per kd-tree is {}",
                MAX_SHAPES
            )
        }
        let scene_aabb = elements
            .iter()
            .fold(AABB::zero(), |acc, elem| acc.merge(&elem.bounding_box()));
        let tree = build_rec(elements, scene_aabb, 0, 0, self.settings);
        let compact = finalize_rec(tree, Vec::new(), Vec::new());
        KdTree {
            tree: compact.1,
            elements: compact.2,
            scene_aabb,
            settings: self.settings,
        }
    }
}

/// KdTree node used during construction. This is a canonical tree node with some data, and a left
/// and right pointer to its children.
///
/// Note that the finalized nodes used in the KdTree will have a different structure (for caching
/// reasons).
struct KdBuildNode<T> {
    /// The position in space where to split a node.
    split: f32,
    /// The chosen axis for the split.
    axis: u8,
    /// if the current node is a leaf.
    leaf: bool,
    /// The elements contains in the leaf.
    elements: Option<Vec<T>>,
    /// Left child.
    left: Option<Box<KdBuildNode<T>>>,
    /// Right child.
    right: Option<Box<KdBuildNode<T>>>,
}

/// build the kd tree, non-compact version
/// - elements: vector of Shapes managed by the current kd-tree node
/// - depth: used to keep track of recursion depth. Starts with 0
/// - node_aabb: bounding box of the current kd-tree node
/// - bad: how many bad refines has been made until now. A bad refine is a non-optimal split, in the
/// hope of finding an optimal split later. Starts with 0.
/// - extensive: true if an extensive search among all axes should be performed. Mostly useless, as
/// the longest axis is usually the best option.
/// - cost(intersect, descend): kdtree parameters, check constructor description
/// - leaf_min: minimum number of primitives per node
fn build_rec<T: Shape>(
    elements: Vec<T>,
    node_aabb: AABB,
    depth: usize,
    bad: usize,
    settings: KdTreeSettings,
) -> KdBuildNode<T> {
    if depth == MAX_DEPTH || elements.len() <= settings.leaf_min as usize {
        KdBuildNode {
            split: NAN,
            axis: 0xFF,
            leaf: true,
            elements: Some(elements),
            left: None,
            right: None,
        }
    } else {
        let mut candidates = [Vec::new(), Vec::new(), Vec::new()];
        // how many axes we searched. Bail out at 0. This is not the current axis!
        let mut searching = DIMENSIONS;
        // the current best axis
        let mut best_axis = 0xFF;
        // the current best split (as offset in the sc structure for a given axis)
        let mut best_split = 0xFF;
        // the best cost for splitting (based in term of SAH_INTERSECT and SAH_DESCEND)
        let mut best_cost = f32::INFINITY;
        // cost of arriving here
        let arrival_cost = settings.intersect_cost * elements.len() as f32;
        // how many imperfect splits have been made until now. A split is imperfect if the cost of
        // splitting is higher than the cost of doing nothing.
        let mut expensive_split = bad;
        // total surface of the node
        let total_surface = node_aabb.surface();
        let inv_surface = 1.0 / total_surface;
        // diagonal of the node
        let aabb_diagonal = node_aabb.top - node_aabb.bot;
        // The axis we are searching. Start with the longest which is usually the best
        let mut axis = node_aabb.longest_axis() as usize;

        // this loop searches the best candidate for split
        while searching > 0 {
            // use bounding boxes of shapes to generate all possible splits for this axis
            candidates[axis] = elements
                .iter()
                .enumerate()
                .flat_map(|x| {
                    // not super efficient (I could store aabbs beforehand) but it's construction time
                    let elem_aabb = x.1.bounding_box();
                    // TODO: replace vec! with https://github.com/rust-lang/rust/issues/65798
                    vec![
                        // (split value, bottom_side, shape index)
                        (elem_aabb.bot[axis as u8], true, x.0),
                        (elem_aabb.top[axis as u8], false, x.0),
                    ]
                    .into_iter()
                })
                .collect::<Vec<_>>();
            candidates[axis].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // start from bot (so all shapes are on the top side)
            let mut bot_count = 0 as usize;
            let mut top_count = elements.len();
            for i in 0..2 * elements.len() {
                let top_side = !(candidates[axis][i].1);
                // decrease the amount of elements on top when crossing the top side of the AABB
                if top_side {
                    top_count -= 1;
                }
                // if the candidate is inside the area managed by this node
                if candidates[axis][i].0 > node_aabb.bot[axis as u8]
                    && candidates[axis][i].0 < node_aabb.top[axis as u8]
                {
                    let other_axes = [
                        ((axis + 1) % DIMENSIONS) as u8,
                        ((axis + 2) % DIMENSIONS) as u8,
                    ];
                    // calculate the surface of the AABBs resulting from the current split
                    let area_bot = 2.0
                        * (aabb_diagonal[other_axes[0]] * aabb_diagonal[other_axes[1]]
                            + (candidates[axis][i].0 - node_aabb.bot[axis as u8])
                                * (aabb_diagonal[other_axes[0]] + aabb_diagonal[other_axes[1]]));
                    let area_top = 2.0
                        * (aabb_diagonal[other_axes[0]] * aabb_diagonal[other_axes[1]]
                            + (node_aabb.top[axis as u8] - candidates[axis][i].0)
                                * (aabb_diagonal[other_axes[0]] + aabb_diagonal[other_axes[1]]));
                    // calculate the percentage corresponding to the total area
                    let perc_bot = area_bot * inv_surface;
                    let perc_top = area_top * inv_surface;
                    // add a bonus if there are NO primitives inside (rays instantly discarded)
                    let bonus = if bot_count == 0 || top_count == 0 {
                        1.0
                    } else {
                        0.0
                    };
                    // calculate the cost of descending and intersecting each shape, considering
                    // the size of the area resulting from the split over the total area as the
                    // probability of having to intersect all the shapes inside it
                    let cost = settings.descend_cost
                        + settings.intersect_cost
                            * (1.0 - bonus)
                            * (perc_bot * bot_count as f32 + perc_top * top_count as f32);
                    // record the best cost
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = axis;
                        best_split = i;
                    }
                }
                // increase the amount of elements on bot when the other AABB side is crossed
                if !top_side {
                    bot_count += 1;
                }
            }
            // no decent cost for splitting on this axis or extensive search, try next axis
            if best_axis == 0xFF || settings.extensive_search {
                searching -= 1;
                axis = (axis + 1) % DIMENSIONS;
            } else {
                searching = 0;
            }
        }
        // this split sucks, keep it for now in the hope of finding a very good split later
        if best_cost > arrival_cost {
            expensive_split += 1;
        }
        // time to end the recursion, this split is garbage
        if best_axis == 0xFF
            || (best_cost > 4.0 * arrival_cost && elements.len() < 16)
            || expensive_split > IMPERFECT_SPLIT_TOLERANCE
        {
            KdBuildNode {
                split: NAN,
                axis: 0xFF,
                leaf: true,
                elements: Some(elements),
                left: None,
                right: None,
            }
        } else {
            let split = candidates[best_axis][best_split].0;
            //split is nice, divide the array of shapes into the two children and continue recursion
            let bot_ids = candidates[best_axis]
                .iter()
                .take(best_split) // they are sorted, take only bot part
                .filter(|x| x.1) // consider left side of AABB
                .map(|x| x.2) // keep only the id in the original array
                .collect::<FnvHashSet<_>>();
            let mut bot_elems = Vec::new();
            let mut top_elems = Vec::new();
            for (index, element) in elements.into_iter().enumerate() {
                if bot_ids.contains(&index) {
                    bot_elems.push(element);
                } else {
                    top_elems.push(element);
                }
            }
            // split also the node aabb
            let mut bot_aabb = node_aabb;
            bot_aabb.top[best_axis as u8] = split;
            let mut top_aabb = node_aabb;
            top_aabb.bot[best_axis as u8] = split;
            // continue recursion
            let left = build_rec(bot_elems, bot_aabb, depth + 1, expensive_split, settings);
            let right = build_rec(top_elems, top_aabb, depth + 1, expensive_split, settings);
            KdBuildNode {
                split,
                axis: best_axis as u8,
                leaf: false,
                elements: None,
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
            }
        }
    }
}

/// Opaque structure containing a node for the kd-tree (compact version).
/// The node is either a leaf or a node. The node has an opaque field containing both
/// split axis in the 0xC0000000 bits and the position of the right child in the 0x3FFFFFFF bits.
/// The left child is right after the current one in the wrapping vector.
enum KdNode {
    //data: 0xC0000000 => axis, 0x3FFFFFFF position of other child.
    Node { split: f32, data: u32 },
    Leaf { offset: u32, count: u32 },
}

/// Transforms a tree to a compact version. The compact version is encoded as an array where the
/// left child is right after the current node, and the right child is somewhere else. So the
/// nodes actually tracks only the right child offset in the vector.
///
/// Moreover the leaves do not store shapes but just the offset in the array and the number of
/// consecutive entries, so **order matters**. This compact version is done to fit more nodes in
/// a single cache line and thus enable faster traversal of the tree.
///
/// Input:
/// - node: The current kd-tree node (extended version)
/// - tree: The tree being built
/// - elems: All the elements contained in the kd-tree.
/// Output:
/// - 0: index of the inserted element in the vector (tree)
/// - 1: vector (tree) of nodes
/// - 2: all the shapes manages by this kd-tree
fn finalize_rec<T: Shape>(
    node: KdBuildNode<T>,
    tree: Vec<KdNode>,
    elems: Vec<T>,
) -> (usize, Vec<KdNode>, Vec<T>) {
    let my_index = tree.len();
    let mut retree = tree;
    if node.leaf {
        let elements = node.elements.unwrap();
        let compact = KdNode::Leaf {
            offset: elems.len() as u32,
            count: elements.len() as u32,
        };
        retree.push(compact);
        let retelems = elems.into_iter().chain(elements).collect();
        (my_index, retree, retelems)
    } else {
        let mut compact = KdNode::Node {
            split: node.split,
            data: (node.axis as u32) << 30,
        };
        retree.push(compact);
        let left_res = finalize_rec(*node.left.unwrap(), retree, elems);
        let mut right_res = finalize_rec(*node.right.unwrap(), left_res.1, left_res.2);
        compact = KdNode::Node {
            split: node.split,
            data: (node.axis as u32) << 30 | ((right_res.0 as u32) & 0x3FFFFFFF),
        };
        right_res.0 = my_index;
        right_res.1[my_index] = compact;
        right_res
    }
}

/// Intersects the AABB and returns the entry and exit points. Used heavily by the acceleration
/// structures. The returned tuple contains entry and exit distance from the ray origin.
fn isect(aabb: &AABB, ray: &Ray, inv_dir: &Vec3) -> Option<(f32, f32)> {
    let mut minxt;
    let mut maxxt;
    let minyt;
    let maxyt;
    let minzt;
    let maxzt;
    if ray.direction.x >= 0.0 {
        minxt = (aabb.bot.x - ray.origin.x) * inv_dir.x;
        maxxt = (aabb.top.x - ray.origin.x) * inv_dir.x;
    } else {
        minxt = (aabb.top.x - ray.origin.x) * inv_dir.x;
        maxxt = (aabb.bot.x - ray.origin.x) * inv_dir.x;
    }
    if ray.direction.y >= 0.0 {
        minyt = (aabb.bot.y - ray.origin.y) * inv_dir.y;
        maxyt = (aabb.top.y - ray.origin.y) * inv_dir.y;
    } else {
        minyt = (aabb.top.y - ray.origin.y) * inv_dir.y;
        maxyt = (aabb.bot.y - ray.origin.y) * inv_dir.y;
    }
    if (minyt > maxxt) || (minxt > maxyt) {
        None
    } else {
        if minyt > minxt {
            minxt = minyt;
        }
        if maxyt < maxxt {
            maxxt = maxyt;
        }
        if ray.direction.z >= 0.0 {
            minzt = (aabb.bot.z - ray.origin.z) * inv_dir.z;
            maxzt = (aabb.top.z - ray.origin.z) * inv_dir.z;
        } else {
            minzt = (aabb.top.z - ray.origin.z) * inv_dir.z;
            maxzt = (aabb.bot.z - ray.origin.z) * inv_dir.z;
        }
        if (minzt > maxxt) || (minxt > maxzt) {
            None
        } else {
            if minzt > minxt {
                minxt = minzt;
            }
            if maxzt < maxxt {
                maxxt = maxzt;
            }
            Some((minxt, maxxt))
        }
    }
}
