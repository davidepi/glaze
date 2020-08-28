use crate::geometry::Ray;
use crate::shapes::{Accelerator, Intersection, Shape, AABB};
use fnv::FnvHashSet;
use std::f32::NAN;

/// Minimum number of assets in a leaf
const LEAF_MIN: usize = 1;
/// Cost of descending the tree to the next node
const SAH_DESCEND: f32 = 1.0;
/// Cost of performing an intersection
const SAH_INTERSECT: f32 = 50.0;
/// Maximum depth of the tree
const MAX_DEPTH: usize = 15;
/// Dimensionality of the tree. We are operating in a three dimensional space.
/// Changing this does not mean the kd-tree will work with more dimensions, the shapes are still 3D.
const DIMENSIONS: usize = 3;
/// Maximum amount of descends in the tree that are not convenient in term of cost (in the hope that
/// a better split will be found)
const IMPERFECT_SPLIT_TOLERANCE: usize = 2;
/// Maximum number of primitives supported. Due to how indexing works in KdNode.
const MAX_PRIMITIVES: u32 = 0x3FFFFFFF;

pub struct KdTree<T: Shape> {
    // the memory layout of this tree is uncommon, check the finalize_rec function doc
    tree: Vec<KdNode>,
    elements: Vec<T>,
    extensive: bool,
}

impl<T: Shape> KdTree<T> {
    pub fn new(extensive_search: bool) -> KdTree<T> {
        KdTree {
            tree: Vec::new(),
            elements: Vec::new(),
            extensive: extensive_search,
        }
    }
}

impl<T: Shape> Shape for KdTree<T> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        unimplemented!()
    }

    fn intersect_fast(&self, ray: &Ray) -> bool {
        unimplemented!()
    }

    fn bounding_box(&self) -> AABB {
        unimplemented!()
    }
}

impl<T: Shape> Accelerator for KdTree<T> {
    type Item = T;
    fn build(self, elements: Vec<Self::Item>) -> Self {
        if elements.len() > MAX_PRIMITIVES as usize {
            panic!(
                "Too many primitives! Maximum number per kd-tree is {}",
                MAX_PRIMITIVES
            )
        }
        let scene_aabb = elements
            .iter()
            .fold(AABB::zero(), |acc, elem| acc.merge(&elem.bounding_box()));
        let tree = build_rec(elements, scene_aabb, 0, 0, self.extensive);
        let compact = finalize_rec(tree, Vec::new(), Vec::new());
        KdTree {
            tree: compact.1,
            elements: compact.2,
            extensive: self.extensive,
        }
    }
}

struct KdBuildNode<T> {
    split: f32,
    axis: u8,
    leaf: bool,
    elements: Option<Vec<T>>,
    left: Option<Box<KdBuildNode<T>>>,
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
fn build_rec<T: Shape>(
    elements: Vec<T>,
    node_aabb: AABB,
    depth: usize,
    bad: usize,
    extensive: bool,
) -> KdBuildNode<T> {
    if depth == MAX_DEPTH || elements.len() <= LEAF_MIN {
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
        let arrival_cost = SAH_INTERSECT * elements.len() as f32;
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
                    // calculate the cost of descending and intersecting the primitives, weighted
                    // on how much percentage of the area is occupied by each primitive
                    let cost = SAH_DESCEND
                        + SAH_INTERSECT
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
            if best_axis == 0xFF || extensive {
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
            let left = build_rec(bot_elems, bot_aabb, depth + 1, expensive_split, extensive);
            let right = build_rec(top_elems, top_aabb, depth + 1, expensive_split, extensive);
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
