use serde::{Deserialize, Serialize};

use nalgebra::{convert, Point3, Vector3};

use crate::{
    camera::ViewFrustum, vertex::BaseFloat, CubeBoundingBox, PointCloud, SHVertex, Vertex,
};

#[derive(Serialize, Deserialize, Debug)]
pub enum Node<F: BaseFloat> {
    Intermediate(IntermediateNode<F>),
    Leaf(LeafNode<F>),
    Empty,
}

impl<F: BaseFloat> Node<F> {
    fn insert(
        &mut self,
        point: Vertex<F>,
        bbox: CubeBoundingBox<F>,
        id: u64,
        level: u32,
        max_node_size: usize,
    ) -> u64 {
        let mut node = self;
        let mut bbox = bbox;
        let mut octants_created = 0;

        let mut level = level;
        let mut id = id;
        loop {
            match node {
                Node::Intermediate(Octant { data: group, .. }) => {
                    let (octant_i, new_bbox) = Node::child_octant(&point, &bbox);
                    bbox = new_bbox;
                    node = &mut group[octant_i];

                    id = next_id(id, level, octant_i);
                    level += 1;
                }
                Node::Leaf(octant) => {
                    if octant.data.0.len() == max_node_size {
                        // split octant and insert in next loop iteration
                        let new_octants = octant.split(&bbox, level, max_node_size);
                        octants_created += new_octants.filled_octants() - 1;
                        *node = Node::Intermediate(new_octants);
                    } else {
                        octant.insert_point(point);
                        return octants_created;
                    }
                }
                Node::Empty => {
                    *node = Node::Leaf(LeafNode::new(id, point, max_node_size));
                    return octants_created + 1;
                }
            }
        }
    }

    pub fn id(&self) -> Option<u64> {
        match self {
            Node::Intermediate(o) => Some(o.id),
            Node::Leaf(o) => Some(o.id),
            Node::Empty => None,
        }
    }

    fn depth(&self) -> usize {
        let mut max_depth = 0;
        let mut stack = vec![(0, self)];
        while let Some((depth, node)) = stack.pop() {
            match node {
                Node::Intermediate(Octant { data: nodes, .. }) => {
                    for n in nodes.iter() {
                        stack.push((depth + 1, &n));
                    }
                }
                Node::Leaf(_) => {
                    if depth + 1 > max_depth {
                        max_depth = depth + 1;
                    }
                }
                Node::Empty => {}
            }
        }
        return max_depth;
    }

    fn child_octant(point: &Vertex<F>, bbox: &CubeBoundingBox<F>) -> (usize, CubeBoundingBox<F>) {
        let z = (point.position.z > bbox.center.z) as usize; // 1 if true
        let y = (point.position.y > bbox.center.y) as usize;
        let x = (point.position.x > bbox.center.x) as usize;
        let octant_i = 4 * z + 2 * y + x;

        let new_size = bbox.size / convert(2.);
        let offset = Vector3::new(
            convert::<_, F>(x as f64 - 0.5),
            convert::<_, F>(y as f64 - 0.5),
            convert::<_, F>(z as f64 - 0.5),
        );
        let new_center: Point3<F> = bbox.center + offset * new_size;
        return (
            octant_i,
            CubeBoundingBox {
                size: new_size,
                center: new_center,
            },
        );
    }

    /// calculates the center and size of the i-th octant
    /// based on the current center and size of an octant
    pub fn octant_box(i: usize, bbox: &CubeBoundingBox<F>) -> CubeBoundingBox<F> {
        let z = i / 4;
        let y = (i - 4 * z) / 2;
        let x = i % 2;

        let new_size = bbox.size / convert(2.);
        let offset = Vector3::new(
            convert::<_, F>(x as f64 - 0.5),
            convert::<_, F>(y as f64 - 0.5),
            convert::<_, F>(z as f64 - 0.5),
        );
        let new_center: Point3<F> = bbox.center + offset * new_size;
        return CubeBoundingBox {
            center: new_center,
            size: new_size,
        };
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Octant<F: BaseFloat, T> {
    pub id: u64,
    pub sh_rep: SHVertex<F>,
    pub data: T,
}

impl<F: BaseFloat, T> Octant<F, T> {
    pub fn is_leaf(&self) -> bool {
        // check if most significant bit is 1
        // 1 means intermediate node
        self.id & (1 << 63) == 0
    }
}

pub type LeafNode<F> = Octant<F, PointCloud<F>>;

impl<F: BaseFloat> LeafNode<F> {
    fn new(id: u64, point: Vertex<F>, max_node_size: usize) -> Self {
        let sh_approximation = SHVertex::new_with_color(
            point.position,
            convert(1.),
            Vector3::new(
                point.color.x as f32 / 255.,
                point.color.y as f32 / 255.,
                point.color.z as f32 / 255.,
            ),
        );
        let mut new_vec = Vec::with_capacity(max_node_size);
        new_vec.push(point);
        Self {
            id,
            data: new_vec.into(),
            sh_rep: sh_approximation,
        }
    }

    fn insert_point(&mut self, point: Vertex<F>) {
        self.data.0.push(point);
    }

    fn split(
        &self,
        bbox: &CubeBoundingBox<F>,
        level: u32,
        max_node_size: usize,
    ) -> IntermediateNode<F> {
        let mut new_data = Box::new([
            Node::Empty,
            Node::Empty,
            Node::Empty,
            Node::Empty,
            Node::Empty,
            Node::Empty,
            Node::Empty,
            Node::Empty,
        ]);
        for v in self.data.0.iter() {
            let (octant_i, _) = Node::child_octant(v, bbox);
            match &mut new_data[octant_i] {
                Node::Intermediate(_) => panic!("unreachable"),
                Node::Leaf(octant) => octant.insert_point(*v),
                Node::Empty => {
                    new_data[octant_i] = Node::Leaf(LeafNode::new(
                        next_id(self.id, level, octant_i),
                        *v,
                        max_node_size,
                    ));
                }
            }
        }
        return IntermediateNode {
            id: self.id | (1 << 63),
            sh_rep: self.sh_rep, // TODO calculate new sh_rep based on children
            data: new_data,
        };
    }

    pub fn points(&self) -> &PointCloud<F> {
        &self.data
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn sh_rep(&self) -> &SHVertex<F> {
        &self.sh_rep
    }

    pub fn level(&self) -> u32 {
        (u32::BITS - self.id.leading_zeros()) / 3
    }
}

pub type IntermediateNode<F> = Octant<F, Box<[Node<F>; 8]>>;

impl<F: BaseFloat> IntermediateNode<F> {
    pub fn level(&self) -> u32 {
        // set most significant bit to zero
        // for leading_zeros to work correctly
        let id = self.id & !(1 << 63);
        (u32::BITS - id.leading_zeros()) / 3
    }
}

impl<F: BaseFloat> IntermediateNode<F> {
    fn filled_octants(&self) -> u64 {
        self.data
            .iter()
            .map(|c| if let Node::Leaf(_) = c { 1 } else { 0 })
            .sum()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Octree<F: BaseFloat> {
    pub root: Node<F>,
    bbox: CubeBoundingBox<F>,
    max_node_size: usize,

    depth: usize,
    num_points: u64,
    num_octants: u64,
}

impl<F: BaseFloat> Octree<F> {
    pub fn new(center: Point3<F>, size: F, max_node_size: usize) -> Self {
        Octree {
            root: Node::Empty,
            max_node_size,
            bbox: CubeBoundingBox::new(center, size),
            depth: 0,
            num_points: 0,
            num_octants: 0,
        }
    }

    pub fn insert(&mut self, point: Vertex<F>) {
        self.num_points += 1;
        match &mut self.root {
            Node::Intermediate(_) => {
                self.num_octants += self.root.insert(point, self.bbox, 0, 0, self.max_node_size);
            }
            Node::Leaf(octant) => {
                if octant.data.0.len() >= self.max_node_size {
                    let group = octant.split(&self.bbox, 0, self.max_node_size);
                    self.num_octants += group.filled_octants() - 1;
                    self.root = Node::Intermediate(group);

                    self.num_octants +=
                        self.root.insert(point, self.bbox, 0, 0, self.max_node_size);

                    return;
                } else {
                    octant.insert_point(point);
                    return;
                }
            }
            Node::Empty => {
                self.root = Node::Leaf(LeafNode::new(0, point, self.max_node_size));
                self.num_octants = 1;
                return;
            }
        }
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    pub fn num_points(&self) -> u64 {
        self.num_points
    }

    pub fn bbox(&self) -> &CubeBoundingBox<F> {
        &self.bbox
    }

    pub fn num_octants(&self) -> u64 {
        self.num_octants
    }

    pub fn flat_points(&self) -> Vec<Vertex<F>> {
        self.into_iter()
            .flat_map(|octant| octant.data.0.clone())
            .collect()
    }

    pub fn max_node_size(&self) -> usize {
        self.max_node_size
    }

    pub fn into_octant_iterator<'a>(&'a self) -> BBoxOctreeIterator<'a, F> {
        let mut stack = Vec::new();
        stack.push((&self.root, self.bbox));
        BBoxOctreeIterator { stack: stack }
    }

    pub fn visible_octants<'a>(&'a self, frustum: &ViewFrustum<F>) -> Vec<OctreeIter<'a, F>> {
        match &self.root {
            Node::Intermediate(Octant { data: root, .. }) => {
                let mut visible_octants = Vec::new();
                let mut queue = vec![(root, self.bbox)];
                while let Some((node, bbox)) = queue.pop() {
                    if bbox.within_frustum(&frustum) {
                        for (i, child) in node.iter().enumerate() {
                            match child {
                                Node::Intermediate(Octant { data: children, .. }) => {
                                    let bbox_child = Node::<F>::octant_box(i, &bbox);
                                    queue.push((children, bbox_child));
                                }
                                Node::Leaf(child) => {
                                    let bbox_child = Node::<F>::octant_box(i, &bbox);
                                    if bbox_child.within_frustum(&frustum) {
                                        visible_octants.push(OctreeIter {
                                            octant: child,
                                            bbox: bbox_child,
                                        })
                                    }
                                }
                                Node::Empty => {}
                            }
                        }
                    }
                }
                return visible_octants;
            }
            Node::Leaf(octant) => {
                if self.bbox.within_frustum(&frustum) {
                    vec![OctreeIter {
                        octant,
                        bbox: self.bbox,
                    }]
                } else {
                    Vec::new()
                }
            }
            Node::Empty => Vec::new(),
        }
    }

    pub fn get<'a>(&'a self, id: u64) -> Option<&'a Node<F>> {
        let mut node = &self.root;
        let mut level = 0;
        loop {
            match node {
                Node::Intermediate(Octant {
                    id: g_id,
                    data: children,
                    ..
                }) => {
                    if *g_id == id {
                        return Some(node);
                    }
                    let index = (id >> (3 * level)) & 7;
                    node = &children[index as usize];
                    level += 1;
                }
                Node::Leaf(octant) => {
                    if octant.id == id {
                        return Some(node);
                    } else {
                        return None;
                    }
                }
                Node::Empty => return None,
            }
        }
    }

    pub fn get_mut<'a>(&'a mut self, id: u64) -> Option<&'a mut Node<F>> {
        let mut node = &mut self.root;
        let mut level = 0;
        loop {
            let node_id = node.id();
            if node_id.is_some() && node_id.unwrap() == id {
                return Some(node);
            }
            match node {
                Node::Intermediate(octant) => {
                    let index = (octant.id >> (3 * level)) & 7;
                    node = &mut octant.data[index as usize];
                    level += 1;
                }
                _ => return None,
            }
        }
    }

    /// collects the ids of all intermediate nodes
    /// ordered in such a way that parents are always before their
    /// child in the list
    pub fn itermediate_octants(&self) -> Vec<u64> {
        let mut octants = Vec::new();
        let mut stack = vec![&self.root];
        while let Some(node) = stack.pop() {
            match node {
                Node::Intermediate(octant) => {
                    octants.push(octant.id);
                    for child in octant.data.iter() {
                        if let Node::Intermediate(o) = child {
                            octants.push(o.id);
                            stack.push(child);
                        }
                    }
                }
                _ => {}
            }
        }
        return octants;
    }
}

impl Into<Node<f32>> for Node<f64> {
    fn into(self) -> Node<f32> {
        match self {
            Node::Intermediate(node) => Node::Intermediate(IntermediateNode {
                id: node.id,
                sh_rep: node.sh_rep.into(),
                data: Box::new(node.data.map(|octant| octant.into())),
            }),
            Node::Leaf(LeafNode {
                id,
                data: points,
                sh_rep,
            }) => Node::Leaf(LeafNode {
                id: id,
                data: points
                    .0
                    .iter()
                    .map(|p| (*p).into())
                    .collect::<Vec<Vertex<f32>>>()
                    .into(),
                sh_rep: sh_rep.into(),
            }),
            Node::Empty => Node::Empty,
        }
    }
}

impl Into<Octree<f32>> for Octree<f64> {
    fn into(self) -> Octree<f32> {
        Octree {
            root: self.root.into(),
            bbox: CubeBoundingBox {
                center: self.bbox.center.cast(),
                size: self.bbox.size as f32,
            },
            max_node_size: self.max_node_size,
            depth: self.depth,
            num_points: self.num_points,
            num_octants: self.num_octants,
        }
    }
}

pub struct OctreeIteratorMut<'a, F: BaseFloat> {
    stack: Vec<&'a mut Node<F>>,
}

impl<'a, F: BaseFloat> IntoIterator for &'a mut Octree<F> {
    type Item = &'a mut LeafNode<F>;
    type IntoIter = OctreeIteratorMut<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::new();
        stack.push(&mut self.root);
        OctreeIteratorMut { stack: stack }
    }
}

impl<'a, F: BaseFloat> Iterator for OctreeIteratorMut<'a, F> {
    type Item = &'a mut LeafNode<F>;

    fn next(&mut self) -> Option<&'a mut LeafNode<F>> {
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Intermediate(Octant { data: octants, .. }) => {
                    for o in octants.iter_mut() {
                        if let Node::Empty = o {
                        } else {
                            self.stack.push(o);
                        }
                    }
                }
                Node::Leaf(octant) => return Some(octant),
                Node::Empty => {
                    panic!("unreachable")
                }
            }
        }
        None
    }
}

pub struct OctreeIterator<'a, F: BaseFloat> {
    stack: Vec<&'a Node<F>>,
}

impl<'a, F: BaseFloat> IntoIterator for &'a Octree<F> {
    type Item = &'a LeafNode<F>;
    type IntoIter = OctreeIterator<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::new();
        stack.push(&self.root);
        OctreeIterator { stack: stack }
    }
}

impl<'a, F: BaseFloat> Iterator for OctreeIterator<'a, F> {
    type Item = &'a LeafNode<F>;

    fn next(&mut self) -> Option<&'a LeafNode<F>> {
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Intermediate(Octant { data: octants, .. }) => {
                    for o in octants.iter() {
                        if let Node::Empty = o {
                        } else {
                            self.stack.push(o);
                        }
                    }
                }
                Node::Leaf(octant) => return Some(octant),
                Node::Empty => {
                    panic!("unreachable")
                }
            }
        }
        None
    }
}

#[derive(Clone, Copy)]
pub struct OctreeIter<'a, F: BaseFloat> {
    pub octant: &'a LeafNode<F>,
    pub bbox: CubeBoundingBox<F>,
}

pub struct BBoxOctreeIterator<'a, F: BaseFloat> {
    stack: Vec<(&'a Node<F>, CubeBoundingBox<F>)>,
}

impl<'a, F: BaseFloat> Iterator for BBoxOctreeIterator<'a, F> {
    type Item = OctreeIter<'a, F>;

    fn next(&mut self) -> Option<OctreeIter<'a, F>> {
        while let Some((node, bbox)) = self.stack.pop() {
            match node {
                Node::Intermediate(Octant { data: group, .. }) => {
                    for (i, child_node) in group.iter().enumerate() {
                        if let Node::Empty = child_node {
                        } else {
                            let bbox_child = Node::<F>::octant_box(i, &bbox);
                            self.stack.push((child_node, bbox_child));
                        }
                    }
                }
                Node::Leaf(octant) => return Some(OctreeIter { octant, bbox }),
                Node::Empty => {
                    panic!("unreachable")
                }
            }
        }
        None
    }
}

/// get the id for an octant based on its parent id
/// each level needs 3 bits to assign an id to each octant
/// e.g.
/// (id = 0b000010, level = 1, o_id = 4 = 0b100) ==> 0b100010
/// see https://stackoverflow.com/questions/46671557/computing-unique-identification-keys-in-a-quadtree-octree-structure
/// for visualization
///
/// because we use a 64 bit id this limits us to a tree depth of 21
#[inline]
fn next_id(id: u64, level: u32, o_id: usize) -> u64 {
    let new_part = (o_id as u64) << (3 * level);
    id + new_part
}
