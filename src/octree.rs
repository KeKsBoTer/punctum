use serde::{Deserialize, Serialize};

use nalgebra::{convert, Point3, Vector3};

use crate::{
    camera::ViewFrustum, vertex::BaseFloat, CubeBoundingBox, PointCloud, SHVertex, Vertex,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct Octant<F: BaseFloat> {
    id: u64,
    data: PointCloud<F>,
    pub sh_rep: SHVertex<F>,
}

impl<F: BaseFloat> Octant<F> {
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
    ) -> Box<[Node<F>; 8]> {
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
                Node::Group(_) => panic!("unreachable"),
                Node::Filled(octant) => octant.insert_point(*v),
                Node::Empty => {
                    new_data[octant_i] = Node::Filled(Octant::new(
                        next_id(self.id, level, octant_i),
                        *v,
                        max_node_size,
                    ));
                }
            }
        }
        return new_data;
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

#[derive(Serialize, Deserialize, Debug)]
pub enum Node<F: BaseFloat> {
    Group(Box<[Node<F>; 8]>),
    Filled(Octant<F>),
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
                Node::Group(group) => {
                    let (octant_i, new_bbox) = Node::child_octant(&point, &bbox);
                    bbox = new_bbox;
                    node = &mut group[octant_i];

                    id = next_id(id, level, octant_i);
                    level += 1;
                }
                Node::Filled(octant) => {
                    if octant.data.0.len() == max_node_size {
                        // split octant and insert in next loop iteration
                        let new_octants = octant.split(&bbox, level, max_node_size);
                        octants_created += Node::filled_octants(&new_octants) - 1;
                        *node = Node::Group(new_octants);
                    } else {
                        octant.insert_point(point);
                        return octants_created;
                    }
                }
                Node::Empty => {
                    *node = Node::Filled(Octant::new(id, point, max_node_size));
                    return octants_created + 1;
                }
            }
        }
    }

    fn depth(&self) -> usize {
        let mut max_depth = 0;
        let mut stack = vec![(0, self)];
        while let Some((depth, node)) = stack.pop() {
            match node {
                Node::Group(nodes) => {
                    for n in nodes.iter() {
                        stack.push((depth + 1, &n));
                    }
                }
                Node::Filled(_) => {
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

    fn filled_octants(octants: &Box<[Node<F>; 8]>) -> u64 {
        octants
            .iter()
            .map(|c| if let Node::Filled(_) = c { 1 } else { 0 })
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
            Node::Group(_) => {
                self.num_octants += self.root.insert(point, self.bbox, 0, 0, self.max_node_size);
            }
            Node::Filled(octant) => {
                if octant.data.0.len() >= self.max_node_size {
                    let group = octant.split(&self.bbox, 0, self.max_node_size);
                    self.num_octants += Node::filled_octants(&group) - 1;
                    self.root = Node::Group(group);

                    self.num_octants +=
                        self.root.insert(point, self.bbox, 0, 0, self.max_node_size);

                    return;
                } else {
                    octant.insert_point(point);
                    return;
                }
            }
            Node::Empty => {
                self.root = Node::Filled(Octant::new(0, point, self.max_node_size));
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
            Node::Group(root) => {
                let mut visible_octants = Vec::new();
                let mut queue = vec![(root, self.bbox)];
                while let Some((node, bbox)) = queue.pop() {
                    if bbox.within_frustum(&frustum) {
                        for (i, child) in node.iter().enumerate() {
                            match child {
                                Node::Group(children) => {
                                    let bbox_child = Node::<F>::octant_box(i, &bbox);
                                    queue.push((children, bbox_child));
                                }
                                Node::Filled(child) => {
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
            Node::Filled(octant) => {
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

    pub fn get<'a>(&'a self, id: u64) -> Option<&'a Octant<F>> {
        let mut node = &self.root;
        let mut level = 0;
        loop {
            match node {
                Node::Group(children) => {
                    let index = (id >> (3 * level)) & 7;
                    node = &children[index as usize];
                    level += 1;
                }
                Node::Filled(octant) => {
                    if octant.id == id {
                        return Some(octant);
                    } else {
                        return None;
                    }
                }
                Node::Empty => return None,
            }
        }
    }

    pub fn get_mut<'a>(&'a mut self, id: u64) -> Option<&'a mut Octant<F>> {
        let mut node = &mut self.root;
        let mut level = 0;
        loop {
            match node {
                Node::Group(children) => {
                    let index = (id >> (3 * level)) & 7;
                    node = &mut children[index as usize];
                    level += 1;
                }
                Node::Filled(octant) => {
                    if octant.id == id {
                        return Some(octant);
                    } else {
                        return None;
                    }
                }
                Node::Empty => return None,
            }
        }
    }
}

impl Into<Node<f32>> for Node<f64> {
    fn into(self) -> Node<f32> {
        match self {
            Node::Group(octants) => Node::Group(Box::new(octants.map(|octant| octant.into()))),
            Node::Filled(Octant {
                id,
                data: points,
                sh_rep,
            }) => Node::Filled(Octant {
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
    type Item = &'a mut Octant<F>;
    type IntoIter = OctreeIteratorMut<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::new();
        stack.push(&mut self.root);
        OctreeIteratorMut { stack: stack }
    }
}

impl<'a, F: BaseFloat> Iterator for OctreeIteratorMut<'a, F> {
    type Item = &'a mut Octant<F>;

    fn next(&mut self) -> Option<&'a mut Octant<F>> {
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Group(octants) => {
                    for o in octants.iter_mut() {
                        if let Node::Empty = o {
                        } else {
                            self.stack.push(o);
                        }
                    }
                }
                Node::Filled(octant) => return Some(octant),
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
    type Item = &'a Octant<F>;
    type IntoIter = OctreeIterator<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        let mut stack = Vec::new();
        stack.push(&self.root);
        OctreeIterator { stack: stack }
    }
}

impl<'a, F: BaseFloat> Iterator for OctreeIterator<'a, F> {
    type Item = &'a Octant<F>;

    fn next(&mut self) -> Option<&'a Octant<F>> {
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Group(octants) => {
                    for o in octants.iter() {
                        if let Node::Empty = o {
                        } else {
                            self.stack.push(o);
                        }
                    }
                }
                Node::Filled(octant) => return Some(octant),
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
    pub octant: &'a Octant<F>,
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
                Node::Group(group) => {
                    for (i, child_node) in group.iter().enumerate() {
                        if let Node::Empty = child_node {
                        } else {
                            let bbox_child = Node::<F>::octant_box(i, &bbox);
                            self.stack.push((child_node, bbox_child));
                        }
                    }
                }
                Node::Filled(octant) => return Some(OctreeIter { octant, bbox }),
                Node::Empty => {
                    panic!("unreachable")
                }
            }
        }
        None
    }
}
