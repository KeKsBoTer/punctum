use serde::{Deserialize, Serialize};

use nalgebra::{convert, Point3, Vector3};

use crate::{
    vertex::{BaseColor, BaseFloat},
    Vertex,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct Octant<F: BaseFloat, C: BaseColor> {
    id: u64,
    data: Vec<Vertex<F, C>>,
}

impl<F: BaseFloat, C: BaseColor> Octant<F, C> {
    fn new(id: u64, data: Vec<Vertex<F, C>>) -> Self {
        Self { id, data }
    }

    fn split(
        &self,
        center: Point3<F>,
        size: F,
        level: u32,
        max_node_size: usize,
    ) -> Box<[Node<F, C>; 8]> {
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
        for v in self.data.iter() {
            let (octant_i, _, _) = Node::child_octant(v, center, size);
            match &mut new_data[octant_i] {
                Node::Group(_) => panic!("unreachable"),
                Node::Filled(Octant { id: _, data }) => data.push(*v),
                Node::Empty => {
                    let mut new_vec = Vec::with_capacity(max_node_size);
                    new_vec.push(*v);
                    new_data[octant_i] =
                        Node::Filled(Octant::new(next_id(self.id, level, octant_i), new_vec));
                }
            }
        }
        return new_data;
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
pub enum Node<F: BaseFloat, C: BaseColor> {
    Group(Box<[Node<F, C>; 8]>),
    Filled(Octant<F, C>),
    Empty,
}

impl<F: BaseFloat, C: BaseColor> Node<F, C> {
    fn insert(
        &mut self,
        point: Vertex<F, C>,
        center: Point3<F>,
        size: F,
        id: u64,
        level: u32,
        max_node_size: usize,
    ) -> usize {
        let mut node = self;
        let mut center = center;
        let mut size = size;
        let mut octants_created = 0;

        let mut level = level;
        let mut id = id;
        loop {
            match node {
                Node::Group(group) => {
                    let (octant_i, new_size, new_center) = Node::child_octant(&point, center, size);
                    center = new_center;
                    size = new_size;
                    node = &mut group[octant_i];

                    id = next_id(id, level, octant_i);
                    level += 1;
                }
                Node::Filled(octant) => {
                    if octant.data.len() == max_node_size {
                        // split octant and insert in next loop iteration
                        let new_octants = octant.split(center, size, level, max_node_size);
                        octants_created += Node::filled_octants(&new_octants) - 1;
                        *node = Node::Group(new_octants);
                    } else {
                        octant.data.push(point);
                        return octants_created;
                    }
                }
                Node::Empty => {
                    let mut new_vec = Vec::with_capacity(max_node_size);
                    new_vec.push(point);
                    *node = Node::Filled(Octant::new(id, new_vec));
                    return octants_created + 1;
                }
            }
        }
    }

    fn traverse<'a, A: FnMut(u64, &'a Vec<Vertex<F, C>>, Point3<F>, F)>(
        &'a self,
        f: &mut A,
        center: Point3<F>,
        size: F,
    ) {
        match self {
            Node::Group(group) => {
                for (i, node) in group.iter().enumerate() {
                    let (center_new, size_new) = Self::octant_box(i, center, size);
                    node.traverse(f, center_new, size_new);
                }
            }
            Node::Filled(Octant { id, data }) => f(*id, data, center, size),
            Node::Empty => {}
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

    fn child_octant(point: &Vertex<F, C>, center: Point3<F>, size: F) -> (usize, F, Point3<F>) {
        let z = (point.position.z > center.z) as usize; // 1 if true
        let y = (point.position.y > center.y) as usize;
        let x = (point.position.x > center.x) as usize;
        let octant_i = 4 * z + 2 * y + x;

        let new_size = size / convert(2.);
        let offset = Vector3::new(
            convert::<_, F>(x as f64 - 0.5),
            convert::<_, F>(y as f64 - 0.5),
            convert::<_, F>(z as f64 - 0.5),
        );
        let sqrt3: F = convert::<_, F>(3.).sqrt();
        let new_center: Point3<F> = center + offset * new_size * sqrt3;
        return (octant_i, new_size, new_center);
    }

    /// calculates the center and size of the i-th octant
    /// based on the current center and size of an octant
    fn octant_box(i: usize, center: Point3<F>, size: F) -> (Point3<F>, F) {
        let z = i / 4;
        let y = (i - 4 * z) / 2;
        let x = i % 2;

        let new_size = size / convert(2.);
        let offset = Vector3::new(
            convert::<_, F>(x as f64 - 0.5),
            convert::<_, F>(y as f64 - 0.5),
            convert::<_, F>(z as f64 - 0.5),
        );
        let sqrt3: F = convert::<_, F>(3.).sqrt();
        let new_center: Point3<F> = center + offset * new_size * sqrt3;
        return (new_center, new_size);
    }

    fn filled_octants(octants: &Box<[Node<F, C>; 8]>) -> usize {
        octants
            .iter()
            .map(|c| if let Node::Filled(_) = c { 1 } else { 0 })
            .sum()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Octree<F: BaseFloat, C: BaseColor> {
    pub root: Node<F, C>,
    center: Point3<F>,
    size: F,
    max_node_size: usize,

    depth: usize,
    num_points: u64,
}

impl<F: BaseFloat, C: BaseColor> Octree<F, C> {
    pub fn new(center: Point3<F>, size: F, max_node_size: usize) -> Self {
        Octree {
            root: Node::Empty,
            max_node_size,
            center,
            size,
            depth: 0,
            num_points: 0,
        }
    }

    pub fn insert(&mut self, point: Vertex<F, C>) -> usize {
        self.num_points += 1;
        match &mut self.root {
            Node::Group(_) => {
                return self
                    .root
                    .insert(point, self.center, self.size, 0, 0, self.max_node_size);
            }
            Node::Filled(octant) => {
                if octant.data.len() >= self.max_node_size {
                    let group = octant.split(self.center, self.size, 0, self.max_node_size);
                    let mut new_octants = Node::filled_octants(&group) - 1;
                    self.root = Node::Group(group);

                    new_octants +=
                        self.root
                            .insert(point, self.center, self.size, 0, 0, self.max_node_size);

                    return new_octants;
                } else {
                    octant.data.push(point);
                    return 0;
                }
            }
            Node::Empty => {
                let mut new_vec = Vec::with_capacity(self.max_node_size);
                new_vec.push(point);
                self.root = Node::Filled(Octant::new(0, new_vec));
                return 1;
            }
        }
    }

    pub fn traverse<'a, A: FnMut(u64, &'a Vec<Vertex<F, C>>, Point3<F>, F)>(&'a self, mut f: A) {
        self.root.traverse(&mut f, self.center, self.size)
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    pub fn num_points(&self) -> u64 {
        self.num_points
    }

    pub fn size(&self) -> F {
        self.size
    }
    pub fn center(&self) -> &Point3<F> {
        &self.center
    }

    pub fn num_octants(&self) -> u64 {
        let mut num_octants = 0;
        self.traverse(|_, _, _, _| num_octants += 1);
        return num_octants;
    }

    pub fn flat_points(&self) -> Vec<Vertex<F, C>> {
        self.into_iter()
            .flat_map(|octant| octant.data.clone())
            .collect()
    }

    pub fn max_node_size(&self) -> usize {
        self.max_node_size
    }
}

#[derive(Clone, Copy)]
pub struct OctreeIter<'a, F: BaseFloat, C: BaseColor> {
    pub data: &'a Vec<Vertex<F, C>>,
    pub center: Point3<F>,
    pub size: F,
    pub id: u64,
}

impl<'a, F: BaseFloat, C: BaseColor> IntoIterator for &'a Octree<F, C> {
    type Item = OctreeIter<'a, F, C>;
    type IntoIter = std::vec::IntoIter<OctreeIter<'a, F, C>>;

    fn into_iter(self) -> Self::IntoIter {
        let mut result = vec![];
        self.traverse(|id, data, center, size| {
            result.push(OctreeIter {
                data,
                center,
                size,
                id,
            })
        });

        result.into_iter()
    }
}

impl Into<Vec<Vertex<f32, f32>>> for Octree<f64, u8> {
    fn into(self) -> Vec<Vertex<f32, f32>> {
        self.into_iter()
            .flat_map(|octant| {
                octant
                    .data
                    .iter()
                    .map(|p| (*p).into())
                    .collect::<Vec<Vertex<f32, f32>>>()
            })
            .collect()
    }
}

impl Into<Node<f32, f32>> for Node<f64, u8> {
    fn into(self) -> Node<f32, f32> {
        match self {
            Node::Group(octants) => Node::Group(Box::new(octants.map(|octant| octant.into()))),
            Node::Filled(Octant { id, data: points }) => Node::Filled(Octant::new(
                id,
                points
                    .iter()
                    .map(|p| (*p).into())
                    .collect::<Vec<Vertex<f32, f32>>>(),
            )),
            Node::Empty => Node::Empty,
        }
    }
}

impl Into<Octree<f32, f32>> for Octree<f64, u8> {
    fn into(self) -> Octree<f32, f32> {
        Octree {
            root: self.root.into(),
            center: self.center.cast(),
            size: self.size as f32,
            max_node_size: self.max_node_size,
            depth: self.depth,
            num_points: self.num_points,
        }
    }
}
