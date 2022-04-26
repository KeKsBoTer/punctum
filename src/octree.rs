use serde::{Deserialize, Serialize};

use nalgebra::{convert, Point3, Vector3};

use crate::{
    vertex::{BaseColor, BaseFloat},
    Vertex,
};

#[derive(Serialize, Deserialize, Debug)]
pub enum Node<F: BaseFloat, C: BaseColor> {
    Group(Box<[Node<F, C>; 8]>),
    Filled(Vec<Vertex<F, C>>),
    Empty,
}

impl<F: BaseFloat, C: BaseColor> Node<F, C> {
    fn insert(&mut self, point: Vertex<F, C>, center: Point3<F>, size: F, max_node_size: usize) {
        let mut node = self;
        let mut center = center;
        let mut size = size;
        loop {
            match node {
                Node::Group(group) => {
                    let (octant_i, new_size, new_center) = Node::child_octant(&point, center, size);
                    center = new_center;
                    size = new_size;
                    node = &mut group[octant_i];
                }
                Node::Filled(data) => {
                    if data.len() == max_node_size {
                        *node = Node::Group(Node::split(data, center, size, max_node_size));
                    } else {
                        data.push(point);
                        return;
                    }
                }
                Node::Empty => {
                    let mut new_vec = Vec::with_capacity(max_node_size);
                    new_vec.push(point);
                    *node = Node::Filled(new_vec);
                    return;
                }
            }
        }
    }

    fn traverse<A: FnMut(&Node<F, C>, Point3<F>, F)>(&self, f: &mut A, center: Point3<F>, size: F) {
        f(self, center, size);
        if let Node::Group(group) = self {
            for (i, node) in group.iter().enumerate() {
                let (center_new, size_new) = Self::octant_box(i, center, size);
                node.traverse(f, center_new, size_new);
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

    fn split(
        vertices: &Vec<Vertex<F, C>>,
        center: Point3<F>,
        size: F,
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
        for v in vertices {
            let (octant_i, _, _) = Node::child_octant(v, center, size);
            match &mut new_data[octant_i] {
                Node::Group(_) => panic!("unreachable"),
                Node::Filled(data) => data.push(*v),
                Node::Empty => {
                    let mut new_vec = Vec::with_capacity(max_node_size);
                    new_vec.push(*v);
                    new_data[octant_i] = Node::Filled(new_vec);
                }
            }
        }
        return new_data;
    }

    fn child_octant(point: &Vertex<F, C>, center: Point3<F>, size: F) -> (usize, F, Point3<F>) {
        let z = (point.position[2] > center.z) as usize; // 1 if true
        let y = (point.position[1] > center.y) as usize;
        let x = (point.position[0] > center.z) as usize;
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
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Octree<F: BaseFloat, C: BaseColor> {
    root: Node<F, C>,
    center: Point3<F>,
    size: F,
    max_node_size: usize,

    depth: usize,
    num_points: u64,
}

impl<F: BaseFloat, C: BaseColor> Octree<F, C> {
    pub fn new(center: Point3<F>, size: F) -> Self {
        Octree {
            root: Node::Empty,
            max_node_size: 1024,
            center,
            size,
            depth: 0,
            num_points: 0,
        }
    }

    pub fn insert(&mut self, point: Vertex<F, C>) {
        match &mut self.root {
            Node::Group(_) => {
                self.root
                    .insert(point, self.center, self.size, self.max_node_size);
            }
            Node::Filled(data) => {
                if data.len() >= self.max_node_size {
                    let group = Node::split(&data, self.center, self.size, self.max_node_size);
                    self.root = Node::Group(group);
                    self.root
                        .insert(point, self.center, self.size, self.max_node_size);
                } else {
                    data.push(point);
                }
            }
            Node::Empty => {
                let mut new_vec = Vec::with_capacity(self.max_node_size);
                new_vec.push(point);
                self.root = Node::Filled(new_vec);
            }
        }
        self.num_points += 1;
    }

    pub fn traverse<A: FnMut(&Node<F, C>, Point3<F>, F)>(&self, mut f: A) {
        self.root.traverse(&mut f, self.center, self.size)
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    pub fn num_points(&self) -> u64 {
        self.num_points
    }
    pub fn num_octants(&self) -> u64 {
        let mut num_octants = 0;
        self.traverse(|_, _, _| num_octants += 1);
        return num_octants;
    }
}