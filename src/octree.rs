use serde::{Deserialize, Serialize};

use nalgebra::{vector, Point3};

use crate::Vertex;

#[derive(Serialize, Deserialize, Debug)]
pub enum Node {
    Group(Box<[Node; 8]>),
    Filled(Vec<Vertex<f64>>),
    Empty,
}

impl Node {
    fn insert(&mut self, point: Vertex<f64>, center: Point3<f64>, size: f64, max_node_size: usize) {
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

    fn traverse<F: FnMut(&Node, Point3<f64>, f64)>(
        &self,
        f: &mut F,
        center: Point3<f64>,
        size: f64,
    ) {
        f(self, center, size);
        if let Node::Group(group) = self {
            for (i, node) in group.iter().enumerate() {
                let (center_new, size_new) = Node::octant_box(i, center, size);
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
        vertices: &Vec<Vertex<f64>>,
        center: Point3<f64>,
        size: f64,
        max_node_size: usize,
    ) -> Box<[Node; 8]> {
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

    fn child_octant(
        point: &Vertex<f64>,
        center: Point3<f64>,
        size: f64,
    ) -> (usize, f64, Point3<f64>) {
        let z = (point.position[2] as f64 > center.z) as usize; // 1 if true
        let y = (point.position[1] as f64 > center.y) as usize;
        let x = (point.position[0] as f64 > center.z) as usize;
        let octant_i = 4 * z + 2 * y + x;

        let new_size = size / 2.;
        let new_center = center
            + new_size * 3_f64.sqrt() * vector!(x as f64 - 0.5, y as f64 - 0.5, z as f64 - 0.5);
        return (octant_i, new_size, new_center);
    }

    fn octant_box(i: usize, center: Point3<f64>, size: f64) -> (Point3<f64>, f64) {
        let z = i / 4;
        let y = (i - 4 * z) / 2;
        let x = i % 2;
        let new_size = size / 2.;
        let new_center = center
            + new_size * 3_f64.sqrt() * vector!(x as f64 - 0.5, y as f64 - 0.5, z as f64 - 0.5);
        return (new_center, new_size);
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Octree {
    root: Node,
    center: Point3<f64>,
    size: f64,
    max_node_size: usize,

    depth: usize,
}

impl Octree {
    pub fn new(center: Point3<f64>, size: f64) -> Self {
        Octree {
            root: Node::Empty,
            max_node_size: 1024,
            center,
            size,
            depth: 0,
        }
    }

    pub fn insert(&mut self, point: Vertex<f64>) {
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
    }

    pub fn traverse<F: FnMut(&Node, Point3<f64>, f64)>(&self, mut f: F) {
        self.root.traverse(&mut f, self.center, self.size)
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }
}
