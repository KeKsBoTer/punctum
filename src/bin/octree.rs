use cgmath::{vec3, EuclideanSpace, Point3};
use las::{Read, Reader};
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::sync::mpsc::channel;
use std::thread;

#[derive(Debug)]
enum Node {
    Group(Box<[Node; 8]>),
    Filled(Vec<Vertex>),
    Empty,
}

impl Node {
    fn insert(
        group: &mut Box<[Node; 8]>,
        point: Vertex,
        center: Point3<f64>,
        size: f64,
        max_node_size: usize,
    ) {
        let (octant_i, new_size, new_center) = Node::child_octant(&point, center, size);
        let mut center = new_center;
        let mut size = new_size;
        let mut node = &mut group[octant_i];
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

    fn traverse<F: FnMut(&Node)>(&self, f: &mut F) {
        f(self);
        if let Node::Group(group) = self {
            for node in group.iter() {
                node.traverse(f);
            }
        }
    }

    fn split(
        vertices: &Vec<Vertex>,
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

    fn child_octant(point: &Vertex, center: Point3<f64>, size: f64) -> (usize, f64, Point3<f64>) {
        let z = (point.position[2] as f64 > center.z) as usize; // 1 if true
        let y = (point.position[1] as f64 > center.y) as usize;
        let x = (point.position[0] as f64 > center.z) as usize;
        let octant_i = 4 * z + 2 * y + x;

        let new_size = size / 2.;
        let new_center =
            center + new_size * 3_f64.sqrt() * vec3(x as f64 - 0.5, y as f64 - 0.5, z as f64 - 0.5);
        return (octant_i, new_size, new_center);
    }
}

#[derive(Debug)]
struct Octree {
    root: Node,
    center: Point3<f64>,
    size: f64,
    max_node_size: usize,
}

impl Octree {
    fn new(center: Point3<f64>, size: f64) -> Self {
        Octree {
            root: Node::Empty,
            max_node_size: 256,
            center,
            size,
        }
    }

    fn insert(&mut self, point: Vertex) {
        match &mut self.root {
            Node::Group(group) => {
                Node::insert(group, point, self.center, self.size, self.max_node_size);
            }
            Node::Filled(data) => {
                if data.len() >= self.max_node_size {
                    let mut group = Node::split(&data, self.center, self.size, self.max_node_size);
                    Node::insert(
                        &mut group,
                        point,
                        self.center,
                        self.size,
                        self.max_node_size,
                    );
                    self.root = Node::Group(group);
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

    fn traverse<F: FnMut(&Node)>(&self, mut f: F) {
        self.root.traverse(&mut f)
    }
}

#[derive(Clone, Copy, Debug)]
struct Vertex {
    pub position: [f64; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

fn main() {
    let mut reader =
        Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las").unwrap();

    let number_of_points = reader.header().number_of_points();
    println!("num points: {}", number_of_points);

    let bounds = reader.header().bounds();
    let min_point = Point3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max_point = Point3::new(bounds.max.x, bounds.max.y, bounds.max.z);
    let size = max_point - min_point;
    let max_size = [size.x, size.y, size.z]
        .into_iter()
        .reduce(|a, b| a.max(b))
        .unwrap();

    let mut octree = Octree::new(min_point.midpoint(max_point), max_size);
    let mut num_inserted = 0;
    let mut rand_gen = StdRng::seed_from_u64(42);
    for (i, p) in reader.points().enumerate() {
        let r: u64 = rand_gen.sample(rand::distributions::Standard);
        if r % 2 != 0 {
            continue;
        }
        let point = p.unwrap();
        let color = point.color.unwrap();
        let point = Vertex {
            position: [point.x, point.y, point.z],
            normal: [0.; 3],
            color: [
                (color.red as f32) / 65536., // = 2**16
                (color.green as f32) / 65536.,
                (color.blue as f32) / 65536.,
                1.,
            ],
        };
        if i == 176283437 {
            println!("now!");
        }
        octree.insert(point);
        num_inserted += 1;
        if i % 100000 == 0 {
            println!("progress: {}%", 100. * (i as f64 / number_of_points as f64));
        }
    }

    println!("done: {:?}", num_inserted);

    // check if all points were inserted into the octree
    let mut check = 0;
    octree.traverse(|node| match node {
        Node::Filled(data) => check += data.len(),
        _ => {}
    });
    assert!(check == num_inserted, "not all points present in octree");
}