use cgmath::{vec3, EuclideanSpace, Point3};
use las::{Read, Reader};
use punctum::Vertex;
use rand::{prelude::StdRng, Rng, SeedableRng};

const NODE_MAX: usize = 512;

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
        gi: usize,
    ) {
        let (octant_i, new_size, new_center) = Node::child_octant(&point, center, size);
        let mut center = new_center;
        let mut size = new_size;
        let mut node = &mut group[octant_i];
        let mut depth = 0;
        loop {
            match node {
                Node::Group(group) => {
                    let (octant_i, new_size, new_center) = Node::child_octant(&point, center, size);
                    center = new_center;
                    size = new_size;
                    node = &mut group[octant_i];
                }
                Node::Filled(data) => {
                    if data.len() == NODE_MAX {
                        *node = Node::Group(Node::split(data, center, size));
                        if depth > 100 {
                            panic!("depth: {} ({})", depth, gi);
                        }
                    } else {
                        data.push(point);
                        return;
                    }
                }
                Node::Empty => {
                    let mut new_vec = Vec::with_capacity(NODE_MAX);
                    new_vec.push(point);
                    *node = Node::Filled(new_vec);
                    return;
                }
            }
            depth += 1;
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

    fn split(vertices: &Vec<Vertex>, center: Point3<f64>, size: f64) -> Box<[Node; 8]> {
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
                    let mut new_vec = Vec::with_capacity(NODE_MAX);
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
}

impl Octree {
    fn new(center: Point3<f64>, size: f64) -> Self {
        Octree {
            root: Node::Empty,
            center,
            size,
        }
    }

    fn insert(&mut self, point: Vertex, gi: usize) {
        match &mut self.root {
            Node::Group(group) => {
                Node::insert(group, point, self.center, self.size, gi);
            }
            Node::Filled(data) => {
                if data.len() >= NODE_MAX {
                    let mut group = Node::split(&data, self.center, self.size);
                    Node::insert(&mut group, point, self.center, self.size, gi);
                    self.root = Node::Group(group);
                } else {
                    data.push(point);
                }
            }
            Node::Empty => {
                let mut new_vec = Vec::with_capacity(NODE_MAX);
                new_vec.push(point);
                self.root = Node::Filled(new_vec);
            }
        }
    }

    fn traverse<F: FnMut(&Node)>(&self, mut f: F) {
        self.root.traverse(&mut f)
    }
}

fn main() {
    let mut reader =
        Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las").unwrap();

    let number_of_points = reader.header().number_of_points();
    println!("num points: {}", number_of_points);

    let bounds = reader.header().bounds();
    let min_point = Point3::new(
        bounds.min.x as f64,
        bounds.min.y as f64,
        bounds.min.z as f64,
    );
    let max_point = Point3::new(
        bounds.max.x as f64,
        bounds.max.y as f64,
        bounds.max.z as f64,
    );
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
            position: [point.x as f32, point.y as f32, point.z as f32],
            normal: [0.; 3],
            color: [
                (color.red as f32) / 65536.,
                (color.green as f32) / 65536.,
                (color.blue as f32) / 65536.,
                1.,
            ],
        };
        if i == 176283437 {
            println!("now!");
        }
        octree.insert(point, i);
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
