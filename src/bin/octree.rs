use cgmath::{vec3, EuclideanSpace, Point3};
use las::{Read, Reader};
use punctum::Vertex;

const NODE_MAX: usize = 256;

#[derive(Debug)]
enum Node {
    Group(Box<[Node; 8]>),
    Filled(Vec<Vertex>),
    Empty,
}

impl Node {
    fn insert(&mut self, point: Vertex, center: Point3<f32>, size: f32) -> Option<Node> {
        match self {
            Node::Group(group) => {
                let (octant_i, new_size, new_center) = Node::child_octant(&point, center, size);

                if let Some(new_octant) = group[octant_i].insert(point, new_center, new_size) {
                    group[octant_i] = new_octant;
                }
                None
            }
            Node::Filled(data) => {
                if data.len() >= NODE_MAX {
                    let mut new_node = Node::Group(Node::split(data, center, size));
                    if new_node.insert(point, center, size).is_some() {
                        panic!("unreachable");
                    }
                    Some(new_node)
                } else {
                    data.push(point);
                    None
                }
            }
            Node::Empty => {
                let mut new_vec = Vec::with_capacity(NODE_MAX);
                new_vec.push(point);
                Some(Node::Filled(new_vec))
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

    fn split(vertices: &Vec<Vertex>, center: Point3<f32>, size: f32) -> Box<[Node; 8]> {
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
            let (octant_i, new_size, new_center) = Node::child_octant(v, center, size);
            if let Some(new_octant) = new_data[octant_i].insert(*v, new_center, new_size) {
                new_data[octant_i] = new_octant;
            }
        }
        return new_data;
    }

    fn child_octant(point: &Vertex, center: Point3<f32>, size: f32) -> (usize, f32, Point3<f32>) {
        let z = (point.position[2] > center.z) as usize; // 1 if true
        let y = (point.position[1] > center.y) as usize;
        let x = (point.position[0] > center.z) as usize;
        let octant_i = 4 * z + 2 * y + x;

        let new_size = size / 2.;
        let new_center =
            center + new_size * 3_f32.sqrt() * vec3(x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5);
        return (octant_i, new_size, new_center);
    }
}

#[derive(Debug)]
struct Octree {
    root: Node,
    center: Point3<f32>,
    size: f32,
}

impl Octree {
    fn new(center: Point3<f32>, size: f32) -> Self {
        Octree {
            root: Node::Empty,
            center,
            size,
        }
    }

    fn insert(&mut self, point: Vertex) {
        if let Some(new_root) = self.root.insert(point, self.center, self.size) {
            self.root = new_root;
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
        bounds.min.x as f32,
        bounds.min.y as f32,
        bounds.min.z as f32,
    );
    let max_point = Point3::new(
        bounds.max.x as f32,
        bounds.max.y as f32,
        bounds.max.z as f32,
    );
    let size = max_point - min_point;
    let max_size = [size.x, size.y, size.z]
        .into_iter()
        .reduce(|a, b| a.max(b))
        .unwrap();

    let mut octree = Octree::new(min_point.midpoint(max_point), max_size);
    let mut num_inserted = 0;
    for (i, p) in reader.points().enumerate() {
        let r: u32 = rand::random();
        if r % 128 != 0 {
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
        octree.insert(point);
        num_inserted += 1;
        if i % 10000 == 0 {
            println!("progress: {}%", 100. * (i as f32 / number_of_points as f32));
        }
    }

    println!("done: {:?}", num_inserted);
    let mut index = 0;
    octree.traverse(|node| {
        println!("{}", index);
        index += 1;
    })
}
