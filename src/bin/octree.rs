use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{Read, Write},
};

use las::{Read as LasRead, Reader};
use nalgebra::{center, vector, Point3, Vector3, Vector4};
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{PointCloud, PointPosition};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
enum Node {
    Group(Box<[Node; 8]>),
    Filled(Vec<Vertex>),
    Empty,
}

impl Node {
    fn insert(&mut self, point: Vertex, center: Point3<f64>, size: f64, max_node_size: usize) {
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

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Octree {
    root: Node,
    center: Point3<f64>,
    size: f64,
    max_node_size: usize,

    depth: usize,
}

impl Octree {
    fn new(center: Point3<f64>, size: f64) -> Self {
        Octree {
            root: Node::Empty,
            max_node_size: 1024,
            center,
            size,
            depth: 0,
        }
    }

    fn insert(&mut self, point: Vertex) {
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

    fn traverse<F: FnMut(&Node, Point3<f64>, f64)>(&self, mut f: F) {
        self.root.traverse(&mut f, self.center, self.size)
    }

    fn depth(&self) -> usize {
        self.root.depth()
    }
}

#[derive(Serialize, Deserialize, PartialEq, Clone, Copy, Debug)]
struct Vertex {
    position: Point3<f64>,
    color: Vector4<u8>,
}

impl PointPosition<f64> for Vertex {
    #[inline]
    fn position(&self) -> &Point3<f64> {
        &self.position
    }
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

    let mut octree = Octree::new(center(&min_point, &max_point), max_size);
    let mut num_inserted = 0;
    for (i, p) in reader.points().enumerate() {
        let point = p.unwrap();
        let color = point.color.unwrap();
        let point = Vertex {
            position: Point3::new(point.x, point.y, point.z),
            color: Vector4::new(
                (color.red / 256) as u8,
                (color.green / 256) as u8,
                (color.blue / 256) as u8,
                255,
            ),
        };
        octree.insert(point);
        num_inserted += 1;
        if i % 100000 == 0 {
            println!("progress: {}%", 100. * (i as f64 / number_of_points as f64));
        }
        if num_inserted > 100000 {
            break;
        }
    }

    let encoded: Vec<u8> = bincode::serialize(&octree).unwrap();
    let mut out_file = File::create("octree.bin").unwrap();
    out_file.write_all(&encoded).unwrap();

    println!("done: {:?}", num_inserted);

    let mut in_file = File::open("octree.bin").unwrap();

    let mut buffer = Vec::with_capacity(in_file.metadata().unwrap().len() as usize);
    in_file.read_to_end(&mut buffer).unwrap();
    let decoded: Octree = bincode::deserialize(&buffer).unwrap();

    println!("decoded octree file!");

    println!("depth: {}", decoded.depth());

    let w = Writer::<punctum::Vertex<f32>>::new();

    let mut counter = 0;
    octree.traverse(move |node, center, size| {
        if let Node::Filled(data) = node {
            let data_32 = data
                .iter()
                .map(|v| punctum::Vertex {
                    position: ((v.position - center.coords) / size).cast(),
                    normal: Vector3::zeros(),
                    color: v.color.cast() / 255.,
                })
                .collect();

            let mut pc = PointCloud::from_vec(&data_32);
            pc.scale_to_unit_sphere();
            counter += 1;

            let mut file =
                File::create(format!("dataset/neuschwanstein/octant_{}.ply", counter)).unwrap();

            let mut ply = Ply::<punctum::Vertex<f32>>::new();
            let mut elm_def = punctum::Vertex::<f32>::element_def("vertex".to_string());
            elm_def.count = data_32.len();
            ply.header.encoding = Encoding::BinaryLittleEndian;
            ply.header.elements.add(elm_def.clone());

            w.write_header(&mut file, &ply.header).unwrap();
            w.write_payload_of_element(&mut file, &data_32, &elm_def, &ply.header)
                .unwrap();
        }
    });
}
