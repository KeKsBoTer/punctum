use cgmath::Point3;
use las::{Read, Reader};
use punctum::{BoundingBox, Vertex};

const NODE_MAX: usize = 256;

#[derive(Debug)]
enum Node {
    Leaf(Octant<Vec<Vertex>>),
    Itermediate(Octant<[Box<Node>; 8]>),
}

impl Node {
    fn bbox(&self) -> &BoundingBox {
        match self {
            Node::Leaf(octant) => &octant.bbox,
            Node::Itermediate(octant) => &octant.bbox,
        }
    }

    fn insert(&mut self, point: Vertex) -> Option<Octant<[Box<Node>; 8]>> {
        match self {
            Node::Leaf(octant) => {
                if octant.data.len() < NODE_MAX {
                    octant.data.push(point);
                    return None;
                } else {
                    println!("split occured");
                    return Some(octant.split());
                }
            }
            Node::Itermediate(octant) => {
                let target_i = octant
                    .data
                    .iter()
                    .enumerate()
                    .find_map(|(i, node)| {
                        if node.bbox().contains(point.position.into()) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .unwrap();
                if let Some(new_octant) = octant.data[target_i].insert(point) {
                    octant.data[target_i] = Box::new(Node::Itermediate(new_octant));
                }
                return None;
            }
        }
    }
}

#[derive(Debug)]
struct Octant<T> {
    data: T,
    bbox: BoundingBox,
}

fn nth_bbox(bbox: &BoundingBox, n: u8) -> BoundingBox {
    let center = bbox.center();
    match n {
        0 => BoundingBox::new(center, Point3::new(bbox.min.x, bbox.max.y, bbox.min.z)),
        1 => BoundingBox::new(center, Point3::new(bbox.max.x, bbox.max.y, bbox.min.z)),

        2 => BoundingBox::new(center, Point3::new(bbox.min.x, bbox.min.y, bbox.min.z)),
        3 => BoundingBox::new(center, Point3::new(bbox.max.x, bbox.min.y, bbox.min.z)),

        4 => BoundingBox::new(center, Point3::new(bbox.min.x, bbox.max.y, bbox.max.z)),
        5 => BoundingBox::new(center, Point3::new(bbox.max.x, bbox.max.y, bbox.max.z)),

        6 => BoundingBox::new(center, Point3::new(bbox.min.x, bbox.min.y, bbox.max.z)),
        7 => BoundingBox::new(center, Point3::new(bbox.max.x, bbox.min.y, bbox.max.z)),
        _ => panic!("0 <= n < 8"),
    }
}

impl Octant<Vec<Vertex>> {
    fn split(&self) -> Octant<[Box<Node>; 8]> {
        let data: [Box<Node>; 8] = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
            Box::new(Node::Leaf(Octant {
                data: Vec::new(),
                bbox: nth_bbox(&self.bbox, i),
            }))
        });
        Octant {
            data: data,
            bbox: self.bbox,
        }
    }
}

#[derive(Debug)]
struct Octree {
    root: Node,
}

impl Octree {
    fn new(bbox: BoundingBox) -> Self {
        Self {
            root: Node::Leaf(Octant {
                data: Vec::new(),
                bbox: bbox,
            }),
        }
    }

    fn insert(&mut self, point: Vertex) {
        let node = &mut self.root;
        match node {
            Node::Leaf(octant) => {
                if octant.data.len() < NODE_MAX {
                    octant.data.push(point);
                } else {
                    println!("split occured");
                    self.root = Node::Itermediate(octant.split());
                }
            }
            Node::Itermediate(octant) => {
                let center = octant.bbox.center();
                let target = octant
                    .data
                    .iter_mut()
                    .find(|node| node.bbox().contains(point.position.into()))
                    .unwrap();
            }
        }
    }

    fn bbox(&self) -> &BoundingBox {
        match &self.root {
            Node::Leaf(octant) => &octant.bbox,
            Node::Itermediate(octant) => &octant.bbox,
        }
    }
}

fn main() {
    let mut reader =
        Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las").unwrap();

    let number_of_points = reader.header().number_of_points();
    println!("num points: {}", number_of_points);

    let bounds = reader.header().bounds();

    let mut octree = Octree {
        root: Node::Leaf(Octant {
            data: Vec::new(),
            bbox: BoundingBox {
                min: Point3::new(
                    bounds.min.x as f32,
                    bounds.min.y as f32,
                    bounds.min.z as f32,
                ),
                max: Point3::new(
                    bounds.max.x as f32,
                    bounds.max.y as f32,
                    bounds.max.z as f32,
                ),
            },
        }),
    };
    for p in reader.points() {
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
    }
    // println!("done: {:?}", octree);
}
