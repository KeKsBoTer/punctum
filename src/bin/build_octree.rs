use std::{borrow::BorrowMut, fs::File, io::BufWriter, time::Duration};

use bincode::{serialize_into, serialized_size};
use las::{point::Point, Read as LasRead, Reader};
use nalgebra::{center, distance, distance_squared, Point3, Vector3};
use pbr::ProgressBar;
use ply_rs::parser::Parser;
use punctum::{BaseFloat, CubeBoundingBox, Octree, SHCoefficients, SHVertex, TeeWriter, Vertex};
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use structopt::StructOpt;

fn vertex_flip_yz<F: BaseFloat>(v: Vertex<F>) -> Vertex<F> {
    Vertex {
        position: Point3::new(v.position.x, v.position.z, v.position.y),
        color: v.color,
    }
}

fn build_octree_from_iter<'a, I, F>(points: I, max_node_size: usize, cube_size: F) -> Octree<F>
where
    I: Iterator<Item = Vertex<F>>,
    F: BaseFloat,
{
    let mut octree: Octree<F> = Octree::new(Point3::origin(), cube_size, max_node_size);

    for point in points {
        octree.insert(point.clone());
    }
    return octree;
}

fn sample_points<F>(rate: usize) -> impl FnMut(&Vertex<F>) -> bool
where
    F: BaseFloat,
{
    let mut rand_gen = StdRng::seed_from_u64(42);

    return move |_p: &Vertex<F>| {
        let r: usize = rand_gen.gen();
        return r % rate == 0;
    };
}

fn las_point_to_vertex(point: Point) -> Vertex<f64> {
    let color = point.color.unwrap();
    Vertex {
        position: Point3::new(point.x, point.y, point.z),
        color: Vector3::new(
            (color.red / 256) as u8, // 65536 = 2**16
            (color.green / 256) as u8,
            (color.blue / 256) as u8,
        ),
    }
}

fn normalize_point<F: BaseFloat>(
    bbox: CubeBoundingBox<F>,
    scale_factor: F,
) -> impl Fn(Vertex<F>) -> Vertex<F> {
    return move |v: Vertex<F>| Vertex {
        position: (&v.position - &bbox.center.coords) * scale_factor / bbox.size,
        color: v.color,
    };
}

fn with_progress_bar<T>(total: u64) -> impl FnMut(T) -> T {
    let mut pb = ProgressBar::new(total);
    pb.set_max_refresh_rate(Some(Duration::from_millis(1000)));

    return move |i: T| {
        pb.inc();
        return i;
    };
}

fn from_laz<'a>(
    reader: &'a mut Reader,
) -> (impl Iterator<Item = Vertex<f64>> + 'a, CubeBoundingBox<f64>) {
    let bounds = reader.header().bounds();

    let min_point = Point3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max_point = Point3::new(bounds.max.x, bounds.max.y, bounds.max.z);
    let bb_size = max_point - min_point;
    let max_size = bb_size.amax();
    let center = center(&min_point, &max_point);

    let bbox = CubeBoundingBox::new(center, max_size);

    let iterator = reader.points().map(|p| p.unwrap()).map(las_point_to_vertex);

    return (iterator, bbox);
}

fn build_octree<I, F>(
    points: I,
    bbox: CubeBoundingBox<F>,
    cube_size: F,
    number_of_points: u64,
    max_node_size: usize,
    sample_rate: Option<usize>,
    flip_yz: bool,
) -> Octree<F>
where
    I: Iterator<Item = Vertex<F>>,
    F: BaseFloat,
{
    let mut point_iter: Box<dyn Iterator<Item = Vertex<F>>> = Box::new(
        points
            .map(normalize_point(bbox, cube_size))
            .map(with_progress_bar(number_of_points)),
    );

    if flip_yz {
        point_iter = Box::new(point_iter.map(vertex_flip_yz));
    }
    if let Some(rate) = sample_rate {
        point_iter = Box::new(point_iter.filter(sample_points(rate)));
    }

    let octree = build_octree_from_iter(point_iter, max_node_size, cube_size);

    return octree;
}

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    /// .las or .ply input file
    #[structopt(name = "input_file", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output", parse(from_os_str))]
    output: PathBuf,

    #[structopt(long, default_value = "1024")]
    max_octant_size: usize,

    #[structopt(long)]
    sample_rate: Option<usize>,

    /// flip y and z values
    #[structopt(long)]
    flip_yz: bool,
}

fn main() {
    let opt = Opt::from_args();
    println!(
        "Building octree from {}:",
        opt.input.as_os_str().to_str().unwrap()
    );

    let mut octree = if opt.input.extension().unwrap() == "las" {
        let mut reader = Reader::from_path(opt.input).unwrap();
        let number_of_points = reader.header().number_of_points();

        let (points, bbox) = from_laz(&mut reader);
        build_octree(
            points,
            bbox,
            100.,
            number_of_points,
            opt.max_octant_size,
            opt.sample_rate,
            opt.flip_yz,
        )
    } else {
        let mut ply_file = std::fs::File::open(opt.input).unwrap();
        let p = Parser::<Vertex<f64>>::new();

        let ply = p.read_ply(&mut ply_file).unwrap();

        let points = ply.payload.get("vertex").unwrap();

        let bbox = CubeBoundingBox::from_points(points);

        let number_of_points = points.len() as u64;
        build_octree(
            points.iter().map(|p| *p),
            bbox,
            100.,
            number_of_points,
            opt.max_octant_size,
            opt.sample_rate,
            opt.flip_yz,
        )
    };

    for leaf_node in octree.borrow_mut().into_iter() {
        let pc = leaf_node.points();
        let (centroid, avg_color) = pc.centroid_and_color();

        let radius = leaf_node
            .points()
            .0
            .iter()
            .map(|p| distance_squared(&p.position, &centroid))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        leaf_node.sh_rep.position = centroid;
        leaf_node.sh_rep.radius = radius.sqrt();
        leaf_node.sh_rep.coefficients = SHCoefficients::new_from_color(avg_color.cast() / 255.);
    }

    let intermediate_nodes = octree.itermediate_octants();
    for id in intermediate_nodes.iter().rev() {
        if let punctum::Node::Intermediate(node) = octree
            .get_mut(*id)
            .expect(&format!("cannot find id {:}", id))
        {
            let mut centroid = Point3::origin();
            for child in node.data.iter() {
                match child {
                    punctum::Node::Intermediate(child) => {
                        centroid += child.sh_rep.position.coords;
                    }
                    punctum::Node::Leaf(child) => {
                        centroid += child.sh_rep.position.coords;
                    }
                    punctum::Node::Empty => {}
                }
            }
            centroid /= node.data.len() as f64;
            let mut max_distance = 0.;
            for child in node.data.iter() {
                match child {
                    punctum::Node::Intermediate(child) => {
                        max_distance = f64::max(
                            max_distance,
                            distance(&centroid, &child.sh_rep.position) + child.sh_rep.radius,
                        )
                    }
                    punctum::Node::Leaf(child) => {
                        max_distance = f64::max(
                            max_distance,
                            distance(&centroid, &child.sh_rep.position) + child.sh_rep.radius,
                        )
                    }
                    punctum::Node::Empty => {}
                }
            }
            node.sh_rep = SHVertex::new(
                centroid,
                max_distance,
                SHCoefficients::new_from_color(Vector3::new(1., 0., 0.)),
            );
        } else {
            unreachable!()
        }
    }

    // we check that all ids are unqiue
    // if not the tree is to deep (or something is wrong in the code :P)
    let mut ids = octree
        .into_iter()
        .map(|octant| octant.id())
        .collect::<Vec<u64>>();
    ids.sort_unstable();

    let in_order = ids.iter().zip(ids.iter().skip(1)).find(|(a, b)| **a == **b);
    if let Some(duplicate) = in_order {
        panic!("duplicate id {:}!", duplicate.0);
    }

    for octant in octree.into_octant_iterator() {
        for p in octant.octant.points().0.iter() {
            if !octant.bbox.contains(&p.position) {
                panic!(
                    "point {:?} not contained by its bounding box (min: {:?}, max: {:?})",
                    p,
                    octant.bbox.min_corner(),
                    octant.bbox.max_corner()
                );
            }
        }
        if !octant.bbox.contains(&octant.octant.sh_rep.position) {
            panic!(
                "sh rep {:?} not contained by its bounding box (min: {:?}, max: {:?})",
                octant.octant.sh_rep.position,
                octant.bbox.min_corner(),
                octant.bbox.max_corner()
            );
        }
    }

    println!(
        "octree stats:\n\tnum_points:\t{}\n\tmax_depth:\t{}\n\tnum_octants:\t{}",
        octree.num_points(),
        octree.depth(),
        octree.num_octants()
    );
    println!(
        "writing octree to {}:",
        opt.output.as_os_str().to_str().unwrap()
    );

    {
        let mut pb = ProgressBar::new(serialized_size(&octree).unwrap());
        pb.set_units(pbr::Units::Bytes);

        let out_file = File::create(opt.output).unwrap();
        let mut out_writer = BufWriter::new(&out_file);
        let mut tee = TeeWriter::new(&mut out_writer, &mut pb);
        serialize_into(&mut tee, &octree).unwrap();

        pb.finish_println("done!");
    }
}
