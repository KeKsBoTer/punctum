use std::{
    borrow::BorrowMut,
    collections::HashMap,
    fs::File,
    io::BufWriter,
    time::{Duration, Instant},
};

use bincode::{serialize_into, serialized_size};
use las::{point::Point, Read as LasRead, Reader};
use nalgebra::{center, distance, distance_squared, Point3, Vector3};
use pbr::ProgressBar;
use ply_rs::parser::Parser;
use punctum::{
    load_cameras, merge_shs, BaseFloat, CubeBoundingBox, Octree, OrthographicProjection,
    PointCloud, SHCoefficients, SHVertex, TeeWriter, Vertex,
};
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use structopt::StructOpt;
use tch::{kind, Tensor};

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
    // use fixed seed for reproducability
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
    with_pb: bool,
) -> Octree<F>
where
    I: Iterator<Item = Vertex<F>>,
    F: BaseFloat,
{
    let mut point_iter: Box<dyn Iterator<Item = Vertex<F>>> =
        Box::new(points.map(normalize_point(bbox, cube_size)));

    if with_pb {
        point_iter = Box::new(point_iter.map(with_progress_bar(number_of_points)));
    }

    if flip_yz {
        point_iter = Box::new(point_iter.map(vertex_flip_yz));
    }
    if let Some(rate) = sample_rate {
        point_iter = Box::new(point_iter.filter(sample_points(rate)));
    }

    let octree = build_octree_from_iter(point_iter, max_node_size, cube_size);

    return octree;
}

fn load_model<P: AsRef<std::path::Path>>(path: P, device: tch::Device) -> tch::CModule {
    let mut model = tch::CModule::load_on_device(path, device).expect("cannot load model");
    model.set_eval();
    return model;
}

const COLOR_CHANNELS: usize = 3;

fn calculate_leaf_sh_batched<P: AsRef<std::path::Path>>(
    octree: &mut Octree<f64>,
    model_path: P,
    with_pb: bool,
) {
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        println!("warning: no cuda support, using CPU");
        tch::Device::Cpu
    };
    let model = load_model(model_path, device);
    let mut pb = if with_pb {
        Some(ProgressBar::new(octree.num_octants()))
    } else {
        None
    };

    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut octant_ids = Vec::new();
    let mut batch_indices = Vec::new();

    let batch_size = 64;

    let mut sh_coefs = HashMap::new();

    for octant in octree.into_iter() {
        let pc: &PointCloud<f64> = octant.points().into();
        let mut pc: PointCloud<f32> = pc.into();
        pc.scale_to_unit_sphere();

        let (pos, color) = pc.position_color();

        let pos_t = Tensor::of_slice(pos.as_slice())
            .reshape(&[-1, 3])
            .to_device(device);

        let color_t = Tensor::of_slice(color.as_slice())
            .reshape(&[-1, 3])
            .to_device(device);
        let batch_idx = batch_indices.len() as i64;
        let batch = Tensor::ones(&[pos_t.size()[0] as i64], kind::INT64_CUDA) * batch_idx;

        points.push(pos_t);
        colors.push(color_t);
        batch_indices.push(batch);
        octant_ids.push(octant.id);

        if batch_idx == batch_size {
            let pos_batch = Tensor::cat(points.as_slice(), 0);
            let color_batch = Tensor::cat(colors.as_slice(), 0);
            let batch_batch = Tensor::cat(batch_indices.as_slice(), 0);

            let coefs = model
                .forward_ts(&[&pos_batch, &color_batch, &batch_batch])
                .unwrap();

            for idx in 0..octant_ids.len() {
                let c = coefs.get(idx as i64);
                let mut new_coefs = [Vector3::<f32>::zeros(); 25];

                let f_coefs = Vec::<f32>::from(&c);
                for (i, v) in f_coefs.iter().enumerate() {
                    new_coefs[i / COLOR_CHANNELS][i % COLOR_CHANNELS] = *v;
                }
                sh_coefs.insert(octant_ids[idx], new_coefs);
            }

            points.clear();
            colors.clear();
            batch_indices.clear();
            octant_ids.clear();
        }
        if let Some(pb) = &mut pb {
            pb.inc();
        }
    }
    if points.len() > 0 {
        let pos_batch = Tensor::cat(points.as_slice(), 0);
        let color_batch = Tensor::cat(colors.as_slice(), 0);
        let batch_batch = Tensor::cat(batch_indices.as_slice(), 0);

        let coefs = model
            .forward_ts(&[&pos_batch, &color_batch, &batch_batch])
            .unwrap();

        for idx in 0..octant_ids.len() {
            let c = coefs.get(idx as i64);
            let mut new_coefs = [Vector3::<f32>::zeros(); 25];

            let f_coefs = Vec::<f32>::from(&c);
            for (i, v) in f_coefs.iter().enumerate() {
                new_coefs[i / COLOR_CHANNELS][i % COLOR_CHANNELS] = *v;
            }
            sh_coefs.insert(octant_ids[idx], new_coefs);
        }
    }

    println!("updating octree...");
    for octant in octree.into_iter() {
        if let Some(coefs) = sh_coefs.get(&octant.id()) {
            let pc = octant.points();
            let (centroid, _) = pc.centroid_and_color();

            let radius =
                pc.0.iter()
                    .map(|p| distance_squared(&p.position, &centroid))
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();

            octant.sh_rep.position = centroid;
            octant.sh_rep.radius = radius.sqrt();
            octant.sh_rep.coefficients = (*coefs).into();
        } else {
            panic!("cannot find octant with id: {:}", octant.id);
        }
    }
}

fn calculate_leaf_mean(octree: &mut Octree<f64>) {
    let mut pb = ProgressBar::new(octree.num_octants());
    pb.message("sh leaf nodes ");
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
        pb.inc();
    }
    pb.finish();
}

fn calculate_intermediate(octree: &mut Octree<f64>, with_pb: bool) {
    let cameras: Vec<Point3<f32>> = load_cameras(
        "sphere.ply",
        OrthographicProjection {
            width: 2.,
            height: 2.,
        },
    )
    .unwrap()
    .iter()
    .map(|c| c.position())
    .collect();

    let intermediate_nodes = octree.itermediate_octants();
    let mut pb = if with_pb {
        let mut pb = ProgressBar::new(intermediate_nodes.len() as u64);
        pb.message("sh intermediate nodes ");
        Some(pb)
    } else {
        None
    };
    for id in intermediate_nodes.iter().rev() {
        if let Some(node) = octree.get_mut(*id) {
            if let punctum::Node::Intermediate(i_node) = node {
                let mut centroid = Point3::origin();
                let mut n = 0;
                for child in i_node.data.iter() {
                    match child {
                        punctum::Node::Intermediate(child) => {
                            centroid += child.sh_rep.position.coords;
                            n += 1;
                        }
                        punctum::Node::Leaf(child) => {
                            centroid += child.sh_rep.position.coords;
                            n += 1;
                        }
                        punctum::Node::Empty => {}
                    }
                }
                centroid /= n as f64;
                let mut max_distance = 0.;
                for child in i_node.data.iter() {
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

                let child_coefs = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| match &i_node.data[i] {
                    punctum::Node::Intermediate(o) => Some(o.sh_rep.coefficients),
                    punctum::Node::Leaf(o) => Some(o.sh_rep.coefficients),
                    punctum::Node::Empty => None,
                });
                let new_coefs = merge_shs(child_coefs, &cameras);

                i_node.sh_rep = SHVertex::new(centroid, max_distance, new_coefs);
                if let Some(pb) = &mut pb {
                    pb.inc();
                }
            } else {
                unreachable!("only intermediate nodes should be found!")
            }
        } else {
            unreachable!("id must be present in octree")
        }
    }
    if let Some(pb) = &mut pb {
        pb.finish();
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    /// .las or .ply input file
    #[structopt(name = "input_file", parse(from_os_str))]
    input: PathBuf,

    // octree output file
    #[structopt(name = "output", parse(from_os_str))]
    output: PathBuf,

    /// maximum number of points per leaf node
    #[structopt(long, default_value = "8192")]
    max_octant_size: usize,

    /// sample every n-th point (random)
    #[structopt(long)]
    sample_rate: Option<usize>,

    /// flip y and z values
    #[structopt(long)]
    flip_yz: bool,

    /// TorchScript model for calculating the sh coefficients
    #[structopt(long, parse(from_os_str))]
    sh_model: Option<PathBuf>,

    /// no progress bar
    #[structopt(long)]
    no_pb: bool,
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
            !opt.no_pb,
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
            !opt.no_pb,
        )
    };
    let now = Instant::now();
    if let Some(model_path) = opt.sh_model {
        calculate_leaf_sh_batched(&mut octree, model_path, !opt.no_pb);
    } else {
        println!("warning: no model provided, will use mean color for sh representatives");
        calculate_leaf_mean(&mut octree);
    }
    let sh_compute_time = now.elapsed();

    calculate_intermediate(&mut octree, !opt.no_pb);

    #[cfg(debug_assertions)]
    {
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

        // we check if all points are contained by their bounding boxes
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
    }

    println!(
        "SH leaf computation time: {:}",
        sh_compute_time.as_secs_f32()
    );

    println!(
        "octree stats:\n\tnum_points:\t{}\n\tmax_depth:\t{}\n\tleaf_octants:\t{}",
        octree.num_points(),
        octree.depth(),
        octree.num_octants()
    );
    println!(
        "writing octree to {}:",
        opt.output.as_os_str().to_str().unwrap()
    );

    {
        let mut pb = if !opt.no_pb {
            let mut pb = ProgressBar::new(serialized_size(&octree).unwrap());
            pb.set_units(pbr::Units::Bytes);
            Some(pb)
        } else {
            None
        };

        let out_file = File::create(opt.output).unwrap();
        let mut out_writer = BufWriter::new(&out_file);

        if let Some(pb) = &mut pb {
            let mut tee = TeeWriter::new(&mut out_writer, pb);
            serialize_into(&mut tee, &octree).unwrap();
            pb.finish_println("done!");
        } else {
            serialize_into(&mut out_writer, &octree).unwrap();
        }
    }
}
