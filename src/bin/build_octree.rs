use std::{fs::File, io::BufWriter};

use bincode::{serialize_into, serialized_size};
use las::{Read as LasRead, Reader};
use nalgebra::{center, Point3, Vector4};
use pbr::ProgressBar;
use punctum::{Octree, TeeWriter, Vertex};
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use structopt::StructOpt;

fn build_octree(
    las_file: &PathBuf,
    max_node_size: usize,
    max_octants: Option<usize>,
    sample_rate: Option<usize>,
    flip_yz: bool,
) -> Octree<f64, u8> {
    let mut reader = Reader::from_path(las_file).unwrap();

    let number_of_points = reader.header().number_of_points();

    // normalize octree to a cube of side length 100
    let cube_size = 100.;

    let bounds = reader.header().bounds();
    let min_point = if flip_yz {
        Point3::new(bounds.min.x, bounds.min.z, bounds.min.y)
    } else {
        Point3::new(bounds.min.x, bounds.min.y, bounds.min.z)
    };
    let max_point = if flip_yz {
        Point3::new(bounds.max.x, bounds.max.z, bounds.max.y)
    } else {
        Point3::new(bounds.max.x, bounds.max.y, bounds.max.z)
    };

    let bb_size = max_point - min_point;
    let max_size = bb_size[bb_size.imax()];
    let center = center(&min_point, &max_point);

    let mut octree = Octree::new(Point3::new(0., 0., 0.), cube_size, max_node_size);
    let mut pb = ProgressBar::new(number_of_points);
    let mut counter = 0;
    let mut octant_counter = 0;

    let mut rand_gen = StdRng::seed_from_u64(42);

    for p in reader.points() {
        counter += 1;
        if let Some(rate) = sample_rate {
            let r: usize = rand_gen.gen();
            if r % rate != 0 {
                continue;
            }
        }

        let point = p.unwrap();
        let color = point.color.unwrap();

        let position = if flip_yz {
            Point3::new(point.x, point.z, point.y)
        } else {
            Point3::new(point.x, point.y, point.z)
        };

        let point = Vertex {
            position: (&position - &center.coords) * cube_size / max_size,
            color: Vector4::new(
                (color.red / 256) as u8, // 65536 = 2**16
                (color.green / 256) as u8,
                (color.blue / 256) as u8,
                255,
            ),
        };
        octant_counter += octree.insert(point);
        if let Some(max) = max_octants {
            if octant_counter >= max {
                break;
            }
        }
        if counter >= number_of_points / 100 {
            pb.add(counter);
            counter = 0;
        }
    }
    pb.add(counter);
    return octree;
}

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_las", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output", parse(from_os_str))]
    output: PathBuf,

    #[structopt(long, default_value = "1024")]
    max_octant_size: usize,

    #[structopt(long)]
    max_octants: Option<usize>,

    #[structopt(long)]
    sample_rate: Option<usize>,

    #[structopt(long)]
    flip_yz: bool,
}

fn main() {
    let opt = Opt::from_args();
    println!(
        "Building octree from {}:",
        opt.input.as_os_str().to_str().unwrap()
    );
    let octree = build_octree(
        &opt.input,
        opt.max_octant_size,
        opt.max_octants,
        opt.sample_rate,
        opt.flip_yz,
    );

    // we check that all ids are unqiue
    // if not the three is to deep (or something is wrong in the code :P)
    let mut ids = octree
        .into_iter()
        .map(|octant| octant.id)
        .collect::<Vec<u64>>();
    ids.sort_unstable();

    let in_order = ids.iter().zip(ids.iter().skip(1)).find(|(a, b)| **a == **b);
    if let Some(duplicate) = in_order {
        panic!("duplicate id {:}!", duplicate.0);
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
