use std::{fs::File, io::BufWriter};

use bincode::{serialize_into, serialized_size};
use las::{Read as LasRead, Reader};
use nalgebra::{center, Point3, Vector4};
use pbr::ProgressBar;
use punctum::{Octree, TeeWriter, Vertex};
use std::path::PathBuf;
use structopt::StructOpt;

fn build_octree(
    las_file: &PathBuf,
    max_node_size: usize,
    max_octants: Option<usize>,
) -> Octree<f64, u8> {
    let mut reader = Reader::from_path(las_file).unwrap();

    let number_of_points = reader.header().number_of_points();

    let bounds = reader.header().bounds();
    let min_point = Point3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max_point = Point3::new(bounds.max.x, bounds.max.y, bounds.max.z);
    let size = max_point - min_point;
    let max_size = [size.x, size.y, size.z]
        .into_iter()
        .reduce(|a, b| a.max(b))
        .unwrap();

    let mut octree = Octree::new(center(&min_point, &max_point), max_size, max_node_size);
    let mut pb = ProgressBar::new(number_of_points);
    let mut counter = 0;
    let mut octant_counter = 0;
    for p in reader.points() {
        let point = p.unwrap();
        let color = point.color.unwrap();
        let point = Vertex {
            position: Point3::new(point.x, point.y, point.z),
            color: Vector4::new(
                (color.red / 256) as u8, // 65536 = 2**16
                (color.green / 256) as u8,
                (color.blue / 256) as u8,
                255,
            ),
        };
        octant_counter += octree.insert(point);
        counter += 1;
        if let Some(max) = max_octants {
            if octant_counter >= max {
                break;
            }
        }
        if counter == 100000 {
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

    #[structopt(short, long, default_value = "1024")]
    max_octant_size: usize,

    #[structopt(short, long)]
    max_octants: Option<usize>,
}

fn main() {
    let opt = Opt::from_args();
    println!(
        "Building octree from {}:",
        opt.input.as_os_str().to_str().unwrap()
    );
    let octree = build_octree(&opt.input, opt.max_octant_size, opt.max_octants);
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
