use std::{fs::File, io::BufReader, path::PathBuf, sync::Arc};

use pbr::ProgressBar;
use punctum::{export_ply, Octree, TeeReader, Vertex};
use rand::prelude::IteratorRandom;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree to PLY converter")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output_ply", parse(from_os_str))]
    output: PathBuf,

    /// number of points sampled from input point cloud
    #[structopt(long, default_value = "1000000")]
    max_points: usize,
}

fn main() {
    let opt = Opt::from_args();
    let octree = Arc::new({
        let in_file = File::open(opt.input.clone()).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {:?}: ", opt.input));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree
    });

    let points = octree.flat_points();

    let mut rng = &mut rand::thread_rng();
    let export_points: Vec<Vertex<f64>> =
        points.into_iter().choose_multiple(&mut rng, opt.max_points);

    export_ply(&opt.output, &export_points.into());
}
