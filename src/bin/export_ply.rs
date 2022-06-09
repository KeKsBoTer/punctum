use std::{fs::File, io::BufReader, path::PathBuf, sync::Arc};

use pbr::ProgressBar;
use punctum::{export_ply, Octree, PointCloud, TeeReader, Vertex};
use rand::{prelude::StdRng, Rng, SeedableRng};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "PLY Exporter")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output_ply", parse(from_os_str))]
    output: PathBuf,

    #[structopt(long)]
    sample_rate: Option<usize>,
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

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!\n");

        octree
    });
    let mut rand_gen = StdRng::seed_from_u64(42);

    let size = octree.size();
    let center = *octree.center();

    let points: Vec<Vertex<f64, u8>> = octree
        .into_iter()
        .flat_map(|octant| {
            octant
                .data
                .iter()
                .filter(|v| {
                    if let Some(sample_rate) = opt.sample_rate {
                        rand_gen.gen::<usize>() % sample_rate == 0
                    } else {
                        (v.position - center).norm_squared() < (size * size / (10. * 10.))
                    }
                })
                .map(|p| *p)
                .collect::<Vec<Vertex<f64, u8>>>()
        })
        .collect();

    println!("num_points: {}", points.len());
    let pc = PointCloud::from_vec(&points);

    export_ply(&opt.output, &pc);
}
