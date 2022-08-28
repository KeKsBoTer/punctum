use std::{f32::consts::PI, fs::File, io::BufReader, path::PathBuf, sync::Arc};

use nalgebra::{Vector3, Vector4};
use pbr::ProgressBar;
use punctum::{export_ply, Octree, TeeReader, Vertex};
use rand::{prelude::StdRng, Rng, SeedableRng};
use structopt::StructOpt;

fn angle_to_rgba(angle: f32) -> Vector3<u8> {
    let mut color = Vector4::new(angle, angle - 2. * PI / 3., angle + 2. * PI / 3., 1.0);
    color.x = (color.x.cos() + 1.) / 2.;
    color.y = (color.y.cos() + 1.) / 2.;
    color.z = (color.z.cos() + 1.) / 2.;
    return Vector3::new(
        (color.x * 255.) as u8,
        (color.y * 255.) as u8,
        (color.z * 255.) as u8,
    );
}

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output_ply", parse(from_os_str))]
    output: PathBuf,
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
    let mut rand_gen = StdRng::seed_from_u64(42);

    let mut points: Vec<Vertex<f32>> = Vec::with_capacity(octree.num_points() as usize);
    let mut pb = ProgressBar::new(octree.num_octants());
    let num_octants = octree.num_octants();

    for octant in octree.into_iter() {
        let color_id = rand_gen.gen::<u64>() % num_octants;
        let color = angle_to_rgba(color_id as f32 / num_octants as f32 * 2. * PI);
        points.extend(
            octant
                .points()
                .0
                .iter()
                .map(|p| Vertex {
                    position: p.position.cast(),
                    color: color,
                })
                .collect::<Vec<Vertex<f32>>>(),
        );
        pb.inc();
    }
    println!("num_points: {}", points.len());

    export_ply(&opt.output, &points.into());
}
