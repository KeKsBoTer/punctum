use std::{borrow::BorrowMut, path::PathBuf};

use nalgebra::Vector4;
use punctum::{load_octree_with_progress_bar, load_raw_coefs, Octree, PointCloud, SHVertex};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "input_coefs", parse(from_os_str))]
    coefs_file: PathBuf,
}

pub fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let mut octree: Octree<f64, u8> = load_octree_with_progress_bar(&filename).unwrap();

    let coefs = load_raw_coefs(opt.coefs_file).unwrap();

    for octant in octree.borrow_mut().into_iter() {
        let pc: &PointCloud<f64, u8> = octant.points().into();
        let centroid = pc.centroid();
        let mut new_coefs = [Vector4::<f32>::zeros(); 121];

        if let Some(cs) = coefs.get(&octant.id()) {
            for i in 0..cs.len() {
                new_coefs[i] = cs[i].into();
            }
        } else {
            println!("id {} not found", octant.id());
        }
        octant.sh_approximation = Some(SHVertex::new(centroid, new_coefs.into()));
    }
}
