use nalgebra::Vector4;
use pbr::ProgressBar;
use ply_rs::{
    parser::Parser,
    ply::{DefaultElement, Property},
};
use punctum::{
    load_octree_with_progress_bar, save_octree_with_progress_bar, sh::lm2flat_index, Octree,
    PointCloud, SHCoefficients, SHVertex,
};
use rayon::prelude::*;
use std::{
    borrow::BorrowMut,
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    sync::Mutex,
};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "coefs_dir", parse(from_os_str))]
    coefs_dir: PathBuf,

    #[structopt(name = "output_octree", parse(from_os_str))]
    output_octree: PathBuf,
}

fn read_coefs<P: AsRef<Path>>(path: P) -> io::Result<SHCoefficients<121>> {
    let mut ply_file = std::fs::File::open(path)?;
    let p = Parser::<DefaultElement>::new();

    let ply = p.read_ply(&mut ply_file).unwrap();

    let coefs = ply.payload.get("sh_coefficients").unwrap();
    let mut new_coefs = [Vector4::<f32>::zeros(); 121];
    for c in coefs {
        let l = if let Property::UChar(l) = c.get("l").unwrap() {
            *l
        } else {
            panic!("l must be uchar")
        };
        let m = if let Property::Char(m) = c.get("m").unwrap() {
            *m
        } else {
            panic!("l must be uchar")
        };
        let i = lm2flat_index(l as u64, m as i64);

        if let Property::ListFloat(values) = c.get("coefficients").unwrap() {
            new_coefs[i].x = values[0];
            new_coefs[i].y = values[1];
            new_coefs[i].z = values[2];
            new_coefs[i].w = values[3];
        } else {
            panic!("coefs must be float list")
        }
    }
    Ok(new_coefs.into())
}

pub fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let mut octree: Octree<f64, u8> = load_octree_with_progress_bar(&filename).unwrap();
    let pb = Mutex::new(ProgressBar::new(octree.num_octants()));

    let sh_coefs = octree
        .into_iter()
        .par_bridge()
        .map(|octant| {
            let pc: &PointCloud<f64, u8> = octant.points().into();
            let centroid = pc.centroid();
            let ply_file = opt.coefs_dir.join(format!("octant_{}.ply", octant.id()));
            let sh_approximation = SHVertex::new(centroid, read_coefs(ply_file).unwrap());
            pb.lock().unwrap().inc();
            (octant.id(), sh_approximation)
        })
        .collect::<HashMap<u64, SHVertex<f64, 121>>>();

    for octant in octree.borrow_mut().into_iter() {
        octant.sh_approximation = Some(*sh_coefs.get(&octant.id()).unwrap());
    }

    save_octree_with_progress_bar(opt.output_octree, &octree).unwrap();
}
