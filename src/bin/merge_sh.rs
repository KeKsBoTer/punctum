use nalgebra::Vector3;
use pbr::ProgressBar;
use ply_rs::{
    parser::Parser,
    ply::{DefaultElement, Property},
};
use punctum::{
    load_octree_with_progress_bar, save_octree_with_progress_bar, sh::lm2flat_index, Octree,
    SHCoefficients,
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

fn read_coefs<P: AsRef<Path> + Clone, const T: usize>(path: P) -> io::Result<SHCoefficients<T>> {
    let path_name = path.as_ref().to_string_lossy().to_string();
    let mut ply_file = std::fs::File::open(path)?;
    let p = Parser::<DefaultElement>::new();

    let ply = p.read_ply(&mut ply_file).unwrap();

    let coefs = ply.payload.get("sh_coefficients").unwrap();
    if coefs.len() < T {
        println!(
            "WARN: '{:}' has only degree {} but expected {}",
            path_name,
            coefs.len(),
            T
        );
    } else if coefs.len() > T {
        panic!(
            "'{:}' has only degree {} but expected {}",
            path_name,
            coefs.len(),
            T
        );
    }
    let mut new_coefs = [Vector3::<f32>::zeros(); T];
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
        } else {
            panic!("coefs must be float list")
        }
    }
    Ok(new_coefs.into())
}

pub fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let mut octree: Octree<f64> = load_octree_with_progress_bar(&filename).unwrap();
    let pb = Mutex::new(ProgressBar::new(octree.num_octants()));

    let sh_coefs = octree
        .into_iter()
        .par_bridge()
        .map(|octant| {
            let ply_file = opt.coefs_dir.join(format!("octant_{}.ply", octant.id()));
            let sh_coefs = read_coefs(ply_file).unwrap();
            pb.lock().unwrap().inc();
            (octant.id(), sh_coefs)
        })
        .collect::<HashMap<u64, SHCoefficients>>();

    for octant in octree.borrow_mut().into_iter() {
        let c = *sh_coefs.get(&octant.id()).unwrap();
        octant.sh_rep.coefficients = c;
    }

    save_octree_with_progress_bar(opt.output_octree, &octree).unwrap();
}
