use std::{
    fs::File,
    io::{BufReader, BufWriter},
    sync::{Arc, Mutex},
};

use nalgebra::Point3;
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{Octree, PointCloud, TeeReader, Vertex};
use rayon::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_las", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output", parse(from_os_str))]
    output_folder: PathBuf,
}

fn export_ply(output_file: &PathBuf, data: &Vec<Vertex<f64, u8>>, center: Point3<f64>, size: f64) {
    let data_32 = data
        .iter()
        .map(|v| punctum::Vertex::<f32, u8> {
            position: ((v.position - center.coords) / size).cast(),
            // normal: Vector3::zeros(),
            color: v.color,
        })
        .collect();

    let mut pc = PointCloud::from_vec(&data_32);
    pc.scale_to_unit_sphere();

    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<punctum::Vertex<f32, u8>>::new();
    let mut elm_def = punctum::Vertex::<f32, u8>::element_def("vertex".to_string());
    elm_def.count = data_32.len();
    ply.header.encoding = Encoding::BinaryLittleEndian;
    ply.header.elements.add(elm_def.clone());

    ply.payload.insert("vertex".to_string(), data_32);

    let w = Writer::<punctum::Vertex<f32, u8>>::new();
    w.write_ply_unchecked(&mut file, &ply).unwrap();
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input.as_os_str().to_str().unwrap();

    let octree = {
        let in_file = File::open(&opt.input).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree
    };

    let pb = Arc::new(Mutex::new(ProgressBar::new(octree.num_octants())));
    {
        pb.lock().unwrap().message(&format!("exporting octants: "));
    }

    let pb_clone = pb.clone();
    octree
        .into_iter()
        .enumerate()
        .par_bridge()
        .into_par_iter()
        .for_each(|(i, node)| {
            let out_file = opt.output_folder.join(format!("octant_{}.ply", i));
            export_ply(&out_file, node.data, node.center, node.size);
            if i % 100 == 0 {
                pb_clone.lock().unwrap().add(100);
            }
        });
    pb.lock().unwrap().finish();
}
