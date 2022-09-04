use bincode::{serialize_into, serialized_size};
use pbr::ProgressBar;
use std::{borrow::BorrowMut, collections::HashMap, fs::File, io::BufWriter, path::PathBuf};
use tch::{kind, IndexOp, Kind, Tensor};
use vulkano::buffer::BufferContents;

use nalgebra::Vector3;
use punctum::{load_octree_with_progress_bar, Octree, PointCloud, TeeWriter};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,
    #[structopt(name = "output_octree", parse(from_os_str))]
    output: PathBuf,
}

fn load_model<P: AsRef<std::path::Path>>(path: P, device: tch::Device) -> tch::CModule {
    let mut model = tch::CModule::load_on_device(path, device).expect("cannot load model");
    model.set_eval();
    return model;
}

const COLOR_CHANNELS: usize = 3;

pub fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let device = tch::Device::Cuda(0);
    let model = load_model(
        "logs/8192 octants l=4 rgb final/traced_model_gpu.pt",
        device,
    );

    let mut octree: Octree<f64> = load_octree_with_progress_bar(&filename).unwrap();

    let mut pb = ProgressBar::new(octree.num_octants());

    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut batch_indices = Vec::new();

    let mut sh_coefs = HashMap::new();

    for octant in octree.into_iter() {
        let pc: &PointCloud<f64> = octant.points().into();
        let mut pc: PointCloud<f32> = pc.into();
        pc.scale_to_unit_sphere();

        let raw_points = pc.points().as_bytes();
        let vertex_data = Tensor::of_data_size(
            raw_points,
            &[pc.points().len() as i64, (3 + COLOR_CHANNELS) as i64],
            Kind::Float,
        )
        .to(device);

        let pos = vertex_data.i((.., ..3));
        let color = vertex_data.i((.., 3..(3 + COLOR_CHANNELS) as i64));

        let batch_idx = batch_indices.len() as i64;
        let batch = Tensor::ones(&[pc.points().len() as i64], kind::INT64_CUDA) * batch_idx;

        points.push(pos);
        colors.push(color);
        batch_indices.push(batch);

        if batch_idx == 512 {
            let pos_batch = Tensor::cat(points.as_slice(), 0);
            let color_batch = Tensor::cat(colors.as_slice(), 0);
            let batch_batch = Tensor::cat(batch_indices.as_slice(), 0);

            println!(
                "pos: {:?}, color: {:?}, batch:{:?}",
                pos_batch.size(),
                color_batch.size(),
                batch_batch.size()
            );
            println!("max: {:?}, min: {:?}", batch_batch.min(), batch_batch.max(),);

            let coefs = model
                .forward_ts(&[&pos_batch, &color_batch, &batch_batch])
                .unwrap();

            for idx in 0..batch_indices.len() {
                let c = coefs.get(idx as i64);
                let mut new_coefs = [Vector3::<f32>::zeros(); 25];

                let f_coefs = Vec::<f32>::from(&c);
                for (i, v) in f_coefs.iter().enumerate() {
                    new_coefs[i / COLOR_CHANNELS][i % COLOR_CHANNELS] = *v;
                }
                sh_coefs.insert(octant.id(), new_coefs);
            }

            points.clear();
            colors.clear();
            batch_indices.clear();
        }
        pb.inc();
    }
    println!("");

    println!("updating octree...");
    for octant in octree.borrow_mut().into_iter() {
        let new_coefs = sh_coefs.get(&octant.id()).unwrap();
        octant.sh_rep.coefficients = (*new_coefs).into();
    }

    {
        let mut pb = ProgressBar::new(serialized_size(&octree).unwrap());
        pb.set_units(pbr::Units::Bytes);
        pb.message(&format!("exporting to {}: ", opt.output.to_str().unwrap()));

        let out_file = File::create(opt.output).unwrap();
        let mut out_writer = BufWriter::new(&out_file);
        let mut tee = TeeWriter::new(&mut out_writer, &mut pb);
        serialize_into(&mut tee, &octree).unwrap();
        println!("");
    }
}
