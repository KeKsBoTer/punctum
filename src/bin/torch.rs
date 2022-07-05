use bincode::{serialize_into, serialized_size};
use pbr::ProgressBar;
use std::{borrow::BorrowMut, fs::File, io::BufWriter, path::PathBuf};
use tch::{kind, IndexOp, Kind, Tensor};
use vulkano::buffer::BufferContents;

use nalgebra::Vector4;
use punctum::{load_octree_with_progress_bar, Octree, PointCloud, SHVertex, TeeWriter};
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

pub fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let device = tch::Device::Cuda(0);
    let model = load_model("traced_model.pt", device);

    let mut octree: Octree<f64, u8> = load_octree_with_progress_bar(&filename).unwrap();

    let mut pb = ProgressBar::new(octree.num_octants());

    // TODO implement batching. rn we work with batchsize = 1
    for octant in octree.borrow_mut().into_iter() {
        let pc: &PointCloud<f64, u8> = octant.points().into();
        let pc: PointCloud<f32, f32> = pc.into();
        let centroid = pc.centroid();

        let raw_points = pc.points().as_bytes();
        let vertex_data =
            Tensor::of_data_size(raw_points, &[pc.points().len() as i64, 7], Kind::Float)
                .to(device);

        let pos = vertex_data.i((.., ..3));
        let color = vertex_data.i((.., 3..));

        let batch = Tensor::zeros(&[pc.points().len() as i64], kind::INT64_CUDA);

        let coefs = model.forward_ts(&[&pos, &color, &batch]).unwrap();
        let f_coefs = Vec::<f32>::from(&coefs.get(0));
        let mut new_coefs = [Vector4::<f32>::zeros(); 121];
        for (i, v) in f_coefs.iter().enumerate() {
            new_coefs[i / 4][i % 4] = *v;
        }

        octant.sh_approximation = Some(SHVertex::new(centroid.cast(), new_coefs.into()));

        pb.inc();
    }
    println!("");

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
