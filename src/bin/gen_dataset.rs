use std::{
    fs::File,
    io::{BufReader, BufWriter},
    sync::{Arc, Mutex},
};

use image::Rgba;
use nalgebra::Vector4;
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{
    Camera, Octree, OfflineRenderer, PointCloud, PointCloudGPU, RenderSettings, TeeReader, Vertex,
};
use rayon::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;
use vulkano::device::DeviceOwned;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output", parse(from_os_str))]
    output_folder: PathBuf,
}

struct RenderPool {
    workers: Vec<Arc<Mutex<OfflineRenderer>>>,
    next: usize,
    size: usize,
}

impl RenderPool {
    fn new(size: usize) -> Self {
        let rs = (0..size)
            .map(|_| {
                Arc::new(Mutex::new(OfflineRenderer::new(
                    64,
                    RenderSettings {
                        point_size: 32.0,
                        ..RenderSettings::default()
                    },
                )))
            })
            .collect::<Vec<Arc<Mutex<OfflineRenderer>>>>();
        RenderPool {
            workers: rs,
            next: 0,
            size,
        }
    }

    fn get(&mut self) -> &Arc<Mutex<OfflineRenderer>> {
        let item = self.workers.get(self.next).unwrap();
        self.next = (self.next + 1) % self.size;
        return item;
    }
}

fn export_ply(
    output_file: &PathBuf,
    pc: Arc<PointCloud<f32, f32>>,
    observed_colors: &Vec<Vertex<f32, f32>>,
) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<punctum::Vertex<f32, f32>>::new();
    let mut elm_def = punctum::Vertex::<f32, f32>::element_def("vertex".to_string());
    elm_def.count = pc.points().len();
    ply.header.encoding = Encoding::BinaryLittleEndian;
    ply.header.elements.add(elm_def.clone());

    let mut elm_def = Vertex::<f32, f32>::element_def("camera".to_string());
    elm_def.count = observed_colors.len();
    ply.header.elements.add(elm_def);

    let w = Writer::<punctum::Vertex<f32, f32>>::new();
    w.write_header(&mut file, &ply.header).unwrap();
    w.write_payload_of_element(
        &mut file,
        pc.points(),
        ply.header.elements.get("vertex").unwrap(),
        &ply.header,
    )
    .unwrap();
    w.write_payload_of_element(
        &mut file,
        observed_colors,
        ply.header.elements.get("camera").unwrap(),
        &ply.header,
    )
    .unwrap();
}

fn load_cameras() -> Vec<Camera> {
    let mut f = std::fs::File::open("sphere.ply").unwrap();

    // create a parser
    let p = ply_rs::parser::Parser::<Vertex<f32, f32>>::new();

    // use the parser: read the entire file
    let in_ply = p.read_ply(&mut f).unwrap();

    in_ply
        .payload
        .get("vertex")
        .unwrap()
        .clone()
        .iter()
        .map(|c| Camera::on_unit_sphere(c.position.into()))
        .collect::<Vec<Camera>>()
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input.as_os_str().to_str().unwrap();

    let octree = Arc::new({
        let in_file = File::open(&opt.input).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree
    });

    let cameras = load_cameras();

    let pb = Arc::new(Mutex::new(ProgressBar::new(octree.num_octants())));
    {
        pb.lock().unwrap().message(&format!("exporting octants: "));
    }

    let num_workers = 12;
    let render_pool = RenderPool::new(num_workers);
    let render_pool = Arc::new(Mutex::new(render_pool));

    // let pc_pool = Arc::new(CpuBufferPool::vertex_buffer(device));

    let pb_clone = pb.clone();
    octree
        .into_iter()
        .enumerate()
        .par_bridge()
        .into_par_iter()
        .for_each(|(i, node)| {
            let data_32 = node
                .data
                .iter()
                .map(|v| punctum::Vertex::<f32, f32> {
                    position: ((v.position - node.center.coords) * 2. / node.size).cast(),
                    color: v.color.cast() / 255.,
                })
                .collect();

            let mut pc = PointCloud::from_vec(&data_32);
            pc.scale_to_unit_sphere();
            let pc = Arc::new(pc);

            let renders: Vec<Rgba<u8>> = {
                let renderer = {
                    let mut r = render_pool.lock().unwrap();
                    r.get().clone()
                };

                let mut renderer = renderer.lock().unwrap();

                let device = renderer.device();

                let pc_gpu = PointCloudGPU::from_point_cloud(device.clone(), pc.clone());

                let renders = cameras
                    .iter()
                    .map(|c| renderer.render(c.clone(), &pc_gpu))
                    .collect();
                renders
            };

            let cam_colors: Vec<Vertex<f32, f32>> = renders
                .iter()
                .zip(cameras.clone())
                .map(|(color, cam)| Vertex {
                    position: *cam.position(),
                    // normal: Vector3::zeros(),
                    color: Vector4::from(color.0).cast() / 255.,
                })
                .collect();

            let out_file = opt.output_folder.join(format!("octant_{}.ply", i));
            export_ply(&out_file, pc, &cam_colors);
            pb_clone.lock().unwrap().inc();
        });
    pb.lock().unwrap().finish();
}
