use std::{
    fs::{self, File},
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
    load_cameras, Octree, OfflineRenderer, OrthographicProjection, PointCloud, PointCloudGPU,
    RenderSettings, TeeReader, Vertex,
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

    /// saves the individually rendered images as pngs
    /// WARNING: this createa A LOT of images  (162 per octant)
    #[structopt(long)]
    export_images: bool,
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
                    128,
                    RenderSettings {
                        point_size: 10,
                        ..RenderSettings::default()
                    },
                    true,
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
    pc: PointCloud<f32, f32>,
    observed_colors: &Vec<Vertex<f32, u8>>,
) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<punctum::Vertex<f32, u8>>::new();
    ply.header.encoding = Encoding::Ascii;

    let mut elm_def_vertex = punctum::Vertex::<f32, u8>::element_def("vertex".to_string());
    elm_def_vertex.count = pc.points().len();
    ply.header.elements.add(elm_def_vertex.clone());

    let mut elm_def_camera = Vertex::<f32, u8>::element_def("camera".to_string());
    elm_def_camera.count = observed_colors.len();
    ply.header.elements.add(elm_def_camera.clone());

    let w = Writer::<punctum::Vertex<f32, u8>>::new();
    w.write_header(&mut file, &ply.header).unwrap();

    let points_u8 = pc
        .points()
        .iter()
        .map(|p| (*p).into())
        .collect::<Vec<Vertex<f32, u8>>>();

    w.write_payload_of_element(&mut file, &points_u8, &elm_def_vertex, &ply.header)
        .unwrap();
    w.write_payload_of_element(&mut file, &observed_colors, &elm_def_camera, &ply.header)
        .unwrap();
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input.as_os_str().to_str().unwrap();

    if !opt.output_folder.exists() {
        std::fs::create_dir(opt.output_folder.clone()).unwrap();
    }

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

    let cameras = load_cameras(
        "sphere.ply",
        OrthographicProjection {
            width: 2.,
            height: 2.,
        },
    )
    .unwrap();

    let pb = Arc::new(Mutex::new(ProgressBar::new(octree.num_octants())));
    {
        pb.lock().unwrap().message(&format!("exporting octants: "));
    }

    let num_workers = 12;
    let render_pool = RenderPool::new(num_workers);
    let render_pool = Arc::new(Mutex::new(render_pool));

    let pb_clone = pb.clone();
    octree
        .into_iter()
        .par_bridge()
        .into_par_iter()
        .for_each(|node| {
            let data_32: Vec<Vertex<f32, f32>> =
                node.points().iter().map(|v| v.clone().into()).collect();

            let mut pc: PointCloud<f32, f32> = data_32.into();
            pc.scale_to_unit_sphere();

            let renders: Vec<Rgba<u8>> = {
                let renderer = {
                    let mut r = render_pool.lock().unwrap();
                    r.get().clone()
                };

                let mut renderer = renderer.lock().unwrap();

                let device = renderer.device();

                let pc_gpu = PointCloudGPU::from_point_cloud(device.clone(), pc.clone());

                let img_folder = opt.output_folder.join(format!("octant_{}", node.id()));
                if opt.export_images && !img_folder.exists() {
                    fs::create_dir(img_folder.clone()).unwrap();
                }

                let renders = cameras
                    .iter()
                    .enumerate()
                    .map(|(view_idx, c)| {
                        let avg_color = renderer.render(c.clone(), &pc_gpu);
                        if opt.export_images {
                            let img = renderer.last_image();

                            img.save(img_folder.join(format!(
                                "view_{}-{}r_{}g_{}b_{}a.png",
                                view_idx,
                                avg_color.0[0],
                                avg_color.0[1],
                                avg_color.0[2],
                                avg_color.0[3],
                            )))
                            .unwrap();
                        }
                        return avg_color;
                    })
                    .collect();

                renders
            };

            let cam_colors: Vec<Vertex<f32, u8>> = renders
                .iter()
                .zip(cameras.clone())
                .map(|(color, cam)| Vertex {
                    position: cam.position(),
                    color: Vector4::from(color.0),
                })
                .collect();

            let out_file = opt.output_folder.join(format!("octant_{}.ply", node.id()));
            export_ply(&out_file, pc, &cam_colors);
            pb_clone.lock().unwrap().inc();
        });
    pb.lock().unwrap().finish();
}
