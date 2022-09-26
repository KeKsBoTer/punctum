use std::{
    fs::{self, File},
    io::{BufReader, BufWriter},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use image::Rgba;
use nalgebra::Vector3;
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{
    load_cameras, LeafNode, Octree, OfflineRenderer, OrthographicProjection, PointCloud,
    PointCloudGPU, RenderSettings, TeeReader, Vertex,
};
use std::path::PathBuf;
use structopt::StructOpt;
use vulkano::device::DeviceOwned;

#[derive(StructOpt, Debug, Clone)]
#[structopt(name = "Dataset generator")]
struct Opt {
    /// octree file
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    /// target folder where the ply files will be saved to
    #[structopt(name = "output", parse(from_os_str))]
    output_folder: PathBuf,

    /// saves the individually rendered images as pngs
    /// WARNING: this createa A LOT of images  (162 per octant)
    #[structopt(long)]
    export_images: bool,

    /// if set no ply files are exported (used for performance measurement)
    #[structopt(long)]
    measure: bool,
}

fn export_ply(output_file: &PathBuf, pc: PointCloud<f32>, observed_colors: &Vec<Vertex<f32>>) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<punctum::Vertex<f32>>::new();
    ply.header.encoding = Encoding::BinaryLittleEndian;

    let mut elm_def_vertex = punctum::Vertex::<f32>::element_def("vertex".to_string());
    elm_def_vertex.count = pc.points().len();
    ply.header.elements.add(elm_def_vertex.clone());

    let mut elm_def_camera = Vertex::<f32>::element_def("camera".to_string());
    elm_def_camera.count = observed_colors.len();
    ply.header.elements.add(elm_def_camera.clone());

    let w = Writer::<punctum::Vertex<f32>>::new();
    w.write_header(&mut file, &ply.header).unwrap();

    let points_u8 = pc
        .points()
        .iter()
        .map(|p| (*p).into())
        .collect::<Vec<Vertex<f32>>>();

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

    let octree: Arc<Octree<f32>> = Arc::new({
        let in_file = File::open(&opt.input).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);
        if opt.measure {
            pb.set_max_refresh_rate(Some(Duration::from_secs(60 * 60 * 1000)));
        }
        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree.into()
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
        let mut pb = pb.lock().unwrap();
        pb.message(&format!("exporting octants: "));
        if opt.measure {
            pb.set_max_refresh_rate(Some(Duration::from_secs(60 * 60 * 1000)));
        }
    }

    let num_workers = 12;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .unwrap();

    let octree_iter = octree.into_iter();
    let all_octants = octree_iter.collect::<Vec<&LeafNode<f32>>>();

    let output_folder = opt.output_folder.as_path();

    let start = Instant::now();
    rayon::scope(|s| {
        all_octants
            .chunks(octree.num_octants() as usize / num_workers)
            .map(|octants| {
                let pb_clone = pb.clone();
                let cameras = cameras.clone();
                s.spawn(move |_| {
                    let mut renderer = OfflineRenderer::new(
                        128,
                        RenderSettings {
                            point_size: 10,
                            ..RenderSettings::default()
                        },
                        true,
                    );

                    for octant in octants {
                        let mut pc: PointCloud<f32> = octant.points().clone().into();
                        pc.scale_to_unit_sphere();

                        let renders: Vec<Rgba<u8>> = {
                            let device = renderer.device();

                            let pc_gpu =
                                PointCloudGPU::from_point_cloud(device.clone(), pc.clone());

                            let img_folder = output_folder.join(format!("octant_{}", octant.id()));
                            if !opt.measure && opt.export_images && !img_folder.exists() {
                                fs::create_dir(img_folder.clone()).unwrap();
                            }

                            let renders = cameras
                                .iter()
                                .enumerate()
                                .map(|(view_idx, c)| {
                                    let avg_color = renderer.render(c.clone(), &pc_gpu);
                                    if !opt.measure && opt.export_images {
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

                        let cam_colors: Vec<Vertex<f32>> = renders
                            .iter()
                            .zip(cameras.clone())
                            .map(|(color, cam)| Vertex {
                                position: cam.position(),
                                color: Vector3::new(color.0[0], color.0[1], color.0[2]),
                            })
                            .collect();

                        if !opt.measure {
                            let out_file =
                                output_folder.join(format!("octant_{}.ply", octant.id()));
                            export_ply(&out_file, pc, &cam_colors);
                        }

                        let mut pb = pb_clone.lock().unwrap();
                        pb.inc();
                    }
                })
            })
            .count();
    });
    pb.lock().unwrap().finish();
    println!("done! Duration: {}s", start.elapsed().as_secs_f32());
}
