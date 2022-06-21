use std::{
    f32::consts::PI,
    fs::File,
    io::BufWriter,
    sync::{Arc, Mutex},
};

use image::Rgba;
use nalgebra::{Point3, Vector3, Vector4};
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{
    OfflineRenderer, OrthographicCamera, PointCloud, PointCloudGPU, RenderSettings, Vertex,
};
use rand::{
    distributions::WeightedIndex,
    prelude::{Distribution, StdRng},
    SeedableRng,
};
use rayon::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;
use vulkano::device::DeviceOwned;

const OCTREE_SIZE_DIST: [f32; 16] = [
    0.0071, 0.0201, 0.0528, 0.1035, 0.1414, 0.1349, 0.1156, 0.0931, 0.0740, 0.0605, 0.0527, 0.0382,
    0.0355, 0.0269, 0.0235, 0.0202,
];

const POSITION_DIST: [f32; 32] = [
    0.0000, 0.0029, 0.0060, 0.0092, 0.0124, 0.0159, 0.0195, 0.0232, 0.0272, 0.0314, 0.0360, 0.0410,
    0.0464, 0.0527, 0.0600, 0.0697, 0.0929, 0.0697, 0.0600, 0.0527, 0.0464, 0.0410, 0.0360, 0.0314,
    0.0272, 0.0232, 0.0195, 0.0159, 0.0124, 0.0092, 0.0060, 0.0029,
];

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "max_octant_size")]
    max_octant_size: usize,

    #[structopt(name = "output", parse(from_os_str))]
    output_folder: PathBuf,

    #[structopt(short, long, default_value = "1024")]
    num_samples: usize,
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
                        point_size: 32,
                        ..RenderSettings::default()
                    },
                    false,
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
    pc: &PointCloud<f32, f32>,
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

fn load_cameras() -> Vec<OrthographicCamera> {
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
        .map(|c| OrthographicCamera::on_unit_sphere(c.position.into()))
        .collect::<Vec<OrthographicCamera>>()
}

fn rand_point(generator: &mut StdRng, dist: &WeightedIndex<f32>) -> Point3<f32> {
    let to_pos = |i: usize| (i as f32 / POSITION_DIST.len() as f32) * 2. - 1.;
    loop {
        let x: f32 = to_pos(dist.sample(generator));
        let y: f32 = to_pos(dist.sample(generator));
        let z: f32 = to_pos(dist.sample(generator));
        let p = Vector3::new(x, y, z);
        if p.norm_squared() < 1. {
            return p.into();
        }
    }
}

fn angle_to_rgba(angle: f32) -> Vector4<f32> {
    let mut color = Vector4::new(angle, angle - 2. * PI / 3., angle + 2. * PI / 3., 1.0);
    color.x = (color.x.cos() + 1.) / 2.;
    color.y = (color.y.cos() + 1.) / 2.;
    color.z = (color.z.cos() + 1.) / 2.;
    return color;
}

fn main() {
    let opt = Opt::from_args();

    let cameras = load_cameras();

    let pb = Arc::new(Mutex::new(ProgressBar::new(opt.num_samples as u64)));
    {
        pb.lock().unwrap().message(&format!("exporting octants: "));
    }

    let num_workers = 24;
    let render_pool = RenderPool::new(num_workers);
    let render_pool = Arc::new(Mutex::new(render_pool));

    let pb_clone = pb.clone();

    let dist = WeightedIndex::new(&OCTREE_SIZE_DIST).unwrap();
    let pos_dist = WeightedIndex::new(&POSITION_DIST).unwrap();

    (0..opt.num_samples).par_bridge().for_each(|i| {
        let mut rand_gen = StdRng::seed_from_u64(i as u64);

        let octant_size: usize =
            ((dist.sample(&mut rand_gen) + 1) * opt.max_octant_size) / OCTREE_SIZE_DIST.len();

        let points: Vec<Vertex<f32, f32>> = (0..octant_size)
            .map(|j| {
                let angle = j as f32 / octant_size as f32 * 2. * PI;
                Vertex {
                    position: rand_point(&mut rand_gen, &pos_dist),
                    color: angle_to_rgba(angle),
                }
            })
            .collect();

        let pc: PointCloud<f32, f32> = points.into();

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
                color: Vector4::from(color.0).cast() / 255.,
            })
            .collect();

        let out_file = opt.output_folder.join(format!("octant_{}.ply", i));
        export_ply(&out_file, &pc, &cam_colors);
        pb_clone.lock().unwrap().inc();
    });
    pb.lock().unwrap().finish();
}
