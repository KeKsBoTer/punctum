use std::{
    f32::consts::PI,
    fs::File,
    io::{BufReader, Write},
    path::Path,
    sync::Arc,
};

use nalgebra::Vector4;
use pbr::ProgressBar;
use punctum::{export_ply, BoundingBox, Octree, PointCloud, TeeReader, Vertex};
use rand::{prelude::StdRng, Rng, SeedableRng};

fn angle_to_rgba(angle: f32) -> Vector4<u8> {
    let mut color = Vector4::new(angle, angle - 2. * PI / 3., angle + 2. * PI / 3., 1.0);
    color.x = (color.x.cos() + 1.) / 2.;
    color.y = (color.y.cos() + 1.) / 2.;
    color.z = (color.z.cos() + 1.) / 2.;
    return Vector4::new(
        (color.x * 255.) as u8,
        (color.y * 255.) as u8,
        (color.z * 255.) as u8,
        255,
    );
}

fn main() {
    let filename = "dataset/octree_16_1024max.bin";
    let octree = Arc::new({
        let in_file = File::open(filename).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree
    });
    let mut rand_gen = StdRng::seed_from_u64(42);

    let mut points: Vec<Vertex<f32, u8>> = Vec::with_capacity(octree.num_points() as usize);
    let mut pb = ProgressBar::new(octree.num_octants());
    let num_octants = octree.num_octants();
    let mut bboxes = Vec::with_capacity(num_octants as usize);

    for octant in octree.into_iter() {
        let color_id = rand_gen.gen::<u64>() % num_octants;
        let color = angle_to_rgba(color_id as f32 / num_octants as f32 * 2. * PI);
        points.extend(
            octant
                .data
                .iter()
                .map(|p| Vertex {
                    position: p.position.cast(),
                    color: color,
                })
                .collect::<Vec<Vertex<f32, u8>>>(),
        );
        bboxes.push(BoundingBox::from_points(octant.data).size());
        pb.inc();
    }

    let mut size_file = File::create("octant_sizes.txt").unwrap();

    for size in bboxes {
        writeln!(&mut size_file, "{} {} {}", size.x, size.y, size.z).unwrap();
    }

    println!("num_points: {}", points.len());
    let pc = PointCloud::from_vec(&points);

    export_ply(&Path::new("test_octree_64.ply").to_path_buf(), &pc);
}
