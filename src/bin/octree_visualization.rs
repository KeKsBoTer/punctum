use std::{
    f32::consts::PI,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
};

use nalgebra::Vector4;
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{BoundingBox, Octree, PointCloud, TeeReader, Vertex};
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::path::PathBuf;

fn export_ply(output_file: &PathBuf, pc: &PointCloud<f32, u8>) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<punctum::Vertex<f32, u8>>::new();
    let mut elm_def = punctum::Vertex::<f32, u8>::element_def("vertex".to_string());
    elm_def.count = pc.points().len();
    ply.header.encoding = Encoding::Ascii;
    ply.header.elements.add(elm_def.clone());

    let w = Writer::<punctum::Vertex<f32, u8>>::new();
    w.write_header(&mut file, &ply.header).unwrap();
    w.write_payload_of_element(
        &mut file,
        pc.points(),
        ply.header.elements.get("vertex").unwrap(),
        &ply.header,
    )
    .unwrap();
}

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
    let filename = "datasets/neuschwanstein/octree_4000_2048max.bin";
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
                // .filter(|p| rand_gen.gen::<u32>() % 32u32 == 0u32)
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
