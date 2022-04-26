use std::{
    env,
    fs::File,
    io::{BufWriter, Read, Stdout, Write},
};

use bincode::{serialize_into, serialized_size};
use las::{Read as LasRead, Reader};
use nalgebra::{center, Point3, Vector3, Vector4};
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use punctum::{Node, Octree, PointCloud, Vertex};
use std::io::prelude::*;

fn build_octree(las_file: &String) -> Octree<f64> {
    let mut reader = Reader::from_path(las_file).unwrap();

    let number_of_points = reader.header().number_of_points();

    let bounds = reader.header().bounds();
    let min_point = Point3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max_point = Point3::new(bounds.max.x, bounds.max.y, bounds.max.z);
    let size = max_point - min_point;
    let max_size = [size.x, size.y, size.z]
        .into_iter()
        .reduce(|a, b| a.max(b))
        .unwrap();

    let mut octree = Octree::new(center(&min_point, &max_point), max_size);
    let mut pb = ProgressBar::new(number_of_points);
    let mut counter = 0;
    for p in reader.points() {
        let point = p.unwrap();
        let color = point.color.unwrap();
        let point = Vertex {
            position: Point3::new(point.x, point.y, point.z),
            // normal: Vector3::zeros(),
            color: Vector4::new(
                color.red as f32 / 65536., // 65536 = 2**16
                color.green as f32 / 65536.,
                color.blue as f32 / 65536.,
                1.,
            ),
        };
        octree.insert(point);
        counter += 1;
        if counter == 100000 {
            pb.add(counter);
            counter = 0;
        }
    }
    pb.add(counter);
    return octree;
}

fn main() {
    let args = env::args();
    if args.len() != 3 {
        panic!("Usage: <point_cloud>.las <outfile.bin>");
    }
    let arguments = args.collect::<Vec<String>>();
    let las_file = arguments.get(1).unwrap();
    let output_file = arguments.get(2).unwrap();

    println!("Building octree from {}:", las_file);
    let octree = build_octree(las_file);
    println!(
        "octree stats:\n\tnum_points:\t{}\n\tmax_depth:\t{}\n\tnum_octants:\t{}",
        octree.num_points(),
        octree.depth(),
        octree.num_octants()
    );
    println!("writing octree to {}:", output_file);
    {
        let mut pb = ProgressBar::new(serialized_size(&octree).unwrap());
        pb.set_units(pbr::Units::Bytes);

        let out_file = File::create(output_file).unwrap();
        let mut out_writer = BufWriter::new(&out_file);
        let mut tee = TeeWriter::new(&mut out_writer, &mut pb);
        serialize_into(&mut tee, &octree).unwrap();

        pb.finish_println(&format!(
            "File size: {:?}Mb",
            out_file.metadata().unwrap().len() / (1024 * 1024)
        ));
    }
    // let mut in_file = File::open("octree.bin").unwrap();

    // let mut buffer = Vec::with_capacity(in_file.metadata().unwrap().len() as usize);
    // in_file.read_to_end(&mut buffer).unwrap();
    // let decoded: Octree<f64> = bincode::deserialize(&buffer).unwrap();

    // println!("decoded octree file!");

    // println!("depth: {}", decoded.depth());

    // let w = Writer::<punctum::Vertex<f32>>::new();

    // let mut counter = 0;
    // octree.traverse(move |node, center, size| {
    //     if let Node::Filled(data) = node {
    //         let data_32 = data
    //             .iter()
    //             .map(|v| punctum::Vertex {
    //                 position: ((v.position - center.coords) / size).cast(),
    //                 normal: Vector3::zeros(),
    //                 color: v.color.cast() / 255.,
    //             })
    //             .collect();

    //         let mut pc = PointCloud::from_vec(&data_32);
    //         pc.scale_to_unit_sphere();
    //         counter += 1;

    //         let mut file =
    //             File::create(format!("dataset/neuschwanstein/octant_{}.ply", counter)).unwrap();

    //         let mut ply = Ply::<punctum::Vertex<f32>>::new();
    //         let mut elm_def = punctum::Vertex::<f32>::element_def("vertex".to_string());
    //         elm_def.count = data_32.len();
    //         ply.header.encoding = Encoding::BinaryLittleEndian;
    //         ply.header.elements.add(elm_def.clone());

    //         w.write_header(&mut file, &ply.header).unwrap();
    //         w.write_payload_of_element(&mut file, &data_32, &elm_def, &ply.header)
    //             .unwrap();
    //     }
    // });
}

/// Tee (T-Split) Writer writes the same data to two child writers.
struct TeeWriter<'a, T1: Write + 'a> {
    w1: &'a mut T1,
    w2: &'a mut ProgressBar<Stdout>,
    counter: u64,
}

impl<'a, T1> TeeWriter<'a, T1>
where
    T1: Write + 'a,
{
    fn new(w1: &'a mut T1, w2: &'a mut ProgressBar<Stdout>) -> Self {
        Self { w1, w2, counter: 0 }
    }
}

impl<'a, T1> Write for TeeWriter<'a, T1>
where
    T1: Write + 'a,
{
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let size1 = self.w1.write(buf)?;
        self.counter += size1 as u64;
        // update only every 10Mb
        if self.counter >= 1024 * 1024 * 10 {
            self.w2.add(self.counter);
            self.counter = 0;
        }
        Ok(size1)
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        self.w2.add(self.counter);
        self.counter = 0;
        self.w1.flush()
    }
}
