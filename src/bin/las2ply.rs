use std::{
    fs::File,
    io::{BufWriter, Write},
};

use las::{Read, Reader};
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer as PlyWriter,
};
use punctum::Vertex;

fn write_vertex(vertices: &Vec<Vertex>, writer: &mut BufWriter<File>) {
    for v in vertices.iter() {
        writer.write(&v.position[0].to_le_bytes()).unwrap();
        writer.write(&v.position[1].to_le_bytes()).unwrap();
        writer.write(&v.position[2].to_le_bytes()).unwrap();
        writer
            .write(&((v.color[0] * 255.) as u8).to_le_bytes())
            .unwrap();
        writer
            .write(&((v.color[1] * 255.) as u8).to_le_bytes())
            .unwrap();
        writer
            .write(&((v.color[2] * 255.) as u8).to_le_bytes())
            .unwrap();
        writer
            .write(&((v.color[3] * 255.) as u8).to_le_bytes())
            .unwrap();
    }
}

fn main() {
    let mut file = File::create("test.ply").unwrap();
    let reader =
        Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las").unwrap();

    let number_of_points = reader.header().number_of_points();
    println!("num points: {}", number_of_points);

    let points_to_write = number_of_points; //number_of_points / 32 - (number_of_points / 32) / 10;

    let ply = {
        let mut ply = Ply::<Vertex>::new();
        ply.header.encoding = Encoding::BinaryLittleEndian;

        let mut def = Vertex::element_def("vertex".to_string());
        def.count = points_to_write as usize;
        ply.header.elements.add(def);
        ply
    };

    let ply_writer = PlyWriter::<Vertex>::new();
    ply_writer.write_header(&mut file, &ply.header).unwrap();

    let mut reader =
        Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las").unwrap();

    let mut buffer = Vec::with_capacity(1000000);

    let mut writer = BufWriter::with_capacity(60 * buffer.capacity(), file);
    let mut points_written = 0;
    for (i, wrapped_point) in reader.points().enumerate() {
        let point = wrapped_point.unwrap();
        let color = point.color.unwrap();

        buffer.push(Vertex {
            position: [point.x as f32, point.y as f32, point.z as f32],
            normal: [0.; 3],
            color: [
                (color.red as f32) / 65536.,
                (color.green as f32) / 65536.,
                (color.blue as f32) / 65536.,
                1.,
            ],
        });
        if buffer.len() == buffer.capacity() {
            write_vertex(&buffer, &mut writer);
            buffer.clear();
            println!(
                "prgress: {:}%",
                (100. * i as f32) / (number_of_points as f32),
            );
        }
        points_written += 1;
        if points_written >= points_to_write {
            writer.flush().unwrap();
            break;
        }
    }
    if !buffer.is_empty() {
        write_vertex(&buffer, &mut writer);
    }
    println!("points written: {}", points_written);
}
