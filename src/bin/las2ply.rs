use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

use las::{Read, Reader};
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer as PlyWriter,
};
use punctum::Vertex;
use rayon::prelude::*;
use std::env;

fn main() {
    let reader =
        Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las").unwrap();

    let number_of_points = reader.header().number_of_points();
    println!("num points: {}", number_of_points);

    let ply = {
        let mut ply = Ply::<Vertex>::new();
        ply.header.encoding = Encoding::Ascii;

        let mut def = Vertex::element_def("vertex".to_string());
        def.count = number_of_points as usize;
        ply.header.elements.add(def);
        ply
    };

    let num_readers = 8;

    let tmp_files: Vec<PathBuf> = (0..num_readers)
        .map(|i| env::temp_dir().join(&Path::new(&format!("las2ply_{}.ply", i))))
        .collect();

    tmp_files
        .clone()
        .into_par_iter()
        .enumerate()
        .for_each(|(file_i, tmp_file_name)| {
            println!("{}", tmp_file_name.to_str().unwrap());
            let file = File::create(tmp_file_name).unwrap();

            let mut writer = BufWriter::new(file);

            let mut reader =
                Reader::from_path("/home/niedermayr/Downloads/3DRM_neuschwanstein_original.las")
                    .unwrap();
            reader
                .seek(file_i as u64 * number_of_points / num_readers)
                .unwrap();

            let mut buffer = Vec::with_capacity(1000000);

            let num_points = if file_i == num_readers as usize - 1 {
                number_of_points
            } else {
                number_of_points / num_readers
            };

            for (i, wrapped_point) in reader.points().take(num_points as usize).enumerate() {
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
                    for v in buffer.iter() {
                        writer
                            .write_fmt(format_args!(
                                "{} {} {} {} {} {} {}\n",
                                v.position[0],
                                v.position[1],
                                v.position[2],
                                (v.color[0] * 255.) as u8,
                                (v.color[1] * 255.) as u8,
                                (v.color[2] * 255.) as u8,
                                (v.color[3] * 255.) as u8
                            ))
                            .unwrap();
                    }
                    buffer.clear();
                    println!(
                        "prgress: {:}%",
                        (100. * i as f32) / (number_of_points as f32 / num_readers as f32),
                    );
                }
            }
        });
    let mut file = File::create("test.ply").unwrap();
    let ply_writer = PlyWriter::<Vertex>::new();
    ply_writer.write_header(&mut file, &ply.header).unwrap();

    Command::new("sh")
        .arg("-c")
        .arg(format!(
            "cat {} >> test.ply",
            tmp_files
                .iter()
                .map(|p| p.as_os_str().to_str().unwrap())
                .collect::<Vec<&str>>()
                .join(" ")
        ))
        .output()
        .expect("failed to execute process");
}
