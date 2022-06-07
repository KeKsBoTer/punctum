use std::{fs::File, io::BufWriter, path::PathBuf};

use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};

use crate::{PointCloud, Vertex};

pub fn export_ply(output_file: &PathBuf, pc: &PointCloud<f32, u8>) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<Vertex<f32, u8>>::new();
    let mut elm_def = Vertex::<f32, u8>::element_def("vertex".to_string());
    elm_def.count = pc.points().len();
    ply.header.encoding = Encoding::Ascii;
    ply.header.elements.add(elm_def.clone());

    let w = Writer::<Vertex<f32, u8>>::new();
    w.write_header(&mut file, &ply.header).unwrap();
    w.write_payload_of_element(
        &mut file,
        pc.points(),
        ply.header.elements.get("vertex").unwrap(),
        &ply.header,
    )
    .unwrap();
}
