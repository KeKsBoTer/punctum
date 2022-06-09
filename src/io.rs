use std::{fs::File, io::BufWriter, path::PathBuf};

use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};

use crate::{
    vertex::{BaseColor, BaseFloat},
    PointCloud, Vertex,
};

pub fn export_ply<F: BaseFloat, C: BaseColor>(output_file: &PathBuf, pc: &PointCloud<F, C>) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<Vertex<F, C>>::new();
    let mut elm_def = Vertex::<F, C>::element_def("vertex".to_string());
    elm_def.count = pc.points().len();
    ply.header.encoding = Encoding::BinaryLittleEndian;
    ply.header.elements.add(elm_def.clone());

    let w = Writer::<Vertex<F, C>>::new();
    w.write_header(&mut file, &ply.header).unwrap();
    w.write_payload_of_element(
        &mut file,
        pc.points(),
        ply.header.elements.get("vertex").unwrap(),
        &ply.header,
    )
    .unwrap();
}