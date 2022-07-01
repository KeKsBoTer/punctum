use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, BufWriter, Read},
    mem,
    path::{Path, PathBuf},
};

use nalgebra::Vector4;
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

pub fn load_raw_coefs<P: AsRef<Path>>(path: P) -> io::Result<HashMap<u64, Vec<Vector4<f32>>>> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();

    reader.read_to_end(&mut buffer)?;

    let id_size = mem::size_of::<u64>();
    let f_size = mem::size_of::<f32>();
    let color_channels = 4;
    let l = 10;
    let coefs: HashMap<u64, Vec<Vector4<f32>>> = buffer
        .chunks_exact(id_size + f_size * (l + 1) * (l + 1) * color_channels)
        .map(|bytes| {
            let mut a: [u8; 8] = Default::default();
            a.copy_from_slice(&bytes[0..8]);
            let id = u64::from_le_bytes(a);

            let mut af: [u8; 4] = Default::default();
            let coefs = bytes[8..]
                .chunks_exact(f_size * 4)
                .map(|bytes| {
                    af.copy_from_slice(&bytes[0..4]);
                    let r = f32::from_le_bytes(af);
                    af.copy_from_slice(&bytes[4..8]);
                    let g = f32::from_le_bytes(af);
                    af.copy_from_slice(&bytes[8..12]);
                    let b = f32::from_le_bytes(af);
                    af.copy_from_slice(&bytes[12..16]);
                    let a = f32::from_le_bytes(af);
                    Vector4::new(r, g, b, a)
                })
                .collect::<Vec<Vector4<f32>>>();
            return (id, coefs);
        })
        .collect();
    Ok(coefs)
}
