use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, BufWriter, Read},
    mem,
    path::{Path, PathBuf},
};

use bincode::{serialize_into, serialized_size};
use nalgebra::Vector4;
use pbr::ProgressBar;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};

use crate::{
    camera::Projection,
    vertex::{BaseColor, BaseFloat},
    Camera, Octree, PointCloud, TeeReader, TeeWriter, Vertex,
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

pub fn load_octree_with_progress_bar<P: AsRef<Path>>(path: P) -> io::Result<Octree<f64, u8>> {
    let p = path.as_ref();
    let filename = p.to_str().unwrap();
    let in_file = File::open(p)?;

    let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

    let mut buf = BufReader::new(in_file);

    pb.message(&format!("decoding {}: ", filename));

    pb.set_units(pbr::Units::Bytes);
    let mut tee = TeeReader::new(&mut buf, &mut pb);

    let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();
    return Ok(octree);
}

pub fn save_octree_with_progress_bar<P: AsRef<Path>>(
    path: P,
    octree: &Octree<f64, u8>,
) -> io::Result<()> {
    let mut pb = ProgressBar::new(serialized_size(&octree).unwrap());
    pb.set_units(pbr::Units::Bytes);

    let out_file = File::create(path)?;
    let mut out_writer = BufWriter::new(&out_file);
    let mut tee = TeeWriter::new(&mut out_writer, &mut pb);
    serialize_into(&mut tee, octree).unwrap();

    pb.finish_println("done!");
    Ok(())
}

pub fn load_cameras<P: AsRef<Path>, C: Projection + Clone>(
    path: P,
    proj: C,
) -> io::Result<Vec<Camera<C>>> {
    let mut f = std::fs::File::open(path)?;

    // create a parser
    let p = ply_rs::parser::Parser::<Vertex<f32, f32>>::new();

    // use the parser: read the entire file
    let in_ply = p.read_ply(&mut f).unwrap();

    Ok(in_ply
        .payload
        .get("vertex")
        .unwrap()
        .clone()
        .iter()
        .map(|c| Camera::on_unit_sphere(c.position.into(), proj.clone()))
        .collect::<Vec<Camera<C>>>())
}
